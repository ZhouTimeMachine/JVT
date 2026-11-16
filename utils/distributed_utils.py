import builtins
import datetime
import os
import copy

import torch
import torch.distributed as dist

from pathlib import Path
from omegaconf import DictConfig

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            kwargs['flush'] = True
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(cfg: DictConfig):
    if cfg.dist_on_itp:
        cfg.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        cfg.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        cfg.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        cfg.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(cfg.gpu)
        os.environ['RANK'] = str(cfg.rank)
        os.environ['WORLD_SIZE'] = str(cfg.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.gpu = int(os.environ['LOCAL_RANK'])

        if 'MASTER_ADDR' in os.environ:
            cfg.dist_url = f"env://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ['SLURM_PROCID'])
        cfg.world_size = int(os.environ['SLURM_NTASKS'])
        # args.gpu = args.rank % torch.cuda.device_count()
        cfg.gpu = int(os.environ['SLURM_LOCALID'])

        if 'MASTER_ADDR' not in os.environ:
            import subprocess
            node_list = os.environ['SLURM_STEP_NODELIST']
            result = subprocess.run(
                ['scontrol', 'show', 'hostname', node_list],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, 'LC_ALL': 'C', 'LANG': 'C'},
            )
            addr = result.stdout.splitlines()[0].strip()
            os.environ['MASTER_ADDR'] = addr

        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'

        cfg.dist_url = f"env://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        cfg.distributed = False
        cfg.rank = 0
        cfg.gpu = 0
        cfg.world_size = 1
        return

    cfg.distributed = True

    requested_backend = cfg.get('dist_backend', 'nccl')
    if cfg.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)
        cfg.dist_backend = requested_backend
    else:
        cfg.device = 'cpu'
        cfg.dist_backend = requested_backend if requested_backend != 'nccl' else 'gloo'

    print(f'| distributed init (rank {cfg.rank}): {cfg.dist_url}, gpu {cfg.gpu}', flush=True)
    torch.distributed.init_process_group(
        backend=cfg.dist_backend, init_method=cfg.dist_url,
        world_size=cfg.world_size, rank=cfg.rank
    )
    if cfg.dist_backend == 'nccl':
        torch.distributed.barrier(device_ids=[cfg.gpu])
    else:
        torch.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)


def add_weight_decay(model, weight_decay=0, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def save_model(args, model_without_ddp, optimizer, epoch, epoch_name=None):
    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }

    # ema
    ema_state_dict1 = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict2 = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict1 and name in ema_state_dict2
        ema_state_dict1[name] = model_without_ddp.ema_params1[i]
        ema_state_dict2[name] = model_without_ddp.ema_params2[i]
    to_save['model_ema1'] = ema_state_dict1
    to_save['model_ema2'] = ema_state_dict2

    save_on_master(to_save, checkpoint_path)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
