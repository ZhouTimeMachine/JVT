import json
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from typing import Any, Dict, Tuple


def save_model_safetensors(state_dict: dict, path: str, metadata: dict = None):
    """Save model state dict as safetensors with scalar metadata."""
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    save_file(state_dict, path, metadata=meta)


def load_model_safetensors(path: str) -> Tuple[dict, dict]:
    """Load model state dict and metadata from safetensors."""
    tensors = load_file(path, device='cpu')
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
    return tensors, metadata


def save_training_state_safetensors(
    optimizer_state: dict,
    lr_scheduler_state: dict,
    path: str,
):
    """Save optimizer + lr_scheduler as a single safetensors file.

    Tensors are stored as safetensors entries; all scalar / JSON-serialisable
    data goes into the metadata header.
    """
    tensors, metadata = _flatten_optimizer_state(optimizer_state)
    metadata['lr_scheduler'] = json.dumps(_to_json_serializable(lr_scheduler_state))
    save_file(tensors, path, metadata=metadata)


def load_training_state_safetensors(path: str) -> Tuple[dict, dict]:
    """Load optimizer state dict and lr_scheduler state dict from safetensors.

    Returns (optimizer_state_dict, lr_scheduler_state_dict).
    """
    tensors = load_file(path, device='cpu')
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
    optimizer_state = _unflatten_optimizer_state(tensors, metadata)
    lr_scheduler_state = json.loads(metadata['lr_scheduler'])
    return optimizer_state, lr_scheduler_state


# ---------------------------------------------------------------------------
# Internal: flatten / unflatten optimizer state dicts
# ---------------------------------------------------------------------------

def _flatten_optimizer_state(optimizer_state: dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    tensors: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, str] = {}

    is_ema = 'opt' in optimizer_state

    if is_ema:
        metadata['__type'] = 'ema'
        metadata['current_step'] = str(optimizer_state['current_step'])
        metadata['decay'] = str(optimizer_state['decay'])
        metadata['every_n_steps'] = str(optimizer_state['every_n_steps'])

        for i, param in enumerate(optimizer_state['ema']):
            tensors[f'ema.{i}'] = param.contiguous()
        metadata['ema_count'] = str(len(optimizer_state['ema']))

        _flatten_inner_optimizer(optimizer_state['opt'], tensors, metadata, prefix='opt.')
    else:
        metadata['__type'] = 'regular'
        _flatten_inner_optimizer(optimizer_state, tensors, metadata, prefix='')

    return tensors, metadata


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert OmegaConf containers and other non-serializable types to plain Python."""
    try:
        from omegaconf import DictConfig, ListConfig
        if isinstance(obj, DictConfig):
            return {k: _to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, ListConfig):
            return [_to_json_serializable(v) for v in obj]
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def _flatten_inner_optimizer(opt_state: dict, tensors: dict, metadata: dict, prefix: str):
    metadata[f'{prefix}param_groups'] = json.dumps(_to_json_serializable(opt_state['param_groups']))

    for param_idx, param_state in opt_state['state'].items():
        for key, value in param_state.items():
            full_key = f'{prefix}state.{param_idx}.{key}'
            if isinstance(value, torch.Tensor):
                tensors[full_key] = value.contiguous()
            else:
                metadata[full_key] = json.dumps(_to_json_serializable(value))


def _unflatten_optimizer_state(tensors: Dict[str, torch.Tensor], metadata: Dict[str, str]) -> dict:
    opt_type = metadata.get('__type', 'regular')

    if opt_type == 'ema':
        ema_count = int(metadata['ema_count'])
        ema_params = tuple(tensors[f'ema.{i}'] for i in range(ema_count))

        inner_opt = _unflatten_inner_optimizer(tensors, metadata, prefix='opt.')

        return {
            'opt': inner_opt,
            'ema': ema_params,
            'current_step': int(metadata['current_step']),
            'decay': float(metadata['decay']),
            'every_n_steps': int(metadata['every_n_steps']),
        }
    else:
        return _unflatten_inner_optimizer(tensors, metadata, prefix='')


def _unflatten_inner_optimizer(tensors: dict, metadata: dict, prefix: str) -> dict:
    param_groups = json.loads(metadata[f'{prefix}param_groups'])

    state: Dict[int, dict] = {}
    state_prefix = f'{prefix}state.'

    param_indices: set = set()
    for key in list(tensors.keys()) + list(metadata.keys()):
        if key.startswith(state_prefix):
            parts = key[len(state_prefix):].split('.', 1)
            try:
                param_indices.add(int(parts[0]))
            except ValueError:
                continue

    for idx in sorted(param_indices):
        state[idx] = {}
        idx_prefix = f'{state_prefix}{idx}.'
        for key, value in tensors.items():
            if key.startswith(idx_prefix):
                state[idx][key[len(idx_prefix):]] = value
        for key, value in metadata.items():
            if key.startswith(idx_prefix):
                state[idx][key[len(idx_prefix):]] = json.loads(value)

    return {'state': state, 'param_groups': param_groups}