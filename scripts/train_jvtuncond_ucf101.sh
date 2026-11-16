GPUS=8
BATCH_SIZE=32  # global=8*32=256
NUM_WORKERS=4

DATA=ucf101
NAME=jvtuncond_ucf101
MAX_EPOCHES=137
SAVE_FREQ=4
SAVE_LAST_FREQ=1

OUTPUT_DIR=outputs
RESUME=$OUTPUT_DIR/$NAME/checkpoint_last.safetensors

torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS \
    train.py \
    dataset=$DATA \
    dataset.num_classes=0 \
    experiment=train_jvtuncond \
    name=$NAME \
    output_dir=$OUTPUT_DIR \
    trainer.max_epoches=$MAX_EPOCHES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.save_last_freq=$SAVE_LAST_FREQ \
    trainer.data.batch_size=$BATCH_SIZE \
    trainer.data.num_workers=$NUM_WORKERS \
    trainer.resume=$resume
