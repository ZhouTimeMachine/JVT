GPUS=8
BATCH_SIZE=32  # global=8*32=256
NUM_WORKERS=4

DATA=skytimelapse
NAME=jvtuncond_sky
MAX_EPOCHES=505
SAVE_FREQ=10
SAVE_LAST_FREQ=2

OUTPUT_DIR=outputs
RESUME=$OUTPUT_DIR/$NAME/checkpoint_last.safetensors

torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS \
    train.py \
    dataset=$DATA \
    experiment=train_jvtuncond \
    name=$NAME \
    output_dir=$OUTPUT_DIR \
    trainer.max_epoches=$MAX_EPOCHES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.save_last_freq=$SAVE_LAST_FREQ \
    trainer.data.batch_size=$BATCH_SIZE \
    trainer.data.num_workers=$NUM_WORKERS \
    trainer.resume=$resume
