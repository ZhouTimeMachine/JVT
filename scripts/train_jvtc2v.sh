GPUS=8
BATCH_SIZE=32  # global=8*32=256
NUM_WORKERS=4

NAME=jvtc2v
OUTPUT_DIR=outputs
RESUME=$OUTPUT_DIR/$NAME/checkpoint_last.safetensors

torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS \
    train.py \
    experiment=train_jvtc2v \
    name=$NAME \
    output_dir=$OUTPUT_DIR \
    trainer.save_freq=4 \
    trainer.data.batch_size=$BATCH_SIZE \
    trainer.data.num_workers=$NUM_WORKERS \
    trainer.resume=$RESUME
