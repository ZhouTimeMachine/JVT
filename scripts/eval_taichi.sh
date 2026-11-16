GPUS=8

OUTPUT_DIR=outputs
DATA=taichi
FAKE_VIDEO_DIR=$OUTPUT_DIR/$DATA
CHECKPOINT=checkpoints/${DATA}.safetensors
FVD_DIR=$OUTPUT_DIR/eval_fvd

REAL_VIDEO_DIR=data/taichi-train
REAL_SEQ_ID_MAP=evaluation/dataset/seq-id-maps/${DATA}_train2048_256x256_seq16_seed42_id_map.json

# infer 2048 samples
torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS} \
    evaluation/infer_uncond.py \
    output_dir=$FAKE_VIDEO_DIR \
    resume=$CHECKPOINT

# eval fvd with 2048 real samples
python evaluation/eval_fvd.py \
    real_video_dir=$REAL_VIDEO_DIR \
    real_seq_id_map=$REAL_SEQ_ID_MAP \
    fake_video_dir=$FAKE_VIDEO_DIR \
    output_dir=$FVD_DIR \
    save_suffix=$DATA
