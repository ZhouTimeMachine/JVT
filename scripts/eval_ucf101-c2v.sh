GPUS=8

OUTPUT_DIR=outputs
DATA=ucf101
FAKE_VIDEO_DIR=$OUTPUT_DIR/${DATA}-c2v
CHECKPOINT=checkpoints/${DATA}-c2v.safetensors
FVD_DIR=$OUTPUT_DIR/eval_fvd

REAL_VIDEO_DIR=data/$DATA
REAL_SEQ_ID_MAP=evaluation/dataset/seq-id-maps/${DATA}_2048_256x256_seq16_stride3_seed42_id_map.json

torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS} \
    evaluation/infer_c2v.py \
    output_dir=$FAKE_VIDEO_DIR \
    resume=$CHECKPOINT

python evaluation/eval_fvd.py \
    real_video_dir=$REAL_VIDEO_DIR \
    real_seq_id_map=$REAL_SEQ_ID_MAP \
    fake_video_dir=$FAKE_VIDEO_DIR \
    output_dir=$FVD_DIR \
    save_suffix=${DATA}-c2v
