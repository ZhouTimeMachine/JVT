GPUS=8

OUTPUT_DIR=outputs
NAME=re10k-rgb
RESULT_DIR=$OUTPUT_DIR/$NAME
FAKE_VIDEO_DIR=$RESULT_DIR/fake
FVD_DIR=$OUTPUT_DIR/eval_fvd
POSE_DIR=$OUTPUT_DIR/eval_pose

CHECKPOINT=checkpoints/${NAME}.safetensors

REAL_VIDEO_DIR=data/re10k/test_256
REAL_SEQ_ID_MAP=evaluation/dataset/seq-id-maps/re10k_test1745_256x256_seq8_id_map.json

torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS \
    evaluation/infer_camctrl.py \
    evaluation=infer_camctrl \
    resume=$CHECKPOINT \
    output_dir=$RESULT_DIR

python evaluation/eval_fvd.py num_frames=8 \
    real_seq_id_map=$REAL_SEQ_ID_MAP \
    real_video_dir=$REAL_VIDEO_DIR \
    fake_video_dir=$FAKE_VIDEO_DIR \
    output_dir=$FVD_DIR \
    save_suffix=$NAME

torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS \
    evaluation/eval_pose_re10k.py \
    dataset.Re10K_EVAL_VIDEO_DIR=$FAKE_VIDEO_DIR \
    save_suffix=$NAME \
    output_dir=$POSE_DIR
