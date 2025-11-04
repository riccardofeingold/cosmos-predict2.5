export HF_HOME=/data/huggingface
export CUDA_VISIBLE_DEVICES=3
EXP=cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_orca_frame_320_256_one_sample_dataset
CHECKPOINTS_DIR=/data/cosmos_predict2.5/imaginaire4-output/cosmos_predict2_action_conditioned/cosmos_predict_action_conditioned/$EXP/checkpoints
# CHECKPOINT_ITER=iter_000020000
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER
CUDA_VISIBLE_DEVICES=0 python examples/action_conditioned.py -i assets/action_conditioned/orca/inference_params_one_sample.json -o outputs/action_conditioned/orca --checkpoint-path $CHECKPOINT_DIR/model.pt --experiment $EXP
