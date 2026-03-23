export HF_HOME=/data/huggingface
read -p "Which CUDA device to use (e.g., 0,1,2,3)? " DEVICE_ID
export CUDA_VISIBLE_DEVICES=$DEVICE_ID
read -p "Give experiment name: " EXP
CHECKPOINTS_DIR=/data/cosmos_predict2.5/imaginaire4-output/cosmos_predict2_action_conditioned/cosmos_predict_action_conditioned/$EXP/checkpoints
CHECKPOINT_ITER=iter_000069000
# CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER
CUDA_VISIBLE_DEVICES=$DEVICE_ID python examples/action_conditioned.py -i assets/action_conditioned/orca/inference_params_one_sample.json -o outputs/action_conditioned/orca_one_sample --checkpoint-path $CHECKPOINT_DIR/model_ema_fp32.pt --experiment $EXP
