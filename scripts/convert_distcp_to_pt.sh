# Get path to the latest checkpoint
read -p "Give experiment name: " EXP
CHECKPOINTS_DIR=/data/cosmos_predict2.5/imaginaire4-output/cosmos_predict2_action_conditioned/cosmos_predict_action_conditioned/$EXP/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR