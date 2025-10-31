export HF_HOME=/data/huggingface
export IMAGINAIRE_OUTPUT_ROOT=/data/cosmos_predict2.5/imaginaire4-output
export EXP=cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_orca_frame_320_256_
# the ~dataloader_train.dataloaders tells hydra to remove the defaults entry for dataloader_train.dataloaders
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- experiment=$EXP ~dataloader_train.dataloaders