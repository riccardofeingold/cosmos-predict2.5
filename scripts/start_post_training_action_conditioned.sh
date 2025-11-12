export HF_HOME=/data/huggingface
export IMAGINAIRE_OUTPUT_ROOT=/data/cosmos_predict2.5/imaginaire4-output

read -p "Give experiment name: " EXP
read -p "Which CUDA device to use (e.g., 0,1,2,3)? " DEVICE_ID
read -p "Which port to use for torchrun (e.g., 12341)? " PORT
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

# the ~dataloader_train.dataloaders tells hydra to remove the defaults entry for dataloader_train.dataloaders
torchrun --nproc_per_node=$(echo $DEVICE_ID | awk -F, '{print NF}') --master_port=$PORT -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- experiment=$EXP ~dataloader_train.dataloaders