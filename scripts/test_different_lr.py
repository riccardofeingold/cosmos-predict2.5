# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Learning rate finder for LoRA fine-tuning.
Usage: torchrun --nproc_per_node=1 scripts/find_lr.py --config=path/to/config.py -- experiments=your_experiment
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("HF_HOME", "/data/huggingface")  # Set HuggingFace cache directory

import argparse
import importlib
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from loguru import logger as logging
from tqdm import tqdm

from cosmos_predict2._src.imaginaire.config import Config
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp


def find_learning_rate(
    model,
    dataloader,
    optimizer_config,
    device,
    start_lr=1e-8,
    end_lr=1e-3,
    num_iter=200,
    output_dir="outputs/lr_finder",
    grad_clip=1.0,
):
    """
    Learning rate range test for DiT models.
    
    Args:
        model: The model to test
        dataloader: Training dataloader
        optimizer_config: Optimizer configuration from config
        device: Device to run on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations to test
        output_dir: Where to save results
        grad_clip: Gradient clipping value
    """
    from torch.optim import AdamW
    
    # Create optimizer with starting LR
    # Get parameters that require grad (for LoRA, only LoRA params should require grad)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = AdamW(trainable_params, lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    lrs = []
    losses = []
    smoothed_losses = []
    best_loss = float('inf')
    avg_loss = 0
    beta = 0.98  # Smoothing factor
    
    model.train()
    
    logging.info(f"Running LR finder from {start_lr:.2e} to {end_lr:.2e} over {num_iter} iterations...")
    
    dataloader_iter = iter(dataloader)
    
    for i in tqdm(range(num_iter), desc="LR Finder"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            logging.warning("Dataloader exhausted, restarting...")
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Move batch to device if needed
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        else:
            batch = batch.to(device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Forward pass
        try:
            if isinstance(batch, dict):
                outputs = model(**batch)
            else:
                outputs = model(batch)
            
            # Extract loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                loss = outputs
                
        except Exception as e:
            logging.error(f"Error during forward pass at iteration {i}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            break
        
        current_loss = loss.item()
        
        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * current_loss
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))
        
        # Track
        lrs.append(current_lr)
        losses.append(current_loss)
        smoothed_losses.append(smoothed_loss)
        
        # Stop if loss explodes (divergence)
        if smoothed_loss > 4 * best_loss and i > 10:
            logging.warning(f"Stopping early at iteration {i} - loss diverged")
            break
        
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        
        optimizer.step()
        
        # Update learning rate
        optimizer.param_groups[0]['lr'] *= lr_mult
        
        if (i + 1) % 20 == 0:
            logging.info(f"Step {i+1}/{num_iter} | LR: {current_lr:.2e} | Loss: {current_loss:.4f} | Smoothed: {smoothed_loss:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(16, 6))
    
    # Plot 1: Loss vs LR (log scale)
    plt.subplot(1, 3, 1)
    plt.plot(lrs, smoothed_losses, label='Smoothed Loss', linewidth=2)
    plt.plot(lrs, losses, alpha=0.3, label='Raw Loss')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Find suggested LR (steepest descent on smoothed loss)
    suggested_lr = None
    recommended_lr = None
    
    if len(smoothed_losses) > 20:
        # Find steepest gradient
        gradients = [(smoothed_losses[i+1] - smoothed_losses[i]) / (lrs[i+1] - lrs[i]) 
                     for i in range(len(smoothed_losses)-1)]
        min_gradient_idx = gradients.index(min(gradients))
        suggested_lr = lrs[min_gradient_idx]
        recommended_lr = suggested_lr / 10  # Conservative recommendation
        
        plt.axvline(suggested_lr, color='r', linestyle='--', 
                   label=f'Steepest descent: {suggested_lr:.2e}', linewidth=2)
        plt.axvline(recommended_lr, color='g', linestyle='--', 
                   label=f'Recommended (÷10): {recommended_lr:.2e}', linewidth=2)
        plt.legend()
    
    # Plot 2: Loss vs iteration
    plt.subplot(1, 3, 2)
    plt.plot(range(len(losses)), losses, alpha=0.5, label='Raw Loss')
    plt.plot(range(len(smoothed_losses)), smoothed_losses, label='Smoothed Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Gradient of loss vs LR
    if len(smoothed_losses) > 1:
        plt.subplot(1, 3, 3)
        gradients_plot = [(smoothed_losses[i+1] - smoothed_losses[i]) 
                          for i in range(len(smoothed_losses)-1)]
        plt.plot(lrs[:-1], gradients_plot, linewidth=2)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss Gradient')
        plt.title('Loss Gradient (lower = steeper descent)')
        plt.grid(True, alpha=0.3)
        if suggested_lr:
            plt.axvline(suggested_lr, color='r', linestyle='--', label='Steepest', linewidth=2)
            plt.legend()
    
    plt.tight_layout()
    plot_path = f"{output_dir}/lr_finder.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_path}")
    
    # Save data
    results = {
        'lrs': [float(lr) for lr in lrs],
        'losses': [float(loss) for loss in losses],
        'smoothed_losses': [float(loss) for loss in smoothed_losses],
        'suggested_lr': float(suggested_lr) if suggested_lr else None,
        'recommended_lr': float(recommended_lr) if recommended_lr else None,
        'params': {
            'start_lr': start_lr,
            'end_lr': end_lr,
            'num_iter': num_iter,
            'grad_clip': grad_clip,
        }
    }
    
    json_path = f"{output_dir}/lr_finder_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {json_path}")
    
    # Print summary
    logging.info("\n" + "="*70)
    logging.info("LEARNING RATE FINDER RESULTS")
    logging.info("="*70)
    if suggested_lr:
        logging.info(f"Steepest descent at LR:      {suggested_lr:.2e}")
        logging.info(f"RECOMMENDED starting LR:     {recommended_lr:.2e}")
        logging.info(f"Conservative range:          {recommended_lr/2:.2e} - {recommended_lr*2:.2e}")
    else:
        logging.warning("Could not determine optimal LR. Try adjusting the range.")
    logging.info("="*70 + "\n")
    
    return results


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    """Launch the LR finder."""
    
    # Initialize distributed environment (required for Megatron parallel state)
    # This sets up the data parallel group even for single GPU
    distributed.init()
    
    logging.info(f"Distributed initialized - Rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")
    
    device = torch.device(f'cuda:{distributed.get_rank()}')
    logging.info(f"Running LR finder on device: {device}")
    
    # Validate config
    config.validate()
    
    # Create the model
    logging.info("Creating model...")
    if isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt"):
        model = create_model_from_consolidated_checkpoint_with_fsdp(config)
    else:
        model = instantiate(config.model)
    
    # Create the dataloader
    logging.info("Creating dataloader...")
    dataloader = instantiate(config.dataloader_train)
    
    # Model should already be on correct device after instantiation with FSDP
    # but ensure it's in training mode
    model.to(device, memory_format=config.trainer.memory_format)
    model.on_train_start()

    # Run LR finder only on rank 0
    if distributed.get_rank() == 0:
        output_dir = Path(config.job.path_local) / "lr_finder"
        
        results = find_learning_rate(
            model=model,
            dataloader=dataloader,
            optimizer_config=config.optimizer,
            device=device,
            start_lr=args.start_lr,
            end_lr=args.end_lr,
            num_iter=args.num_iter,
            output_dir=str(output_dir),
            grad_clip=getattr(config.trainer, 'grad_clip_norm', 1.0),
        )
        
        logging.info(f"LR finder complete! Check results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Rate Finder")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "--start_lr", 
        type=float, 
        default=1e-8,
        help="Starting learning rate for the search"
    )
    parser.add_argument(
        "--end_lr", 
        type=float, 
        default=1e-3,
        help="Ending learning rate for the search"
    )
    parser.add_argument(
        "--num_iter", 
        type=int, 
        default=200,
        help="Number of iterations to run"
    )
    parser.add_argument(
        "opts",
        help="Modify config options (same format as train.py)",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    
    # Load config (same as train.py)
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    
    # Override config options
    overrides = list(args.opts)
    # Force some settings for LR finding
    overrides.extend([
        "job.wandb_mode=disabled",  # Disable wandb for LR finding
        "experiment=cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_orca_frame_320_256_one_sample_dataset_with_lora",
        # "~dataloader_train.dataloaders"
    ])
    config = override(config, overrides)
    
    # Launch LR finder
    launch(config, args)