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
from pathlib import Path

from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
import torch
from loguru import logger
import numpy as np

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_predict2.config import SetupArguments, InferenceArguments
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference


class Inference:
    """Base model inference."""

    def __init__(self, args: SetupArguments):
        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.rank0 = distributed.is_rank0()
        self.setup_args = args
        self.pipe = Video2WorldInference(
            experiment_name=args.experiment,
            ckpt_path=args.checkpoint_path,
            s3_credential_path="",
            context_parallel_size=args.context_parallel_size,
        )
        if self.rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            LazyConfig.save_yaml(self.pipe.config, args.output_dir / "config.yaml")

        if self.rank0 and args.enable_guardrails:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            self.text_guardrail_runner = None
            self.video_guardrail_runner = None

    def generate(self, args: InferenceArguments, output_dir: Path):
        if self.rank0:
            logger.info(f"Running {args.inference_type} generation")
            output_dir.mkdir(parents=True, exist_ok=True)

        for i_sample, (sample_name, sample) in enumerate(args.samples.items()):
            if self.rank0:
                logger.info(f"[{i_sample + 1}/{len(args.samples)}] Processing sample {sample_name}")
            if sample.input_path is not None:
                input_path = str(sample.input_path)
            else:
                input_path = ""

            # run text guardrail on the prompt
            if self.rank0:
                if self.text_guardrail_runner is not None:
                    logger.info("Running guardrail check on prompt...")
                    if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):
                        logger.critical("Guardrail blocked text2world generation. Prompt: {sample.prompt}")
                        exit(1)
                    else:
                        logger.success("Passed guardrail on prompt")
                elif self.text_guardrail_runner is None:
                    logger.warning("Guardrail checks on prompt are disabled")

            video: torch.Tensor = self.pipe.generate_vid2world(
                prompt=sample.prompt,
                input_path=input_path,
                guidance=args.guidance,
                num_video_frames=args.num_output_frames,
                num_latent_conditional_frames=args.num_input_frames,
                resolution=args.resolution,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
            )
            if self.rank0:
                output_path = output_dir / sample_name
                video = (1.0 + video[0]) / 2

                # run video guardrail on the video
                if self.video_guardrail_runner is not None:
                    logger.info("Running guardrail check on video...")
                    frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                    if processed_frames is None:
                        logger.critical("Guardrail blocked video2world generation.")
                        exit(1)
                    else:
                        logger.success("Passed guardrail on generated video")
                    # Convert processed frames back to tensor format
                    processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                    video = processed_video.to(video.device, dtype=video.dtype)
                else:
                    logger.warning("Guardrail checks on video are disabled")

                save_img_or_video(video, str(output_path), fps=16)
                logger.info(f"Saved video for {sample_name} to {output_path}.mp4")

        distributed.barrier()

    def cleanup(self):
        self.pipe.cleanup()
