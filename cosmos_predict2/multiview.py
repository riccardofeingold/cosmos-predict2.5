import os
from cosmos_predict2.multiview_config import MultiviewInferenceArguments, MultiviewSetupArguments
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2_multiview.scripts.inference import NUM_CONDITIONAL_FRAMES_KEY, Vid2VidInference
from cosmos_predict2._src.predict2_multiview.datasets.local_dataset import (
    LocalMultiviewAugmentorConfig,
    LocalMultiviewDatasetBuilder,
)
from cosmos_predict2._src.predict2_multiview.configs.vid2vid.defaults.driving import (
    MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION,
)

NUM_DATALOADER_WORKERS = 8


class MultiviewInference:
    def __init__(self, args: MultiviewSetupArguments):
        self.experiment_name = args.experiment
        self.checkpoint_path = args.checkpoint_path
        self.context_parallel_size = args.context_parallel_size
        self.output_dir = args.output_dir

    def generate(self, inference_args: MultiviewInferenceArguments):
        # Enable deterministic inference
        os.environ["NVTE_FUSED_ATTN"] = "0"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.enable_grad(False)  # Disable gradient calculations for inference

        pipe = Vid2VidInference(
            self.experiment_name, self.checkpoint_path, context_parallel_size=self.context_parallel_size
        )

        rank0 = True
        if self.context_parallel_size > 1:
            rank0 = torch.distributed.get_rank() == 0
        if rank0:
            os.makedirs(self.output_dir, exist_ok=True)
            LazyConfig.save_yaml(pipe.config, os.path.join(self.output_dir, "config.yaml"))

        driving_dataloader_config = MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[pipe.config.model.config.resolution]
        driving_dataloader_config.n_views = inference_args.n_views

        dataset = LocalMultiviewDatasetBuilder(inference_args.input_paths).build_dataset(
            LocalMultiviewAugmentorConfig(
                resolution=pipe.config.model.config.resolution,
                driving_dataloader_config=driving_dataloader_config,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_DATALOADER_WORKERS,
        )
        if len(dataloader) == 0:
            raise ValueError("No input data found")

        for i, batch in enumerate(dataloader):
            batch["ai_caption"] = [inference_args.prompt]
            batch[NUM_CONDITIONAL_FRAMES_KEY] = inference_args.num_input_frames
            video = pipe.generate_from_batch(
                batch,
                guidance=inference_args.guidance,
                seed=inference_args.seed,
                stack_mode=inference_args.stack_mode,
                num_steps=inference_args.num_steps,
            )
            if rank0:
                output_path = f"{self.output_dir}/output_{i}"
                save_img_or_video((1.0e0 + video[0]) / 2, output_path, fps=inference_args.fps)
                log.info(f"Saved video to {output_path}.mp4")

        pipe.cleanup()
