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
import gc
import torch

from cosmos_gradio.gradio_app.gradio_app import GradioApp
from cosmos_gradio.gradio_app.gradio_ui import create_gradio_UI
from cosmos_gradio.deployment_env import DeploymentEnv
from gradio_app.model_config import ModelConfig
from cosmos_predict2.text2image import Text2Image_Worker, Text2Image_Params
from cosmos_predict2.video2world import Video2World_Worker, Video2World_Params
from cosmos_predict2.multiview_worker import Multiview_Worker, Multiview_Params
from cosmos_predict2._src.imaginaire.utils import log


def create_text2image():
    log.info("Creating predict pipeline and validator")
    global_env = DeploymentEnv()
    pipeline = Text2Image_Worker(checkpoint_dir=global_env.checkpoint_dir)
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def create_video2world():
    log.info("Creating predict pipeline and validator")
    global_env = DeploymentEnv()
    pipeline = Video2World_Worker(
        num_gpus=global_env.num_gpus,
        checkpoint_dir=global_env.checkpoint_dir,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def create_multiview():
    log.info("Creating predict pipeline and validator")
    global_env = DeploymentEnv()
    assert global_env.num_gpus == 8, "Multiview currently requires 8 GPUs"
    pipeline = Multiview_Worker(
        num_gpus=global_env.num_gpus,
        checkpoint_dir=global_env.checkpoint_dir,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


if __name__ == "__main__":
    model_cfg = ModelConfig()
    global_env = DeploymentEnv()

    # configure server to use the correct worker in the worker procs
    factory_module = {
        "text2image": "gradio_app.gradio_bootstrapper",
        "video2world": "gradio_app.gradio_bootstrapper",
        "multiview": "gradio_app.gradio_bootstrapper",
    }

    factory_function = {
        "text2image": "create_text2image",
        "video2world": "create_video2world",
        "multiview": "create_multiview",
    }

    validators = {
        "text2image": Text2Image_Params.validate_kwargs,
        "video2world": Video2World_Params.validate_kwargs,
        "multiview": Multiview_Params.validate_kwargs,
    }
    global_env = DeploymentEnv()

    log.info(f"Starting Gradio app with deployment config: {global_env!s}")

    app = GradioApp(
        num_gpus=global_env.num_gpus,
        validator=validators[global_env.model_name],
        factory_module=factory_module[global_env.model_name],
        factory_function=factory_function[global_env.model_name],
        output_dir=global_env.output_dir,
    )

    interface = create_gradio_UI(
        app.infer,
        header=model_cfg.header[global_env.model_name],
        default_request=model_cfg.default_request[global_env.model_name],
        help_text=model_cfg.help_text[global_env.model_name],
        uploads_dir=global_env.uploads_dir,
        output_dir=global_env.output_dir,
        log_file=global_env.log_file,
    )

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=[global_env.output_dir, global_env.uploads_dir],
    )
