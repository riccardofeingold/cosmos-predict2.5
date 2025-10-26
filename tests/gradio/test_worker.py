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

import json
import os

from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.model_ipc.model_server import ModelServer
from loguru import logger as log

from cosmos_predict2.config import InferenceArguments, SetupArguments
from cosmos_predict2.gradio.sample_data import sample_request_image2world, sample_request_multiview

"""
compare to
COSMOS_INTERNAL=0 PYTHONPATH=. torchrun --nproc_per_node=8 examples/inference.py assets/base/image2world.json outputs/image2world
COSMOS_INTERNAL=0 PYTHONPATH=. torchrun --nproc_per_node=8 examples/multiview.py assets/multiview/multiview.json
"""

global_env = DeploymentEnv()


def test_video2world_args():
    log.info(json.dumps(SetupArguments.model_json_schema(), indent=2))
    log.info(json.dumps(InferenceArguments.model_json_schema(), indent=2))
    model_params = InferenceArguments(**sample_request_image2world)
    log.info(json.dumps(model_params.model_dump(mode="json"), indent=2))


def test_multiview_args():
    from cosmos_predict2.multiview_config import MultiviewInferenceArguments, MultiviewSetupArguments

    log.info(json.dumps(MultiviewSetupArguments.model_json_schema(), indent=2))
    log.info(json.dumps(MultiviewInferenceArguments.model_json_schema(), indent=2))
    model_params = MultiviewInferenceArguments(**sample_request_multiview)
    log.info(json.dumps(model_params.model_dump(mode="json"), indent=2))


def test_video2world():
    from cosmos_predict2.gradio.video2world_worker import Video2World_Worker

    model_params = InferenceArguments(**sample_request_image2world)
    pipeline = Video2World_Worker(num_gpus=1)

    model_params = model_params.model_dump()
    model_params["output_dir"] = "outputs/predict2/v2w/"
    pipeline.infer(model_params)


def test_multiview():
    from cosmos_predict2.multiview_config import MultiviewInferenceArgumentsWithInputPaths

    with ModelServer(
        num_gpus=8,
        factory_module="cosmos_predict2.gradio.gradio_bootstrapper",
        factory_function="create_multiview",
    ) as pipeline:
        model_params = MultiviewInferenceArgumentsWithInputPaths(**sample_request_multiview)
        model_params = model_params.model_dump(mode="json")
        model_params["output_dir"] = "outputs/predict2/mv/"
        pipeline.infer(model_params)


if __name__ == "__main__":
    log.info(f"test_worker current dir={os.getcwd()}")
    log.info(global_env)
    if global_env.model_name == "video2world":
        test_video2world()
    elif global_env.model_name == "multiview":
        test_multiview()
    else:
        test_video2world_args()
        test_multiview_args()
