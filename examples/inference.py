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

"""Base model inference script."""

from dataclasses import dataclass
from pathlib import Path
from cosmos_predict2.config import SetupArguments, InferenceArguments, init_script
import tyro


@dataclass
class Args:
    params_file: Path
    """Path to the inference parameters file."""
    setup: tyro.conf.OmitArgPrefixes[SetupArguments]
    """Setup arguments."""


def main(
    args: Args,
):
    inference_args = InferenceArguments.from_file(args.params_file)

    from cosmos_predict2._src.imaginaire.utils import log

    log.info(f"{args.setup}")
    log.info(f"{inference_args}")

    from cosmos_predict2.inference import Inference

    inference = Inference(args.setup)
    inference.generate(inference_args, output_dir=args.setup.output_dir)
    inference.cleanup()


if __name__ == "__main__":
    init_script()

    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.PositionalRequiredArgs,))
    main(args)
