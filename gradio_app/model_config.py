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
from dataclasses import dataclass


default_request_t2i = json.dumps(
    {
        "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
        "aspect_ratio": "16,9",
    },
    indent=2,
)

default_request_v2w = json.dumps(
    {
        "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
        "input_path": "packages/cosmos-predict2/assets/video2world/input0.jpg",
        "num_input_frames": 1,
        "guidance": 7,
        "seed": 0,
    },
    indent=2,
)

default_request_mv = json.dumps(
    {
        "input_root": "datasets/car_input",
        "num_input_frames": 0,
        "n_views": 7,
        "stack_mode": "time",
        "fps": 10,
    },
    indent=2,
)

help_text_v2w = """
                    ### Generation Parameters:
                    - `input_path` (string): Path to input image/video file (default: "assets/video2world/input0.jpg")
                    - `prompt` (string): Text description of desired output (default: empty string)
                    - `negative_prompt` (string): What to avoid in generation (default: predefined negative prompt)
                    - `aspect_ratio` (string): Output aspect ratio (default: "16:9")
                    - `num_conditional_frames` (int): Number of conditioning frames (default: 1)
                    - `guidance` (float): Classifier-free guidance scale (default: 7.0)
                    - `seed` (int): Random seed for reproducibility (default: 0)
                """

help_text_t2i = """
                    ### Generation Parameters:
                    - `prompt` (string): Detailed text description of the desired video content and scene (default: predefined positive prompt)
                    - `negative_prompt` (string): Text describing elements to exclude from generation (default: empty string)
                    - `aspect_ratio` (string): Output video aspect ratio in width:height format (default: "16:9")
                    - `seed` (int): Random seed value for reproducible generation results (default: 0)
                """

help_text_mv = """
                    ### Generation Parameters:
                    - `input_root` (string): Path to the directory containing input images (default: "datasets/car_input")
                    - `num_input_frames` (int): Number of input frames to use (choices: 0=text2world, 1=image2world, 2=video2world; default: 0)
                    - `n_views` (int): Number of views to generate (default: 7)
                    - `stack_mode` (string): How to stack input frames ("time" or "height"; default: "time")
                    - `fps` (int): Frames per second for the output video (default: 10)
                    - `guidance` (float): Classifier-free guidance scale (default: 7.0)
                    - `seed` (int): Random seed for reproducibility (default: 1)
                    - `num_steps` (int): Number of diffusion steps (default: 35)
                """


@dataclass
class ModelConfig:
    header = {
        "video2world": "Cosmos-Predict2.5 Video2World",
        "text2image": "Cosmos-Predict2.5 Text2Image",
        "multiview": "Cosmos-Predict2.5 Multiview",
    }

    help_text = {
        "video2world": help_text_v2w,
        "text2image": help_text_t2i,
        "multiview": help_text_mv,
    }

    default_request = {
        "video2world": default_request_v2w,
        "text2image": default_request_t2i,
        "multiview": default_request_mv,
    }
