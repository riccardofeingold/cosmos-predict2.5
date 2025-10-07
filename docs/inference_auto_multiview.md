# Auto Multiview Inference Guide

This guide provides instructions on running inference with Cosmos-Predict2.5/auto/multiview models.

We recommend first reading the [Inference Guide](inference.md).

## Prerequisites

1. [Setup Guide](./setup.md)

## Example

Multiview requires multi-GPU.

Run multi-GPU inference with example assets:

```bash
torchrun --nproc_per_node=8 examples/multiview.py assets/multiview/multiview.json outputs/multiview
```

All variants require sample input videos. For Text2World, they are not used. For Image2World, only the first frame is used. For Video2World, the first 2 frames are used.

| Variant | Arguments |
| --- | --- |
| Text2World | `{ "num_input_frames": 0 }` |
| Image2World | `{ "num_input_frames": 1 }` |
| Video2World | `{ "num_input_frames": 2 }` |

### Outputs

#### multiview/text2world

<video src="https://github.com/user-attachments/assets/aae580f5-1379-4416-81ad-c863b51d4cf9" width="500" alt="multiview/text2world" controls></video>
