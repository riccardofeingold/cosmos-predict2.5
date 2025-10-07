from dataclasses import dataclass
from typing import Literal
import pydantic
from pathlib import Path
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid
from typing_extensions import Self
from typing import Annotated
import enum
import yaml
import json
import os
from functools import cached_property


def init_script():
    """Initialize script environment."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if "RANK" in os.environ:
        from cosmos_predict2._src.imaginaire.utils import distributed

        distributed.init()


def _resolve_path(v: Path) -> Path:
    """Resolve path to absolute."""
    return v.expanduser().resolve()


ResolvedFilePath = Annotated[pydantic.FilePath, pydantic.AfterValidator(_resolve_path)]


def _validate_checkpoint_uuid(v: str) -> str:
    """Validate checkpoint UUID."""
    get_checkpoint_by_uuid(v)
    return v


CheckpointUuid = Annotated[str, pydantic.AfterValidator(_validate_checkpoint_uuid)]


class Arguments(pydantic.BaseModel):
    """Parent class for arguments."""

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Loads the arguments from a json/yaml file."""
        if path.suffix in [".json"]:
            data = json.loads(path.read_text())
        elif path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(path.read_text())
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        # Input paths are relative to the file path
        cwd = os.getcwd()
        os.chdir(path.parent)
        self = cls.model_validate(data)
        os.chdir(cwd)

        return self


class ModelSize(str, enum.Enum):
    _2B = "2B"
    _14B = "14B"

    def __str__(self) -> str:
        return self.value


class ModelVariant(str, enum.Enum):
    BASE = "base"
    AUTO_MULTIVIEW = "auto/multiview"
    ROBOT_MULTIVIEW = "robot/multiview"
    ROBOT_MULTIVIEW_AGIBOT = "robot/multiview-agibot"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, kw_only=True)
class ModelKey:
    post_trained: bool = True
    size: ModelSize = ModelSize._2B
    variant: ModelVariant = ModelVariant.BASE

    @cached_property
    def name(self) -> str:
        parts = [str(self.size)]
        if self.variant == ModelVariant.BASE:
            parts.append("post-trained" if self.post_trained else "pre-trained")
        else:
            parts.append(str(self.variant))
        return "/".join(parts)

    def __str__(self) -> str:
        return self.name


MODEL_CHECKPOINTS = {
    ModelKey(post_trained=False): get_checkpoint_by_uuid("d20b7120-df3e-4911-919d-db6e08bad31c"),
    ModelKey(): get_checkpoint_by_uuid("81edfebe-bd6a-4039-8c1d-737df1a790bf"),
    ModelKey(post_trained=False, size=ModelSize._14B): get_checkpoint_by_uuid("54937b8c-29de-4f04-862c-e67b04ec41e8"),
    ModelKey(variant=ModelVariant.AUTO_MULTIVIEW): get_checkpoint_by_uuid("6b9d7548-33bb-4517-b5e8-60caf47edba7"),
    ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW): get_checkpoint_by_uuid("0e8177cc-0db5-4cfd-a8a4-b820c772f4fc"),
    ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT): get_checkpoint_by_uuid(
        "7f6b99b7-7fac-4e74-8dbe-a394cb56ef99"
    ),
}
"""Mapping from model key to checkpoint."""

MODEL_KEYS = {k.name: k for k in MODEL_CHECKPOINTS.keys()}
"""Mapping from model name to model key."""


def get_model_literal(variant: ModelVariant = ModelVariant.BASE) -> Literal:
    """Get model literal for a given variant."""
    return Literal[tuple(k.name for k in MODEL_CHECKPOINTS.keys() if k.variant == variant)]


DEFAULT_MODEL_KEY = ModelKey()
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_MODEL_KEY]
DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


class SetupArguments(Arguments):
    """Common arguments for model setup."""

    model_config = pydantic.ConfigDict(extra="forbid")

    # Required parameters
    output_dir: Path
    """Output directory."""

    # Optional parameters
    model: get_model_literal() = DEFAULT_MODEL_KEY.name
    """Model name."""
    checkpoint_path: ResolvedFilePath | None = None
    """Path to the checkpoint."""
    experiment: str | None = None
    """Experiment name."""
    context_parallel_size: pydantic.PositiveInt | None = None
    """Context parallel size. Default to all nodes."""
    disable_guardrails: bool = False
    """Disable guardrails if this is set to True."""
    offload_guardrail_models: bool = False
    """Offload guardrail models to CPU to save GPU memory."""

    @cached_property
    def enable_guardrails(self) -> bool:
        return not self.disable_guardrails

    @cached_property
    def model_key(self) -> ModelKey:
        return MODEL_KEYS[self.model]

    @pydantic.model_validator(mode="after")
    def validate_model_key(self) -> Self:
        checkpoint = MODEL_CHECKPOINTS[self.model_key]
        if self.checkpoint_path is None:
            self.checkpoint_path = checkpoint.s3.uri
        if self.experiment is None:
            if checkpoint.experiment is None:
                raise ValueError(f"Experiment name is not set for {checkpoint.name}")
            self.experiment = checkpoint.experiment
        if self.context_parallel_size is None:
            self.context_parallel_size = int(os.environ.get("WORLD_SIZE", "1"))
        return self


class InferenceType(str, enum.Enum):
    TEXT2WORLD = "text2world"
    IMAGE2WORLD = "image2world"
    VIDEO2WORLD = "video2world"

    def __str__(self) -> str:
        return self.value


class InferenceSample(pydantic.BaseModel):
    """Sample to generate."""

    model_config = pydantic.ConfigDict(extra="forbid")

    prompt: str | None = None
    """Prompt text."""
    prompt_path: ResolvedFilePath | None = None
    """Path to the input prompt."""
    input_path: ResolvedFilePath | None = None
    """Path to the input image/video."""

    @pydantic.model_validator(mode="after")
    def validate_prompt(self) -> Self:
        if self.prompt_path is not None:
            self.prompt = self.prompt_path.read_text().strip()
        return self


class InferenceArguments(Arguments):
    """Arguments for base model inference."""

    model_config = pydantic.ConfigDict(extra="forbid")

    # Required parameters
    inference_type: InferenceType
    """Inference type."""
    samples: dict[str, InferenceSample]
    """Samples to generate."""

    # Optional parameters
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    """Negative prompt."""

    # Advanced parameters
    seed: int = 1
    """Seed value."""
    guidance: Annotated[int, pydantic.Field(ge=0, le=7)] = 7
    """Guidance value."""
    resolution: str = "none"
    """Resolution of the video (H,W). Be default it will use model trained resolution. 9:16"""
    num_output_frames: pydantic.PositiveInt = 77
    """Number of video frames to generate"""

    @pydantic.model_validator(mode="after")
    def validate_samples(self) -> Self:
        if self.inference_type != InferenceType.TEXT2WORLD:
            for sample in self.samples.values():
                if sample.input_path is None:
                    raise ValueError("Input file is required")
        return self

    @cached_property
    def num_input_frames(self) -> int:
        match self.inference_type:
            case InferenceType.TEXT2WORLD:
                return 0
            case InferenceType.IMAGE2WORLD:
                return 1
            case InferenceType.VIDEO2WORLD:
                return 2
            case _:
                assert_never(self.inference_type)
