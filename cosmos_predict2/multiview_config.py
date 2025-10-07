from typing import Literal, Annotated
from pathlib import Path
import pydantic
from typing_extensions import Self
from cosmos_predict2.config import (
    MODEL_CHECKPOINTS,
    ModelKey,
    Arguments,
    ResolvedFilePath,
    ModelVariant,
    SetupArguments,
    get_model_literal,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.AUTO_MULTIVIEW)
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_MODEL_KEY]

VIEW_INDEX_DICT = {
    "front_wide": 0,
    "cross_right": 1,
    "rear_right": 2,
    "rear": 3,
    "rear_left": 4,
    "cross_left": 5,
    "front_tele": 6,
}

StackMode = Literal["time", "height"]


class MultiviewSetupArguments(SetupArguments):
    """Arguments for multiview setup."""

    model_config = pydantic.ConfigDict(extra="forbid")

    model: get_model_literal(ModelVariant.AUTO_MULTIVIEW) = DEFAULT_MODEL_KEY.name


class ViewConfig(pydantic.BaseModel):
    """Configuration for a single view."""

    model_config = pydantic.ConfigDict(extra="forbid")

    video_path: ResolvedFilePath | None = None
    """Path to the input video for this view."""


class MultiviewInferenceArguments(Arguments):
    """Arguments for multiview inference."""

    model_config = pydantic.ConfigDict(extra="forbid")

    prompt: str | None = None
    """Text prompt for generation."""
    prompt_path: ResolvedFilePath | None = None
    """Path to file containing the prompt."""

    guidance: Annotated[int, pydantic.Field(ge=0, le=7)] = 3
    """Guidance value for generation."""
    seed: int = 0
    """Random seed for generation."""
    n_views: int = pydantic.Field(default=7, description="Number of views to generate")
    """Number of views to generate."""
    num_input_frames: Annotated[int, pydantic.Field(ge=0, le=2)] = 2
    """Number of input frames (0-2)."""
    control_weight: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 1.0
    """Control weight for generation."""
    stack_mode: StackMode = "time"
    """Stacking mode for frames."""

    front_wide: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Front wide view configuration."""
    rear: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear view configuration."""
    rear_left: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear left view configuration."""
    rear_right: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear right view configuration."""
    cross_left: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Cross left view configuration."""
    cross_right: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Cross right view configuration."""
    front_tele: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Front tele view configuration."""

    fps: pydantic.PositiveInt = 30
    """Frames per second for output video."""
    num_steps: pydantic.PositiveInt = 35
    """Number of generation steps."""

    @pydantic.model_validator(mode="after")
    def validate_prompt(self) -> Self:
        """Validate and load prompt from file if prompt_path is provided."""
        if self.prompt_path is not None:
            self.prompt = self.prompt_path.read_text().strip()
        return self

    @property
    def input_paths(self) -> dict[str, Path | None]:
        """Get input paths for all views."""
        input_paths = {
            "front_wide": self.front_wide.video_path,
            "rear": self.rear.video_path,
            "rear_left": self.rear_left.video_path,
            "rear_right": self.rear_right.video_path,
            "cross_left": self.cross_left.video_path,
            "cross_right": self.cross_right.video_path,
            "front_tele": self.front_tele.video_path,
        }
        return input_paths
