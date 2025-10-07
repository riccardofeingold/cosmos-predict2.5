import pytest

from cosmos_predict2.config import ModelKey, ModelSize, ModelVariant


@pytest.mark.L0
def test_model_key():
    assert ModelKey().name == "2B/post-trained"
    assert ModelKey(size=ModelSize._14B).name == "14B/post-trained"
    assert ModelKey(variant=ModelVariant.AUTO_MULTIVIEW).name == "2B/auto/multiview"
    assert ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW).name == "2B/robot/multiview"
    assert ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT).name == "2B/robot/multiview-agibot"
