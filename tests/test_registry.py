import pytest
from app.models.registry import build_model, MODEL_IDS


def test_registry_lists_and_unknown_raises():
    assert "en_text_hartmann" in MODEL_IDS
    with pytest.raises(KeyError):
        build_model("nonexistent_model")


@pytest.mark.slow
def test_build_hartmann():
    m = build_model("en_text_hartmann")
    assert m.model_id == "en_text_hartmann" and m.modality == "text"
