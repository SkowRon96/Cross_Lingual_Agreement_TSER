import glob
import pytest
from pathlib import Path
from app.config.emotions import Emotion
from app.models.face.vit_face import aggregate_face_scores


def test_aggregate_face_scores_is_mean_over_crops():
    per_crop = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
    out = aggregate_face_scores(per_crop)
    assert out == pytest.approx([0.5, 0.5, 0.0])


@pytest.mark.slow
@pytest.mark.skipif(not glob.glob("datasets/RAVDESS/Actor_01/01-*.mp4"),
                    reason="RAVDESS dataset not present")
def test_vit_face_predicts_on_real_ravdess_video():
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("en_face_vit")
    assert m.modality == "face"
    video = sorted(glob.glob("datasets/RAVDESS/Actor_01/01-*.mp4"))[0]
    s = Sample(sample_id="RAVDESS:probe", studies=[1], corpus="RAVDESS", language="en",
               text="Kids are talking by the door",
               audio=AudioRef(wav_path=Path("a.wav")), has_face=True, video_path=Path(video),
               gold_raw="neutral", gold=Emotion.NEUTRAL)
    [pred] = m.predict_batch([s])
    assert pred.modality == "face"
    assert pred.pred in set(Emotion)                      # a face was found -> a real prediction
    assert pred.abstained is False
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-5


@pytest.mark.slow
def test_vit_face_abstains_without_video(tmp_path):
    from app.models.registry import build_model
    from app.data.schema import Sample, AudioRef
    m = build_model("en_face_vit")
    s = Sample(sample_id="t:noface", studies=[1], corpus="t", language="en", text="x",
               audio=AudioRef(wav_path=tmp_path / "a.wav"), has_face=False, video_path=None,
               gold_raw="neutral", gold=Emotion.NEUTRAL)
    [pred] = m.predict_batch([s])
    assert pred.abstained is True and pred.pred is None and pred.scores == {}
