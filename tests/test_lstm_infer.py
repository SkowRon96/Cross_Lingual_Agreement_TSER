import json
import subprocess
import pytest
from pathlib import Path
from app.config.settings import Settings

_S = Settings()
_HAS = (_S.lstm_model_dir / "model.h5").exists() and _S.tf_python.exists()


@pytest.mark.slow
@pytest.mark.skipif(not _HAS, reason="trained LSTM model or .venv-tf not present")
def test_infer_script_runs_via_tf_venv():
    payload = {"texts": ["jestem bardzo szczęśliwy i zadowolony z dzisiejszego dnia",
                          "jestem smutny i przygnębiony, wszystko stracone"]}
    proc = subprocess.run(
        [str(_S.tf_python), str(_S.lstm_infer_script), str(_S.lstm_model_dir)],
        input=json.dumps(payload), text=True, capture_output=True,
        env={"TF_CPP_MIN_LOG_LEVEL": "3", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout)                       # stdout must be pure JSON
    assert out["labels"] == ["neutral", "happiness", "sadness", "fear", "anger", "disgust", "surprise"]
    assert len(out["probs"]) == 2 and len(out["probs"][0]) == 7
    # clearly-happy sentence -> happiness is the argmax (model has ~97% happiness recall)
    assert out["labels"][max(range(7), key=lambda i: out["probs"][0][i])] == "happiness"
