from pathlib import Path
from app.config.settings import Settings


def test_defaults_point_into_repo(tmp_path):
    s = Settings(datasets_root=tmp_path / "datasets", results_root=tmp_path / "results")
    assert s.index_dir == tmp_path / "results" / "index"
    assert s.seed == 42


def test_nemo_and_cameo_globs(tmp_path):
    (tmp_path / "datasets" / "nEMO" / "data").mkdir(parents=True)
    (tmp_path / "datasets" / "CAMEO" / "data").mkdir(parents=True)
    (tmp_path / "datasets" / "nEMO" / "data" / "train-00000.parquet").touch()
    (tmp_path / "datasets" / "nEMO" / "data" / "train-00001.parquet").touch()
    (tmp_path / "datasets" / "CAMEO" / "data" / "nemo-00000.parquet").touch()
    s = Settings(datasets_root=tmp_path / "datasets", results_root=tmp_path / "results")
    assert [p.name for p in s.nemo_parquets()] == ["train-00000.parquet", "train-00001.parquet"]
    assert [p.name for p in s.cameo_parquets()] == ["nemo-00000.parquet"]
