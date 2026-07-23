"""Project settings: dataset/results paths and run config."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TSER_")

    datasets_root: Path = Field(default=_REPO_ROOT / "datasets")
    results_root: Path = Field(default=_REPO_ROOT / "results")
    seed: int = 42

    @property
    def index_dir(self) -> Path:
        return self.results_root / "index"

    @property
    def ravdess_audio_dir(self) -> Path:
        return self.datasets_root / "RAVDESS" / "Audio_Speech_Actors_01-24"

    @property
    def ravdess_video_dir(self) -> Path:
        return self.datasets_root / "RAVDESS"

    @property
    def hf_home(self) -> Path:
        return _REPO_ROOT / "models" / "hf_cache"

    def nemo_parquets(self) -> list[Path]:
        return sorted((self.datasets_root / "nEMO" / "data").glob("*.parquet"))

    def cameo_parquets(self) -> list[Path]:
        return sorted((self.datasets_root / "CAMEO" / "data").glob("*.parquet"))
