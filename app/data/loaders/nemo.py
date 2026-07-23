"""nEMO (Polish) loader: standalone HF parquet shards -> Sample list (Study 1)."""
from pathlib import Path

import pyarrow.parquet as pq

from app.data.schema import AudioRef, Sample


def load_nemo(parquets: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for shard in parquets:
        cols = pq.read_table(
            shard,
            columns=["file_id", "emotion", "normalized_text", "speaker_id", "gender", "age"],
        ).to_pydict()
        n = len(cols["file_id"])
        for i in range(n):
            samples.append(Sample(
                sample_id=f"nEMO:{cols['file_id'][i]}",
                studies=[1],
                corpus="nEMO",
                language="pl",
                text=cols["normalized_text"][i],
                audio=AudioRef(parquet_path=shard, row_index=i),
                gold_raw=cols["emotion"][i],
                gold=None,
                speaker_id=cols["speaker_id"][i],
                gender=cols["gender"][i],
                age=str(cols["age"][i]),
            ))
    return samples
