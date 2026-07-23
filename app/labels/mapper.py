"""LabelMapper: map a model's native score vector into the canonical space.

Aggregates probabilities of native labels that collapse to the same canonical
emotion, discards the mass of dropped labels, and renormalizes over the
emotions the model actually supports.
"""
from collections.abc import Sequence

from app.config.emotions import Emotion
from app.labels.mapping import map_label


class LabelMapper:
    def __init__(
        self,
        native_labels: list[str],
        overrides: dict[str, Emotion | None] | None = None,
    ) -> None:
        self.native_labels = list(native_labels)
        self.overrides = overrides
        # Precompute native index -> Emotion | None (None means dropped).
        self._targets: list[Emotion | None] = [
            map_label(lbl, overrides=overrides) for lbl in self.native_labels
        ]

    @property
    def supported(self) -> set[Emotion]:
        return {t for t in self._targets if t is not None}

    def map_label(self, native: str) -> Emotion | None:
        return map_label(native, overrides=self.overrides)

    def map_scores(self, scores: Sequence[float]) -> dict[Emotion, float]:
        if len(scores) != len(self.native_labels):
            raise ValueError(
                f"Expected {len(self.native_labels)} scores, got {len(scores)}"
            )
        aggregated: dict[Emotion, float] = {}
        kept_mass = 0.0
        for target, score in zip(self._targets, scores):
            if target is None:
                continue
            aggregated[target] = aggregated.get(target, 0.0) + float(score)
            kept_mass += float(score)
        if kept_mass <= 0.0:
            return {}
        return {emotion: value / kept_mass for emotion, value in aggregated.items()}
