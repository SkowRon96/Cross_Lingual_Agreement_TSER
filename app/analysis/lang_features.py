"""Typological distance from Polish (URIEL/lang2vec) + resourcedness tier (Joshi 2020).

Values are cached here for determinism and offline runs. If `lang2vec` is installed the
verify script can recompute SYNTACTIC_DISTANCE and diff against this table. RESOURCE_TIER
uses Joshi et al. (2020) classes (0=lowest .. 5=highest).

USER TO CONFIRM the exact values before the Study-2 report is finalized. lang2vec is not
installed in the current .venv, so these cached values are in force and flagged as assumed.
"""
# Syntactic distance from Polish (0 = identical). Cached values approximating lang2vec
# 'syntactic' cosine distance to 'pol'; recompute with lang2vec to confirm.
SYNTACTIC_DISTANCE: dict[str, float] = {
    "pl": 0.0, "ru": 0.34, "de": 0.49, "en": 0.52,
    "fr": 0.55, "it": 0.53, "es": 0.54, "bn": 0.66,
}
# Joshi et al. (2020) resource classes (5 = high like en/de/fr; bn lower).
RESOURCE_TIER: dict[str, int] = {
    "en": 5, "de": 5, "fr": 5, "es": 5, "it": 4, "ru": 4, "pl": 4, "bn": 3,
}

# ISO-639-1 -> 639-3 for optional lang2vec recomputation.
ISO1_TO_ISO3: dict[str, str] = {
    "pl": "pol", "en": "eng", "fr": "fra", "it": "ita",
    "es": "spa", "de": "deu", "ru": "rus", "bn": "ben",
}

ASSUMED = True  # True while values are the cached fallback (lang2vec not used)
