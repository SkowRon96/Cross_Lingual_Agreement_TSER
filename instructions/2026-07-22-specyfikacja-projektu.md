# Specyfikacja projektu — Cross-Lingual Agreement (TSER)

**Cross-Lingual Agreement Between Text and Speech Emotion Recognition: Towards Robust Multimodal Perception in Social Robots**

- **Data:** 2026-07-22
- **Status:** design zatwierdzony, przed planem implementacji
- **Źródła:** [konspekt_artykulu_ESM.md](konspekt_artykulu_ESM.md), [programming_insctruction.md](programming_insctruction.md)
- **Zasada nadrzędna:** projekt badawczy — żadnego zmyślania/konfabulacji/zmieniania wyników. Wszystko odzwierciedla stan rzeczywisty.

> Dokument spisany po polsku (plan/instrukcja, §5/§8). Kod, wyniki i testy są po angielsku.

---

## 1. Cel i zakres

Zmierzyć **zgodność międzymodalną tekst↔mowa** w rozpoznawaniu emocji, wielojęzycznie, z polskim jako punktem odniesienia. Nie budujemy systemu — badamy, kiedy i jak modalności się rozjeżdżają.

**Zakres:** pełny — Badanie 1 (PL+EN, głębokie) + Badanie 2 (8 języków, szerokie). Termin elastyczny (indywidualna zgoda ESM); tempo dyktuje dokładność, nie tniemy jakości.

- **Badanie 1 — PL:** standalone **nEMO** (tekst+mowa).
- **Badanie 1 — EN:** **RAVDESS** z oryginału Zenodo (tekst+mowa+**twarz**; jedyny punkt trójmodalny).
- **Badanie 2:** **CAMEO**, 13 splitów / 8 języków (tekst+mowa).

## 2. Kluczowe decyzje (podjęte w brainstormingu)

1. **Zakres:** pełny (Badanie 1 + 2), termin elastyczny.
2. **Przestrzeń emocji:** Ekman-6 + neutral (7): `anger, disgust, fear, happiness, sadness, surprise, neutral`.
3. **Etykiety spoza Ekmana:** scalamy uznane (`calm→neutral`, `excitement/enthusiasm→happiness`), resztę odrzucamy (DROP) z raportowaniem liczb.
4. **Autorski model tekstowy (LSTM):** **inferencja z zapisanych wag** — `Models/test.h5` jest w rzeczywistości wytrenowanym 7-klasowym modelem emocji (`Dense(7)`+`sigmoid`, zweryfikowane 2026-07-23 z konfiguracji h5), a NIE modelem sentymentu. **Retraining zbędny.** word2vec potrzebny co najwyżej do rekonstrukcji/weryfikacji tokenizera (patrz §8).
5. **Środowisko:** Mac M1, 16 GB unified, CPU/MPS (bez GPU). Backend audio-LLM **wymienny** + mikro-benchmark przed pełnym runem.
6. **RAVDESS twarz:** pobieranie per-aktor → ekstrakcja klatek → kasowanie wideo (oszczędność dysku).
7. **Architektura:** Podejście A — potok etapów z typowanym, cache'owanym magazynem predykcji.
8. **Główna mowa (nagłówkowa κ) także dla EN:** `speech_zeroshot` (Qwen2-Audio) — spójność PL/EN/Badanie 2.

## 3. Środowisko wykonawcze

- Python **3.12** (`python3.12`, zweryfikowane 3.12.1). Jeden `.venv` w korzeniu repo.
- **Mac M1, 16 GB unified memory, bez GPU.** Kod device-agnostic (cpu/mps/cuda) — przenośny.
- Pydantic wszędzie (config, `Sample`, `Prediction`, obiekty wyników).
- **Wąskie gardło = RAM, nie dysk.** Qwen2-Audio 7B w fp16 (~16 GB wag) **nie zmieści się** — potrzebna kwantyzacja 4-bit (MLX, natywne Apple Silicon), mniejszy audio-LLM, albo API. `bitsandbytes` (CUDA) na Macu nie działa. Decyzja po benchmarku.
- Dysk: ~45 GB wolnego — wystarcza (RAVDESS wideo pobierane→ekstrakcja→kasowanie).

## 4. Harmonizacja etykiet (jedno źródło prawdy)

**Master-mapa `raw → kanoniczna`** (do gold i do predykcji):

| raw | → kanoniczna |
|---|---|
| anger/angry | anger |
| disgust | disgust |
| fear | fear |
| happiness/happy/joy | happiness |
| sadness/sad | sadness |
| surprise/surprised | surprise |
| neutral | neutral |
| calm | neutral *(scalenie)* |
| excitement/enthusiasm | happiness *(scalenie)* |
| none of them | neutral *(gdy model nie ma osobnego neutral)* / DROP *(gdy ma)* |
| hope, pride, contempt, frustration, gratitude, love, sarcasm, anxiety, apology, assertiveness, concern, encouragement, poker | **DROP** |

**Mechanika native→kanoniczna:** score'y natywne mapowane, sumowane gdy kilka→jedna, masa klas DROP odrzucona, renormalizacja po klasach realnie obsługiwanych przez model; `argmax`=`pred`, `max`=`confidence`; `pred_raw` zachowany.

**Realne wsparcie klas per model** (z kart modeli HF, zweryfikowane 2026-07-22):

| model_id | model | pokrywa | brakuje | extra → |
|---|---|---|---|---|
| `pl_text_lstm` | autorski LSTM (wagi `test.h5`, inferencja) | wszystkie 7 | — | — |
| `pl_text_roberta6` ⚠️ | visegradmedia Emotion_RoBERTa_polish6 | anger,fear,disgust,sadness,happiness,neutral | **surprise** | none-of-them→neutral; **kolejność etykiet ZAŁOŻONA** L0..5=anger,fear,disgust,sadness,joy,none (niezweryfikowana — brak `label2id` w repo, README się nie odtwarza); model **drugorzędny, nigdy w nagłówku** |
| `xlmr_pooled10` | poltextlab xlm-roberta pooled-emotions10-v3 | anger,fear,disgust,sadness,happiness,neutral | **surprise** | hope,pride→DROP; enthusiasm→happiness; none-of-them→DROP |
| `mling_tabularis` | tabularisai multilingual-emotion | wszystkie 7 | — | contempt,frustration,gratitude,love→DROP |
| `en_text_hartmann` | j-hartmann distilroberta | wszystkie 7 | — | — |
| `en_speech_iemocap` | speechbrain wav2vec2 IEMOCAP | anger,happiness,sadness,neutral | **fear,disgust,surprise** | — |
| `speech_cnn_tf` (RAVDESS-8) | autorski parallel CNN-Transformer | wszystkie 7 | — | calm→neutral |
| `speech_cnn_tf` (nEMO-6) | autorski parallel CNN-Transformer | anger,fear,happiness,sadness,surprise,neutral | disgust *(nEMO go nie ma)* | — |
| `speech_zeroshot` | Qwen2-Audio zero-shot | wszystkie 7 (prompt) | — | parse-fail→abstain |
| `en_face_vit` | trpakov vit-face-expression | wszystkie 7 | — | — |

**Reguła wspólnego wsparcia (zgodność):** κ liczona na przecięciu klas obsługiwanych przez oba porównywane modele, z wykluczeniem próbek, których gold leży poza tym przecięciem, oraz abstynencji. **Zawsze raportujemy N użyte / N wykluczone** — jawne, świadome ograniczenie, nie przeoczenie.
- **PL tekst↔mowa** z `roberta6`/`poltextlab`: wykluczamy gold=`surprise` (oba nie mają surprise).
- **EN mowa „druga opinia" (IEMOCAP):** wykluczamy gold ∈ `{fear, disgust, surprise}`.

**Uwaga bezpieczeństwa:** kolejność `LABEL_0..5` w visegradmedia nie jest w karcie — przypiąć z `config.json`/modelu przy ładowaniu (walidacja indeksów).

## 5. Architektura — Podejście A (potok etapów z cache'owanym magazynem predykcji)

Pięć rozdzielonych etapów: **`ingest` → `predict` → `analyze` → `report`** (+ `train-lstm`, `benchmark`). Predykcje to jedyny drogi artefakt — liczone raz, cache'owane; analiza/raport czytają je wielokrotnie.

### 5.1 Kontrakt danych i `ingest`

`Sample` (Pydantic): `sample_id ({corpus}:{original_id})`, `study`, `corpus`, `language`, `text`, `audio_ref ((parquet,row) lub .wav)`, `has_face`/`video_ref`, `gold_raw`, `gold` (kanoniczna lub None=DROP), metadane (`speaker_id, gender, age, intensity`).

- Audio **dekodowane na żądanie** (nie kopiujemy bajtów). `load_audio(sample, sr)` z parquet/wav + resampling.
- **RAVDESS** budowany z oryginału Zenodo (audio+wideo), parowanie audio↔wideo↔tekst po identyfikatorze z nazwy pliku (emocja/aktor/zdanie/powtórzenie). CAMEO `ravdess` (audio-only, hash id) tylko w Badaniu 2.
- Wyjście: `results/index/{corpus}.parquet`.

**Caveat metodologiczny (do dyskusji w artykule):** w korpusach aktorskich (RAVDESS: 2 stałe zdania; nEMO: te same zdania w różnych emocjach) **kanał tekstowy niesie mało sygnału emocjonalnego** → modele tekstowe ciążą ku „neutral". To spodziewana właściwość (spójna z H1), nie błąd — jawnie interpretowana.

### 5.2 Interfejs modeli

Protokół `EmotionModel` (`model_id, modality, native_labels, label_mapper, predict_batch()→[Prediction]`); podtypy `TextModel`/`SpeechModel`/`FaceModel`. `Prediction`: `sample_id, model_id, modality, pred_raw, pred, confidence, scores, abstained, abstain_reason`. Rejestr `registry.py` (stały, znany roster).

**Wrinkle'e (świadome ograniczenia):**
- **IEMOCAP-4** — patrz reguła wspólnego wsparcia (§4).
- **Leakage** — autorski model mowy trenowany na RAVDESS/nEMO → wyciek na tych zbiorach. Świadomy (nEMO to jedyny dostępny zbiór PL; analogicznie RAVDESS). Każda modalność ma parę „własny/leaky" + „czysty" (mowa: `speech_zeroshot` czysty; EN dodatkowo `en_speech_iemocap`). Do wniosków „bliżej gold" — model czysty. Raport z jawnym oznaczeniem leakage.
- **StandardScaler** autorskiego modelu mowy prawdopodobnie niezapisany — dopasowanie na zbiorze ewaluacyjnym (przybliżenie) lub normalizacja per-próbka; znane ograniczenie odtworzenia.
- **Confidence zero-shot** przybliżony (log-prob odpowiedzi jeśli backend daje).
- **TF na M1** (autorski LSTM, Keras) — `tensorflow-macos` obok PyTorcha; wrapper izolowany.

**Feature pipeline mowy (autorski):** log-mel `n_fft=1024, win=512, hop=256, n_mels=128, hamming`, `sr=48000`, 3 s (pad do 144000). Parallel: `(B,1,128,563)`. Klasy: RAVDESS-8 / nEMO-6 (osobne wagi `.pt`, `parallel CNN-Transformer _v1`). Realna dokładność held-out ~61% (README 94–96% = walidacja/wyciek) — raportujemy realną.

**Twarz:** ekstrakcja klatek z wideo RAVDESS → detekcja+crop twarzy → `vit-face-expression` per klatka → agregacja (średnia softmax) → 1 predykcja/wypowiedź.

**Zero-shot mowy:** interfejs `SpeechZeroShotBackend` (impl: `qwen_mlx` lokalnie 4-bit / `qwen_api`). Prompt o dokładnie 7 klas kanonicznych + parser wyjścia; `temperature=0`.

### 5.3 Magazyn predykcji i cache

- `results/predictions/study{N}/{corpus}/{model_id}.parquet` + `manifest.json`.
- **Cache** po sygnaturze `model_id + config_hash + hash(sample_id set)`; `--force` wymusza. Zmiana promptu/wersji → recompute.
- **Zapis przyrostowy + wznawianie** (checkpoint co N; pomijanie policzonych `sample_id`) — kluczowe dla Qwen na 41 tys.
- **Determinizm:** ziarna; audio-LLM greedy/`temperature=0`.
- **Abstynencje jawne** (`pred=None` + reason), nie ciche dropowanie.

### 5.4 Analiza i statystyki

Czyta predykcje + gold, zwraca obiekty Pydantic → `results/analysis/study{N}/*.json` + tidy-CSV. Ustalone ziarno (powtarzalny bootstrap).

**Badanie 1:** (1) baseline per modalność (accuracy, macro-F1, macierz pomyłek); (2) zgodność międzymodalna — Cohen's κ pary (PL: tekst–mowa; EN: tekst–mowa, tekst–twarz, mowa–twarz), Fleiss' κ trójka RAVDESS; (3) rzetelność w obrębie modalności (κ między dwoma modelami tej samej modalności); (4) konflikty — która modalność bliżej gold, rozkład per emocja, korelacja confidence.

**Badanie 2:** Cohen's κ tekst–mowa per język (8); analiza **opisowa** (n=8), pozycja/outlier PL; opisowa korelacja κ z dystansem typologicznym / zasobnością.

**Precyzyjne użycie testów:**
- Cohen's κ (pary), Fleiss' κ (trójka) — na wspólnym wsparciu, z wykluczeniami (§4).
- **Bootstrap 1000×** — 95% CI percentylowe każdej κ; do porównania dwóch κ (PL vs EN) — **CI różnicy** z tych samych resampli.
- **McNemar** — tylko wewnątrz jednego zbioru, sparowana poprawność dwóch modalności na tych samych próbkach.
- **PL vs EN** (niesparowane) → **chi-kwadrat / test dwóch proporcji**, nie McNemar.
- **Confidence:** Pearson + Spearman między modalnościami (z zastrzeżeniem zero-shot).

**Wiele modeli na modalność:** pełna **macierz** κ (tekst×mowa) + **główna para czystych** na nagłówek:

| | tekst (główny) | mowa (główny) |
|---|---|---|
| PL | `mling_tabularis` | `speech_zeroshot` |
| EN | `en_text_hartmann` | `speech_zeroshot` |

Modele leaky/zawężone → pełna macierz + kontrola rzetelności, nie nagłówek.

### 5.5 Raportowanie

Deterministyczna regeneracja z `results/analysis/*.json`. **Jednolity styl:** wspólny moduł (czcionki/DPI/siatka) + **stała colorblind-safe paleta-per-emocja** i stała kolejność klas wszędzie.

- Rysunki: **PDF (wektor) + PNG 300 DPI** → `results/figures/study{N}/`.
- Tabele: **LaTeX + CSV + Markdown + PNG** (render obrazkowy tej samej tabeli, spójny styl) → `results/tables/study{N}/`.
- Artefakty: macierze pomyłek per model; tabele κ z bootstrap CI + pełna macierz tekst×mowa; Fleiss' κ (RAVDESS); tabela PL vs EN + istotność; konflikty (rozkład per emocja, „która bliżej gold", scatter confidence); Badanie 2 — κ dla 8 języków z wyróżnionym PL + scatter typologia/zasobność.

## 6. Struktura `app/`, CLI, `launch.json`

Kod z `old_projects` **przepisywany do `app/`** (nie importowany); `old_projects` nietknięty.

```
app/
  config/  settings.py · studies.py · emotions.py
  labels/  canonical.py · mapping.py · mapper.py
  data/    schema.py · ingest.py · loaders/{nemo,cameo,ravdess}.py · audio.py · video.py
  models/  base.py · registry.py
           text/{hf_text.py, lstm/{architecture,wrapper}.py}   # train.py opcjonalny — wagi test.h5 już są
           speech/{architectures.py, cnn_transformer.py, speechbrain_iemocap.py,
                   zeroshot/{base,qwen_mlx,qwen_api,prompt}.py}
           face/vit_face.py
  predict/ runner.py · store.py
  analysis/ metrics · agreement · bootstrap · tests · conflicts · correlation · results
  report/  style.py · figures.py · tables.py
  cli.py
tests/     # mapowanie etykiet, matematyka κ, wykluczenia, schema ingestu
```

**CLI:**
```
python -m app.cli ingest    --corpus nemo|cameo|ravdess
python -m app.cli train-lstm                                  # OPCJONALNY — wagi test.h5 już istnieją
python -m app.cli predict   --study 1 --corpus nemo --model all|<id>
python -m app.cli analyze   --study 1
python -m app.cli report    --study 1
python -m app.cli benchmark --model speech_zeroshot --n 20
```

**`launch.json`:** osobna konfiguracja debug per komenda, interpreter `.venv` (`python3.12`).

**Poprzeczne:** Pydantic wszędzie; device-agnostic; **TDD** na elementach krytycznych dla poprawności (mapowanie score'ów, κ na znanych przypadkach, logika wykluczeń, schema).

## 7. Ograniczenia (jawnie w artykule)

- Brak polskiego zbioru trójmodalnego — twarz tylko EN (RAVDESS).
- n=8 języków — Badanie 2 opisowe/eksploracyjne, nie dowód inferencyjny.
- Zero-shot audio-LLM — niższy pułap niż dedykowane klasyfikatory (spójne z PolEval Task 4).
- Mowa aktorska/wyreżyserowana (RAVDESS, nEMO) — ograniczona trafność ekologiczna; kanał tekstowy słabo emocjonalny w tych korpusach.
- Leakage autorskich modeli na RAVDESS/nEMO — świadomy, oznaczony; wnioski „bliżej gold" z modeli czystych.
- **Model `pl_text_roberta6` (visegradmedia) — kolejność etykiet ZAŁOŻONA, niezweryfikowana** (L0..5=anger,fear,disgust,sadness,joy,none-of-them). Repo nie publikuje `label2id`; próby odzysku (README-fingerprint, sonda nEMO) nie dały pewnego mapowania. Używany jako model **drugorzędny** (nigdy w nagłówku κ, tylko w pełnej macierzy / kontroli „model-model"); jawnie oflagowany. Jeśli mapowanie okaże się błędne, zaniży to zgodność roberta6 — dlatego nie wpływa na wnioski nagłówkowe.

## 8. Otwarte punkty (do rozstrzygnięcia w implementacji)

1. **Backend Qwen2-Audio** (MLX-4bit lokalnie vs API) — decyzja po mikro-benchmarku (20 próbek: czas/próbka + RAM, ekstrapolacja na 41 tys.).
2. **StandardScaler** autorskiego modelu mowy — strategia normalizacji (dopasowanie na eval vs per-próbka).
3. **Zakres pobrania RAVDESS z Zenodo** (Video_Speech per-aktor) + pipeline klatek/detekcji twarzy.
4. **none-of-them / poker** — potwierdzone: none-of-them→neutral (bez osobnego neutral) / DROP (z neutral); poker→DROP.
5. **Tokenizer autorskiego LSTM** — `finalwordindex.pkl` (76 KB) wygląda na zbyt mały wobec słownika modelu (`input_dim=294182`). Zweryfikować dopasowanie do `test.h5`; w razie niezgodności zrekonstruować `word_index` ze słownika word2vec (deterministycznie).
6. **Retraining / poprawa autorskiego LSTM — opcja awaryjna, TYLKO na sam koniec.** Domyślnie używamy zapisanych wag (`test.h5`, §2 pkt 4). Jeśli końcowe wyniki modelu tekstowego PL okażą się słabe (np. baseline accuracy/F1 znacząco odstający od pozostałych modeli tekstowych), dopiero wtedy rozważamy retrening/dostrojenie LSTM: word2vec (mamy) + dane chatbota + odbudowa tokenizera. Nie robimy tego na starcie — to ostatni krok, uruchamiany warunkowo po ocenie wyników.
