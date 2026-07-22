# Konspekt artykułu — ESM

## Tytuł

**Cross-Lingual Agreement Between Text and Speech Emotion Recognition: Towards Robust Multimodal Perception in Social Robots**

*(Zgodność międzymodalna tekstu i mowy w rozpoznawaniu emocji: w stronę odpornej percepcji multimodalnej w robotyce społecznej)*

**Termin:** koniec lipca 2026, zgoda organizatora ESM na indywidualny termin.

---

## Motyw i kontekst

W dotychczasowym dorobku badano poszczególne modalności (tekst, mowa, twarz, sygnały biomedyczne) głównie **osobno, na różnych zbiorach danych**. Brakuje badań na zbiorach, gdzie kilka sygnałów (tekst/mowa/twarz) jest dostępnych jednocześnie, co uniemożliwia badanie zależności między modalnościami — kluczowe pytanie dla koncepcji systemu multimodalnego zaplanowanego na rok V (moduł decyzyjny robota społecznego).

Dodatkowe argumenty za tym, że temat wart jest zbadania:

- **Realny problem dla robotyki społecznej.** Robot społeczny odbiera sygnały emocjonalne z wielu kanałów naraz (co użytkownik mówi, jak brzmi, jaką ma minę). Kiedy kanały są zgodne, decyzja jest prosta; kiedy się rozjeżdżają (np. sarkazm, tłumiony gniew), system musi wiedzieć, któremu kanałowi ufać bardziej — a to wymaga wcześniejszej wiedzy o tym, jak często i w jaki sposób modalności faktycznie się nie zgadzają.
- **Wielojęzyczność jako praktyczne ograniczenie wdrożeniowe.** Systemy HRI projektowane w jednym języku (najczęściej angielskim) są zwykle wdrażane bez sprawdzenia, czy ich założenia przenoszą się na inne języki. Jeśli wzorce zgodności międzymodalnej różnią się między językami, oznacza to, że moduł decyzyjny wytrenowany/dostrojony na danych angielskich może dawać mylące wyniki w innym języku — to bezpośrednie ryzyko dla jakości interakcji człowiek-robot w praktyce.
- **Aktualność problemu zasobów dla języka polskiego.** Polski jest wciąż językiem niskozasobowym w kontekście mowy emocjonalnej (potwierdzone m.in. przez PolEval 2025 Task 4 i skromną liczbę dedykowanych korpusów), a jednocześnie coraz częściej pojawia się w dużych, wielojęzycznych zbiorach (CAMEO) i benchmarkach. To dobry moment, by ilościowo sprawdzić, jak ten niedobór przekłada się na jakość i spójność rozpoznawania emocji, zamiast tylko go sygnalizować.
- **Metodologiczna luka w istniejącej literaturze.** Prace o niespójności międzymodalnej (np. InconVAD) analizują to zjawisko w obrębie jednego języka; prace cross-lingwalne (np. Multimodal Dual Attention Transformer) mierzą accuracy klasyfikacji, nie zgodność między kanałami. Nikt nie połączył obu wątków w jedno systematyczne porównanie.
- **Rosnąca dostępność narzędzi zero-shot.** Pojawienie się dużych modeli audio-językowych (np. Qwen2-Audio) po raz pierwszy pozwala tanio i porównywalnie zbadać wiele języków jednocześnie, bez konieczności trenowania osobnego klasyfikatora na każdy z nich — to otwiera możliwość badania, które wcześniej było praktycznie nieopłacalne kosztowo.

Artykuł ma dostarczyć **pierwszego elementu** tej brakującej analizy: nie budowy systemu, lecz badania, czy i jak modalności się ze sobą zgadzają — w kontekście wielojęzycznym, z językiem polskim jako punktem odniesienia.

## Cel pracy

Zbadać **zgodność międzymodalną między tekstem a mową** w rozpoznawaniu emocji:

1. dogłębnie — na dwóch/trzech w pełni kontrolowanych zbiorach (PL, EN, opcjonalnie z modalnością twarzy),
2. szeroko — na 8 językach jednocześnie, by sprawdzić, czy wzorce zgodności są uniwersalne, czy zależne od języka/rodziny językowej,

oraz określić, jaką rolę w tym kontekście odgrywają ograniczone zasoby dla języka polskiego.

## Luka badawcza

- Brak zbiorów trójmodalnych (tekst+mowa+twarz) dla języka polskiego — udokumentowana, nie rozwiązywana w tym artykule.
- Istniejące prace o niespójności międzymodalnej (np. InconVAD) badają to **wyłącznie dla jednego języka**.
- Istniejące prace o cross-lingual SER (np. Multimodal Dual Attention Transformer, Cross-Lingual Cross-Corpus SER) mierzą **accuracy klasyfikacji**, nie zgodność między modalnościami.
- Nikt nie połączył obu wątków: systematycznego pomiaru zgodności tekst–mowa na wielu językach jednocześnie, z polskim jako punktem odniesienia.

Pełny zestaw hipotez (H1–H4) znajduje się przy odpowiednim badaniu niżej — H1, H2 dotyczą Badania 1, H3, H4 dotyczą Badania 2.

---

## Podział na dwa niezależne badania

Całość dzieli się na dwa badania, które **nie zależą od siebie nawzajem** — różne dane, w większości różne modele, różne pytania badawcze. Można je realizować równolegle lub w dowolnej kolejności; w razie presji czasowej artykuł ma sens nawet z samym Badaniem 1 (Badanie 2 staje się wtedy materiałem na kolejny, osobny artykuł).

---

## Badanie 1 — warstwa głęboka (PL, EN)

**Pytanie badawcze:** Jak dokładnie modalności rozjeżdżają się w konkretnych przypadkach — gdzie, jak często i która modalność jest bliżej prawdy, gdy dochodzi do konfliktu?

### Zbiory danych

| Zbiór                                                          | Modalności                 | Rola                                                  |
| --------------------------------------------------------------- | --------------------------- | ----------------------------------------------------- |
| **nEMO** (PL, przez CAMEO)                                | tekst (transkrypcja) + mowa | analiza PL                                            |
| **RAVDESS** (EN, przez CAMEO + oryginalne wideo z Zenodo) | tekst + mowa + twarz        | analiza EN, jedyny punkt z pełną trójmodalnością |

### Modele

**PL**

- Tekst: własny LSTM (dane z chatbota) + [huggingface.co/visegradmedia-emotion/Emotion_RoBERTa_polish6](https://huggingface.co/visegradmedia-emotion/Emotion_RoBERTa_polish6) + [huggingface.co/poltextlab/xlm-roberta-large-pooled-emotions10-v3](https://huggingface.co/poltextlab/xlm-roberta-large-pooled-emotions10-v3) + [huggingface.co/tabularisai/multilingual-emotion-classification](https://huggingface.co/tabularisai/multilingual-emotion-classification)
- Mowa: własny CNN/Transformer (z wcześniejszej pracy) + model zero-shot (audio-LLM, ten sam co w Badaniu 2 — brak gotowego dedykowanego klasyfikatora SER dla PL na Hugging Face)

**EN**

- Tekst: [huggingface.co/j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) + [huggingface.co/poltextlab/xlm-roberta-large-pooled-emotions10-v3](https://huggingface.co/poltextlab/xlm-roberta-large-pooled-emotions10-v3) + [huggingface.co/tabularisai/multilingual-emotion-classification](https://huggingface.co/tabularisai/multilingual-emotion-classification)
- Mowa: własny CNN/Transformer (RAVDESS) + [huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) (celowo nie model trenowany na RAVDESS, by uniknąć wycieku danych)
- Twarz:  [huggingface.co/trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression) (na klatkach z oryginalnego wideo RAVDESS)

### Eksperymenty

1. **Baseline per modalność** — accuracy/F1-macro, macierze pomyłek względem prawdziwej etykiety.
2. **Zgodność międzymodalna** — Cohen's kappa dla każdej pary modalności (PL: tekst–mowa; EN: tekst–mowa, tekst–twarz, mowa–twarz); Fleiss' kappa dla trójki modalności w RAVDESS.
3. **Wiarygodność modeli tej samej modalności** — zgodność między dwoma niezależnymi modelami tekstowymi/mowy jako kontrola solidności wyniku.
4. **Analiza konfliktów** — przypadki, gdzie modalności się nie zgadzają: która modalność jest bliżej prawdziwej etykiety; rozkład konfliktów per klasa emocji; korelacja pewności predykcji (confidence score) między modalnościami.

### Statystyki i metryki

- Accuracy, F1-macro, macierze pomyłek
- Cohen's kappa (pary modalności), Fleiss' kappa (trójka modalności, RAVDESS)
- Test chi-kwadrat / McNemar — istotność różnic PL vs EN
- Bootstrap (np. 1000 resampli) — przedziały ufności dla każdej kappy
- Korelacja Pearsona/Spearmana — zgodność confidence score między modalnościami

### Oczekiwane wyniki

- H1: Zgodność tekst–mowa umiarkowana (kappa ~0,4–0,6), nie idealna — modalności wnoszą częściowo niezależną informację.
- H2: W RAVDESS silniejsza zgodność mowa–twarz niż tekst–mowa (oba kanały niewerbalne).
- Konkretna, policzalna charakterystyka przypadków konfliktu — materiał bezpośrednio zasilający projekt modułu decyzyjnego robota społecznego.

---

## Badanie 2 — warstwa szeroka (8 języków)

**Pytanie badawcze:** Gdzie plasuje się polski na tle innych języków pod względem zgodności tekst–mowa, i czy wzorzec zgodności zależy od rodziny językowej lub zasobności danego języka.

### Zbiory danych

| Zbiór                                                                           | Modalności  | Rola                               |
| -------------------------------------------------------------------------------- | ------------ | ---------------------------------- |
| **CAMEO — pełna kolekcja** (8 języków: PL, EN, FR, IT, ES, DE, RU, BN) | tekst + mowa | pozycja PL na tle innych języków |

Wszystkie splity CAMEO mają jednolity schemat (audio, transkrypcja, etykieta emocji, mówca, płeć, wiek, język) — ta sama pętla przetwarzania dla każdego języka.

### Modele

- Jeden model audio-LLM (np. Qwen2-Audio) w trybie zero-shot, stosowany do mowy we wszystkich językach a drugi model [huggingface.co/poltextlab/xlm-roberta-large-pooled-emotions10-v3](https://huggingface.co/poltextlab/xlm-roberta-large-pooled-emotions10-v3) stosowany do tekstu we wszystkich językach— bez treningu, bez ryzyka niespójności metodologicznej między językami.

### Eksperymenty

1. **Sweep 8-językowy** — ta sama miara zgodności tekst–mowa policzona dla każdego z 8 języków, zestawiona na jednym wykresie/tabeli; test, czy PL jest outlierem.
2. **Test efektu rodziny językowej** — wstępne zestawienie zgodności z odległością typologiczną między językami (słowiańska, romańska, germańska).
3. **Diagnoza zasoby vs struktura języka** — zestawienie wyniku PL z przybliżonym rozmiarem danych treningowych danego języka w modelu (jeśli dostępne w karcie modelu), by rozróżnić niedobór zasobów od cech strukturalnych języka.

### Statystyki i metryki

- Cohen's kappa tekst–mowa per język
- Analiza opisowa (nie inferencyjna, z uwagi na n=8) — pozycja PL na tle rozkładu
- Korelacja (opisowa) między zgodnością a przybliżonym dystansem typologicznym / zasobnością języka

### Oczekiwane wyniki

- H3: Polski uzyska niższą zgodność/skuteczność niż języki wyżej zasobne — jeśli potwierdzone, daje policzalny dowód na tezę o niedoborze zasobów, centralną dla całej rozprawy.
- H4 (eksploracyjna, n=8, bez pełnej mocy statystycznej): języki tej samej rodziny (słowiańska, romańska, germańska) mogą wykazywać bardziej zbliżone wzorce zgodności niż języki z różnych rodzin.

---

## Ograniczenia (do jawnego zaznaczenia w artykule)

- Brak polskiego zbioru trójmodalnego — analiza z twarzą możliwa tylko dla EN (RAVDESS), Badanie 1.
- n=8 języków to za mało na pełnoprawną regresję — Badanie 2 jako analiza eksploracyjna/opisowa, nie dowód statystyczny.
- Model zero-shot (audio-LLM) może mieć niższy pułap skuteczności niż dedykowane klasyfikatory — ograniczenie znane i akceptowalne (spójne z niskim baseline'em w PolEval Task 4).
- Wyniki bazują na mowie aktorskiej/wyreżyserowanej (RAVDESS, nEMO), nie spontanicznej — ograniczenie ekologicznej trafności.

## Struktura artykułu (szkic sekcji)

1. Wstęp — motywacja z perspektywy robotyki społecznej, luka badawcza
2. Related Work — cross-modal inconsistency (InconVAD), cross-lingual SER (MDAT i inne), zasoby dla polskiego (PolEval Task 4, CAMEO)
3. Metodologia — osobno dla Badania 1 i Badania 2 (zbiory, modele, pipeline, metryki)
4. Wyniki — Sekcja 4.1: Badanie 1 (PL/EN, w tym trójmodalność); Sekcja 4.2: Badanie 2 (8 języków)
5. Dyskusja — implikacje dla modułu decyzyjnego robota społecznego, diagnoza zasoby vs struktura
6. Wnioski i kierunki dalsze — jawne wskazanie luki (brak polskiego zbioru trójmodalnego) jako mostu do kolejnego etapu doktoratu
