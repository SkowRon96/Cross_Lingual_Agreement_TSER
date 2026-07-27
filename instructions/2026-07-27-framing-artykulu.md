# Framing artykułu — szkic roboczy (do decyzji z promotorem)

> Dokument roboczy. Wszystkie liczby z realnych wyników (`results/`). Standalone baseline'y mowy
> nEMO/RAVDESS są odświeżane po poprawce resamplingu Qwena — **zmienią się o ~0.02–0.07 w górę,
> ale NIE zmieniają κ ani wniosków** (κ limitowane kanałem tekstu).

## 1. Punkt wyjścia vs to, co faktycznie wyszło

**Zakładana teza (tytuł doktoratu):** cross-lingual agreement między rozpoznawaniem emocji z tekstu
i mowy — ku odpornej percepcji multimodalnej w robotach społecznych.

**Co faktycznie pokazują dane:** zgodność text↔speech w standardowych korpusach SER jest
**zdominowana przez to, czy kanał tekstowy niesie emocję**, a nie przez „zgodność modalności" per se.
Dlatego proponuję **przeramować** artykuł: z „mierzymy cross-lingual agreement" na
**„dlaczego zgodność text↔speech jest trudna do zmierzenia w korpusach SER — i co każda modalność
osobno ujawnia (także wielojęzycznie)"**. To uczciwe i mocniejsze niż udawanie wysokiej zgodności.

## 2. Rekomendowane wkłady (contributions), w kolejności siły

**C1 (główny, metodologiczny — wynik pozytywny z negatywnego).**
Cross-modalna zgodność text↔speech jest ograniczona emocjonalnością tekstu korpusu. Dowód
kontrolowanym kontrastem (ten sam język EN, te same modalności, różni się tylko naturalność tekstu):
- RAVDESS (aktorski, 2 stałe zdania): text↔speech **κ = 0.000** [0.000, 0.000]
- MELD (realny dialog): text↔speech **κ = 0.202** [0.194, 0.210]
- mechanizm: model tekstowy na RAVDESS przewiduje *neutral* dla 100% klipów (macro-F1 0.048),
  na MELD używa wszystkich 7 emocji (acc 0.54, macro-F1 0.42);
- **odwrócenie konfliktów**: gdy modalności się różnią, na RAVDESS bliżej gold jest **mowa**
  (956 vs 119), na MELD **tekst** (4065 vs 1889).
- **potwierdzenie „odstającego" tekstu (trimodalnie):** na RAVDESS jedyna realna zgodność jest
  **mowa↔twarz κ=0.297**, a text↔speech i text↔face = 0.00 → sygnał emocjonalny jest w mowie i
  twarzy, tekst nie zgadza się z niczym. Na MELD odwrotnie (tekst↔mowa 0.20; twarz largest-face
  ≈ szum 0.04). „Który kanał niesie sygnał" zależy od korpusu.
- **Wniosek dla pola TSER:** aktorskie korpusy SER (stałe/neutralne zdania) są nieprzydatne do
  badań zgodności text↔speech — ich warstwa językowa jest emocjonalnie pusta.

**C2 (drugi filar — o modelu).**
Zero-shot audio-LLM (Qwen2-Audio) jako klasyfikator emocji mowy jest **skrajnie nierówny**
(accuracy 0.24–0.85 na 16 korpusach) i **zwija się ku negatywnym emocjom** (fear/sadness/anger;
surprise/happiness/disgust prawie nieużywane) na trudniejszych korpusach. **To NIE jest czysty
podział angielski-vs-reszta** (niemiecki pavoque 0.77 wysoko, angielskie emns 0.25 / jl_corpus 0.30
nisko). Dodatkowo: **preprocessing audio (resampling) mierzalnie wpływa** na wynik zero-shot
(te same klipy RAVDESS: 0.78 przy prostym resamplingu vs 0.85 przy natywnym 16 kHz).

**C3 (opisowy, wielojęzyczny — Badanie 2).**
CAMEO, 8 języków, κ text↔speech per język; pozycja PL; κ zależy od emocjonalności tekstu korpusu
bardziej niż od języka. **n=8 → wyłącznie eksploracyjne/opisowe, bez mocy inferencyjnej.**

**C4 (narzędziowy/reprodukowalność — realny wkład, nie tylko aneks).**
**Reużywalny, korpus- i model-agnostyczny framework do pomiaru zgodności międzymodalnej w
rozpoznawaniu emocji** — udokumentowany w [`docs/PIPELINE.md`](../docs/PIPELINE.md). Wartość to
**metodologia zaklęta w etapach** (ingest→predict→analyze→report): harmonizacja do przestrzeni
kanonicznej (Ekman-6+neutral) z fail-loud i jawnym raportowaniem odrzuceń non-Ekman; **reguła
wspólnego wsparcia** (κ liczone tylko na klasach wspieranych przez oba modele, z raportem
`n_used/n_excluded`); **role modeli** (czyste/headline vs leaky/secondary) egzekwowane wszędzie;
cache predykcji wersjonowany sygnaturą configu; statystyki (Cohen/Fleiss κ, bootstrap CI,
McNemar, dwie proporcje/χ², konflikty); deterministyczny raport. Dodanie nowego korpusu = 1 loader,
nowego modelu = 1 wpis w registry, modelu spoza maszyny (GPU-only) = `import-predictions`.
Zawiera też weryfikację i naprawę autorskich modeli mowy (`_v2`, held-out nEMO 0.887 / RAVDESS 0.713).
- **Opcja:** wydać jako **companion repo** przy artykule (recenzenci cenią). Do „software-paper"
  (np. JOSS) trzeba by dopolerować pakowanie — to osobny, mały nakład.
- **Uczciwie:** to nie nowość ML (modele gotowe), lecz reużywalny harness + metodologia; kod
  badawczy, nie zpakowana biblioteka.

## 3. Proponowana struktura + mapowanie wynik → artefakt

| sekcja | treść | artefakt |
|---|---|---|
| Wstęp | pytanie o zgodność multimodalną; przeramowanie | — |
| Dane | 4 zbiory (nEMO, RAVDESS, MELD, CAMEO), 58 975 próbek | `datasets_overview.md` (Tab. 1) |
| Modele | 10 modeli, 3 modalności, dokładność autorów | `models_overview.md` (Tab. 2) |
| Metody | harmonizacja, wspólne wsparcie, κ/bootstrap/McNemar/chi², leakage | `*.report.json`, opis §4/§5.4 |
| **Wyniki C1** | κ RAVDESS 0.00 vs MELD 0.20; trimodalnie (speech↔face 0.30); konflikty; baseline'y | **`article/cross_modal_kappa`**, **`article/modality_baselines`**, **`d1_headline_kappa`**, **`d3_text_cm_contrast`**, `conflict_*` |
| **Wyniki C2** | Qwen 0.24–0.85; kolaps na trudnych | **`d2_qwen_accuracy_collapse`**, `cm_*_speech_zeroshot` |
| Wyniki C3 | κ per język, PL, typologia | `study2/study2`, `study2_scatter` |
| Wyniki (uzup.) | istotność; twarz (RAVDESS trimod.); robustness bez neutral | `study1/pl_vs_en`, Fleiss (RAVDESS trio ≈ 0), kontrola §6 |
| **C4 (narzędzie)** | framework: komponenty, użycie, rozszerzanie | [`docs/PIPELINE.md`](../docs/PIPELINE.md), `datasets_overview`, `models_overview` |
| Ograniczenia | poniżej | — |
| Aneks | per-emocja, 26 macierzy CAMEO | `per_emotion_breakdown.csv`, `figures/study2/cm_*` |

## 4. Tytuł roboczy (opcje)

1. *„Cross-Modal Text–Speech Emotion Agreement Is Bounded by Text Emotionality: Evidence from
   Acted vs Natural Multilingual Corpora"* (rekomendacja — oddaje C1).
2. *„When the Text Says Nothing: Why Acted Speech Corpora Cannot Measure Text–Speech Emotion
   Agreement"* (mocniejszy, bardziej prowokacyjny).
3. *„Cross-Lingual Text–Speech Emotion Agreement for Social Robots: A Reality Check"* (najbliżej
   oryginalnego tytułu doktoratu, łagodniejszy pivot).

## 5. Abstrakt roboczy (szkic)

> Badamy zgodność między rozpoznawaniem emocji z tekstu i z mowy w czterech korpusach (polskim
> nEMO, angielskich RAVDESS i MELD, wielojęzycznym CAMEO; 58 975 wypowiedzi, 10 modeli w 3
> modalnościach). Pokazujemy, że w standardowych aktorskich korpusach zgodność text↔speech jest
> bliska zeru **nie dlatego, że modalności się nie zgadzają, lecz dlatego, że warstwa tekstowa jest
> emocjonalnie pusta**: na RAVDESS (stałe, neutralne zdania) model tekstowy przewiduje neutral dla
> wszystkich wypowiedzi (κ=0.00), podczas gdy na naturalnym dialogu MELD zgodność staje się
> mierzalna (κ=0.20), a modalność „bliższa gold" odwraca się z mowy na tekst. Pokazujemy też, że
> zero-shotowy audio-LLM jest skrajnie nierówny między korpusami (0.24–0.85) i zwija predykcje ku
> emocjom negatywnym na trudniejszych zbiorach. Wnioskujemy, że aktorskie korpusy SER są
> nieprzydatne do badań zgodności międzymodalnej, i formułujemy zalecenia dot. doboru danych dla
> multimodalnej percepcji emocji w robotach społecznych.

## 6. Ograniczenia (jawnie)

**Kontrola odpornościowa (wynik, nie ograniczenie):** wykluczenie próbek gold=neutral NIE zmienia
κ text↔speech (RAVDESS 0.00→0.00, MELD 0.202→0.203, nEMO 0.015→0.013, mimo że neutral to ~połowa
MELD) → niskie κ nie jest artefaktem dominującej klasy neutralnej.

- **Cross-lingual pozostaje słabo rozstrzygnięty:** MELD ratuje mierzalność tylko dla angielskiego;
  strona polska/wielojęzyczna (nEMO, CAMEO) nadal cierpi na pusty/aktorski tekst → wątek opisowy.
- **Mowa aktorska/TV** — ograniczona trafność ekologiczna; MELD zaszumione (śmiech, nakładanie).
- **Twarz:** RAVDESS 0.47 (FER→wideo domain shift); MELD 0.23 (largest-face, **nie zweryfikowany
  mówca** — drugorzędna/zaszumiona, nigdy w nagłówku).
- **Badanie 2:** n=8 → opisowe, bez inferencji.
- **`pl_text_roberta6`:** kolejność etykiet ZAŁOŻONA — tylko kontrola, nie nagłówek.
- **Confidence zero-shot** — niewiarygodne, pomijane w korelacjach.
- **Leakage** autorskich modeli mowy/LSTM na RAVDESS/nEMO — oznaczone; wnioski „bliżej gold" z modeli czystych.

## 7. Otwarte decyzje dla promotora

1. **Jak mocno prowadzić wynikiem negatywnym C1?** (metodologiczne ostrzeżenie jako główny wkład vs
   złagodzony pivot bliższy oryginalnemu tytułowi).
2. **Czy dążyć do „prawdziwego" cross-lingual** (dodać emocjonalny dialog nie-angielski, jeśli
   istnieje — trudne) czy zostać przy uczciwym „opisowo + kontrast EN"?
3. **Twarz MELD** — raportować best-effort z caveatem czy pominąć (zostawić twarz tylko RAVDESS)?
4. **Miejsce publikacji** — to determinuje, czy C1 (metodologia) czy C2 (audio-LLM) jest głównym haczykiem.
