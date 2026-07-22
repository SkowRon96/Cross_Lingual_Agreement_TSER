# Instrukcja programistyczna — aplikacja badawcza

**Cross-Lingual Agreement Between Text and Speech Emotion Recognition: Towards Robust Multimodal Perception in Social Robots**

> ⚠️ **UWAGA!** To jest projekt badawczy — pod żadnym pozorem nie wolno zmyślać, konfabulować ani zmieniać wyników. Wszystkie badania muszą odzwierciedlać rzeczywisty stan rzeczy.

---

## 1. Źródło informacji

Głównym źródłem informacji jest [konspekt_artykulu_ESM.md](konspekt_artykulu_ESM.md), w którym zaplanowany jest cały proces badawczy. Bazuj na tych danych.

## 2. Cel aplikacji badawczej

- Prowadzenie prac badawczych nad rozpoznawaniem emocji w robotyce społecznej — jako materiał do artykułu oraz składowa doktoratu.
- Przygotowanie aplikacji (na podstawie konspektu), która przeprowadza analizy na modelach i zbiorach danych.
- Zebranie odpowiednich rezultatów oraz opracowanie wniosków dla problemu opisanego w konspekcie.

## 3. Kategorie / klasy emocji

- Trzymamy się podstawowej kategoryzacji emocji wg teorii Paula Ekmana: **radość, smutek, strach, złość (gniew), wstręt (odraza), zaskoczenie** + **brak emocji** (wypowiedź neutralna).
- Jeśli kategorie w różnych źródłach się różnią, na etapie planowania można spróbować je ujednolicić — **zapytaj mnie wtedy o decyzję i przedstaw propozycję**.
- Jeśli modele nie obsługują danej emocji, należy ją pomijać.

## 4. Kod i środowisko wykonawcze

- Kod pisany w Pythonie (zainstalowany 3.12); jeśli pojawi się rekomendacja nowszej wersji, można ją zainstalować.
- Tworzymy jeden `venv` dla całego projektu.
- Dbamy o optymalizację i wydajność. **Brak GPU — tylko Apple M1.** Jeśli coś będzie się wykonywać bardzo długo, dopuszczalne jest przeniesienie projektu na Windows + GPU.
- Przygotuj odpowiednie konfiguracje w `launch.json` — uruchamianie odbywa się głównie w trybie debug w VS Code.
- Używaj Pydantica.

## 5. Struktura folderów

| Folder | Przeznaczenie |
| --- | --- |
| `app` | Wszystkie skrypty pomocnicze oraz cała aplikacja badawcza. |
| `datasets` | Zbiory danych, na których bazujemy. **Uwaga:** zbiór RAVDESS nie został pobrany — jest duży i nie mieści się na dysku (do przemyślenia, co da się zrobić / co pobrać z https://zenodo.org/records/1188976). |
| `instructions` | Instrukcje tworzone przez developera oraz generowane checkpointy, plany itp. |
| `models` | Wszystkie modele; autorskie trzeba będzie wyekstrahować z `old_projects`. |
| `old_projects` | Stare projekty zawierające sporo kodu (również treningowego) — baza pod modele oraz wgląd w poprzednie badania. |
| `results` | Wyniki pogrupowane według badań (dobrze opisane i nazwane). |

## 6. Badania na obrazach twarzy

Ten aspekt należy oszacować:
- zbiór danych RAVDESS — co pobrać? jak wyciągać zdjęcia twarzy i odpowiadające im etykiety emocji?
- modele — czy w obecnym systemie (Apple M1, bez GPU) przeprowadzenie takiego badania jest możliwe i ile może zająć?

## 7. Repozytorium

Dla lepszej analizy zostanie stworzone repozytorium (musi być to koniecznie pierwszy krok, nawet przed planowaniem):
- przedstaw instrukcję
- wyklucz zbiory danych/modele/stare projekty
- old_projects nie mogą być przez Ciebie zmieniane
- przygotuj requirements.txt pod ponowne odtworzenie venva

## 8. Wyniki

- Wszystkie statystyki i testy badań opisane są w konspekcie artykułu.
- Zależy mi na tabelach, rysunkach, macierzach i grafikach nadających się do umieszczenia w artykule.
- Styl wyników musi być jednolity, oryginalny i naukowy.
- Tylko instrukcje `.md` są po polsku — reszta (kod, wyniki, testy) ma być po angielsku.
- Zapisuj checkpointy i chętnie pytaj o moje zdanie.
