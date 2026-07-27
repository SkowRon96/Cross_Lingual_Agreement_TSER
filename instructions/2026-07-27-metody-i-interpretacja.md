# Metody i interpretacja (proza do rozprawy/artykułu)

> Wersja robocza sekcji metodologicznej. Odwołania w formacie [Autor Rok] do uzupełnienia w
> bibliografii pracy. Liczby przykładowe zgodne z `results/`.

## Metody

### Przestrzeń etykiet i harmonizacja

Wszystkie korpusy sprowadzono do wspólnej, kanonicznej przestrzeni sześciu emocji Ekmana wraz z
kategorią neutralną — {gniew, wstręt, strach, radość, smutek, zaskoczenie, neutralny} [Ekman 1992].
Ponieważ zbiory różnią się natywnymi zestawami etykiet, mapowanie do przestrzeni kanonicznej
realizowano jedną, deterministyczną tablicą odwzorowań. Synonimy scalano wprost (np. *joy*/*happy*
→ radość, *angry* → gniew), a wybrane kategorie spoza kanonu Ekmana, lecz semantycznie do niego
sprowadzalne, łączono świadomie (*calm* → neutralny; *excitement*/*enthusiasm* → radość). Pozostałe
etykiety spoza kanonu (m.in. *sarcasm*, *poker*, *anxiety*, *hope*, *pride*, *apology*) odrzucano
(DROP), raportując liczbę odrzuceń dla każdego korpusu w raporcie harmonizacji. Odwzorowanie działa
w trybie *fail-loud*: napotkanie nieznanej etykiety przerywa przetwarzanie, co wyklucza ciche,
niekontrolowane potraktowanie nowej klasy. Dzięki temu korpusy o rozłącznych zestawach etykiet stają
się porównywalne, a każde wykluczenie jest jawne, nie przypadkowe.

### Reguła wspólnego wsparcia

Zgodność między dwoma modelami liczono wyłącznie na próbkach, których etykieta wzorcowa (gold) należy
do przecięcia klas obsługiwanych przez oba porównywane modele. Próbki, których gold leży poza tym
przecięciem, oraz próbki, na których którykolwiek z modeli wstrzymał się od decyzji (abstynencja),
wykluczano z danej pary; dla każdej pary raportowano liczbę próbek użytych (`n_used`) i wykluczonych
(`n_excluded`) wraz z powodem. Konkretnie: dla modeli tekstowych bez klasy *zaskoczenie*
(`pl_text_roberta6`, `xlmr_pooled10`) wykluczano próbki o gold = zaskoczenie; dla czteroklasowego
modelu mowy IEMOCAP (`en_speech_iemocap`) — próbki o gold ∈ {strach, wstręt, zaskoczenie}. Reguła
zapobiega karaniu modelu za klasę, której strukturalnie nie może wytworzyć, i jest jawnie
zaznaczona jako świadome ograniczenie, nie przeoczenie.

### Modele i ich role

Zastosowano dziesięć modeli w trzech modalnościach (tekst, mowa, twarz). Każda modalność ma model
**czysty/nagłówkowy** — niewytrenowany na danych ewaluacyjnych — oraz modele **drugorzędne/z wyciekiem
danych (leaky)**, trenowane na RAVDESS/nEMO lub obarczone założoną kolejnością etykiet. Wnioski
nagłówkowe (poziom zgodności, „która modalność bliżej wzorca") opierają się wyłącznie na modelach
czystych; modele drugorzędne wchodzą jedynie do pełnej macierzy κ i do kontroli rzetelności w obrębie
modalności. Zeroshotowy audio-LLM (Qwen2-Audio) pełni rolę czystego, nagłówkowego modelu mowy we
wszystkich korpusach, ponieważ nie był dostrajany na żadnym z nich.

### Potok obliczeniowy

Przetwarzanie zorganizowano w rozdzielone, cache'owane etapy (ingest → predict → analyze → report).
Predykcje są jedynym kosztownym artefaktem — liczone raz, wersjonowane sygnaturą konfiguracji i
wielokrotnie odczytywane przez analizę i raportowanie. Całość działa z ustalonym ziarnem
generatora losowego, co gwarantuje powtarzalność (m.in. bootstrapu).

### Miary zgodności i skuteczności

Zgodność par modalności mierzono współczynnikiem κ Cohena [Cohen 1960], a zgodność trójki modalności
(tekst–mowa–twarz w RAVDESS) — współczynnikiem κ Fleissa [Fleiss 1971]. Skuteczność pojedynczych
modeli opisywano dokładnością (accuracy), makro-uśrednionym F1 (macro-F1) oraz macierzą pomyłek.
Makro-F1 raportowano obok dokładności celowo: przy silnie niezrównoważonych klasach ujawnia on
sytuację, w której model ignoruje część kategorii (wysoka dokładność przy niskim macro-F1).

### Przedziały ufności

Dla każdego κ wyznaczono 95-procentowy przedział ufności metodą bootstrapu percentylowego
[Efron i Tibshirani 1993]: 1000 losowań ze zwracaniem po indeksach próbek, każdorazowo z ponownym
przeliczeniem κ, przy ustalonym ziarnie. Porównanie zgodności między językami (PL vs EN) opierano na
przedziale ufności **różnicy** κ, wyznaczonym z niezależnych losowań obu prób (próby pochodzą z
różnych korpusów, więc nie są sparowane).

### Testy istotności

Do porównania **poprawności** dwóch modalności na tych samych próbkach, w obrębie jednego korpusu,
użyto testu McNemara [McNemar 1947] (dla małej liczby wyników niezgodnych — dokładny test dwumianowy;
w pozostałych przypadkach — statystyka χ² z poprawką na ciągłość). Do porównań **niesparowanych**
między różnymi próbami (np. dokładność PL vs EN) użyto testu dwóch proporcji (z), nigdy McNemara —
rozróżnienie to jest istotne, gdyż McNemar wymaga par obserwacji na tych samych jednostkach.

### Analiza konfliktów i korelacja pewności

Analizę konfliktów prowadzono na próbkach, na których dwie modalności wydały **różne**, niepuste
predykcje: zliczano, która modalność jest bliżej wzorca, oraz rozkład konfliktów według emocji
wzorcowej. Korelację pewności modeli (Pearsona i Spearmana) liczono między modalnościami na
wspólnych, niepustych próbkach; pomijano modele zeroshotowe, których wskaźnik pewności jest
niewiarygodny.

### Badanie drugie (opisowe)

W badaniu wielojęzycznym (osiem języków, korpus CAMEO) wyznaczano κ tekst–mowa dla każdego języka i
odnoszono je opisowo do dystansu typologicznego od polskiego (URIEL/lang2vec [Littell i in. 2017])
oraz do klasy zasobności języka [Joshi i in. 2020]. Ze względu na n = 8 analiza ma charakter
eksploracyjno-opisowy i nie formułuje wniosków inferencyjnych.

## Interpretacja

**Jak czytać κ.** Współczynnik κ mierzy zgodność ponad poziom przypadku; wartość 0 oznacza zgodność
**na poziomie losowym**, a nie „brak danych". Do orientacyjnej interpretacji przyjęto skalę
[Landis i Koch 1977] (0–0,2 znikoma, 0,2–0,4 słaba, 0,4–0,6 umiarkowana, …), traktując ją jako
konwencję, nie próg twardy.

**Kluczowe zastrzeżenie interpretacyjne.** κ **łączy w jednej liczbie** dwa różne zjawiska: realną
niezgodność modalności oraz sytuację, w której jedna modalność jest niemal stała. Zgodność bliska
zeru, gdy model jednej modalności zapada się do jednej klasy (jak model tekstowy na aktorskim
RAVDESS, przewidujący *neutralny* dla wszystkich wypowiedzi), odzwierciedla **nieinformatywny kanał**,
a nie konflikt między modalnościami. Dlatego κ interpretujemy zawsze łącznie z: (i) makro-F1 i
macierzą pomyłek (które ujawniają zapadnięcie), (ii) analizą konfliktów (która wskazuje, która
modalność jest bliżej wzorca), oraz (iii) porównaniem korpusów kontrastujących ten sam język i te
same modalności, różniących się jedynie emocjonalnością warstwy tekstowej.

**Istotność.** κ uznajemy za istotnie różne od poziomu losowego, gdy jego przedział ufności nie
obejmuje zera; różnicę dwóch κ — gdy przedział różnicy nie obejmuje zera.

**Wyciek danych.** Dokładność modeli leaky na ich korpusie treningowym zawyża generalizację; wnioski
„bliżej wzorca" wyprowadzamy z modeli czystych. Analogicznie, dla modelu o **założonej** kolejności
etykiet (`pl_text_roberta6`) wyniki traktujemy wyłącznie jako kontrolne, nigdy nagłówkowe.

**Pewność zeroshot.** Wskaźnika pewności modeli zeroshotowych nie interpretujemy ilościowo i
pomijamy go w korelacjach.

**Zakres wnioskowania.** Badanie drugie (n = 8) jest opisowe; wszelkie prawidłowości między
zgodnością a dystansem typologicznym czy zasobnością języka formułujemy jako hipotezy do dalszej
weryfikacji, nie jako wyniki inferencyjne.
