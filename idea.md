Główny motyw w pracy jest taki, że nie da się wytrenować dobrego modelu do detekcji "od 0". Ewentualnie da się, ale musi być to model o podobnej wielkości co modele generatywne (dziesiątki miliardów parametrów) trenowany na podobnej ilości danych co one - dla nas niewykonalne. Bierze się to głownie stąd, że każda firma tworząca modele ma swoje własne dane. A co za tym idzie modele tworzą różne reprezentacje (nie wiem czy to najlepsze słowo) tych danych. Więc ten model do detekcji musiałby być wystarczająco duży aby zapamiętać(?) większość tych reprezentacji (może stylów to lepsze słowo?) i do tego trenowany na wystarczająco dużej ilości danych aby wyciągnąć jak największą część tych reprezentacji. Danych musi być dużo, jako, że modele generatywne są duże. Stąd pojawił się pomysł o detekcji czy tekst został wygenerowany przez jeden ustalony model czy nie, więcej w 4) Drugi Fine tuning.

# 1. Wybranie modelów
Wybranie lokalnych generatywnych modelów, wstępnie około 20 (+ ewentualnie API OpenAI). Pewnie do wielkości 70B (nie wiem jakie ograniczenia sprzętowe mamy). Wybór tych modelów jest bardzo ważny, ponieważ od tego zależy jakie porównania będą możliwe w 4).

# 2. Dane:
Potrzebujemy tekstów napisanych przez człowieka i wygenerowanych przez różne modele.

a) Napisane przez człowieka:

Wstępnie znalazłem to: https://ai.google.com/research/NaturalQuestions.
Na pewno przydałyby się też teksty innego rodzaju niż odpowiedzi na pytania. Na przykład, bardziej zbliżone do wpisów na mediach społecznościowych.

b) Wygenerowane:

Tutaj raczej będzie trzeba wygenerować samemu. Z każdego modelu z 1) podobną liczba próbek (jak liczyć próbki? liczba zdań? liczba tokenów? pewnie zależy od finalnego zdefiniowania co klasyfikujemy). Chciałbym, żeby były generowane z różnymi parametrami (temperatura, top_p, top_k).
Też, dobrze by było, żeby były jak najbardziej różnorodne, cześć odpowiedzi na pytania, cześć wpisy na mediach społecznościowych, itp.

# 3. EDA Danych:
Tutaj pewnie można policzyć dużo metryk/feature'ów na podstawie tekstów. Wydaje mi się, że tutaj Pani Agnieszka może bardzo pomóc.

# 4. Eksperymenty:

Przede wszystkim tutaj trzeba ustalić co dokładnie klasyfikujemy (na pewno będzie to klasyfikacja binarna tekstu), widzę 3 możliwości:

a) 1 predykcja dla całego tekstu (nie został / został wygenerowany przez AI)

b) 1 predykcja dla każdego zdania (nie zostało / zostało wygenerowany przez AI), tutaj może pojawić się problem, że często ciężko po pojedynczym zdaniu ocenić

c) 1 predykcja dla każdego zdania, ale w kontekście całego tekstu. Przewaga nad b) jest taka, że mamy dostęp do całego tekstu podczas wykonywania predykcji dla ustalonego zdania. To jest troszkę trudniejsze do zaimplementowania, ale wydaj mi się, że najlepsza opcja.

## Generalne uwagi do tworzenia dataset'ów do poniższych eksperymentów:
-Raczej każdy dataset będzie podzbiorem wszystkich danych zebranych w 2)

-Chciałbym żeby proporcje, różnych rodzajów tekstów, chociaż trochę były zbliżone do takich jak występują w praktyce (przykładowo wśród odpowiedzi na pytania 20% to wygenerowane teksty, a we wpisach z mediów społecznościowych 40%, pewnie potrzebne będzie jakieś źródło do dokładnych %)

-Ustalić, czy wśród tekstów wygenerowanych przez modele, liczba tekstów wygenerowany przez każdy model z 1) powinna być taka sama (czy może jakoś powiązana z popularnością modelu)

-Proporcje TRAIN-VAL-TEST - 60:20:20? (do ustalenia)

## Baseline:
Tutaj po części chciałbym potwierdzić motyw z początku. Ustalić np. 4 wielkości modelu i 4. wielkości zbioru treningowego (wielkość modelu/datasetu rośnie wykładniczo; zawsze ten sam VAL i TEST do ewaluacji). Wytrenować te 16 modelów "od 0" (zgodnie z tym co ustaliliśmy na początku 4)) i pokazać, że potrzeba o wiele więcej parametrów i danych, aby wyniki były zadowalające, przynajmniej mam nadzieję, że tak wyjdzie... :)
Ewentualnie jakaś prosta regresja liniowa na podstawie feature'ów z 3).

## Pierwszy Fine-tuning:
Dataset: Pewnie jedna z wersji z dataset'ów z Baseline'u. Na pewno ten sam zbiór VAL I TEST, aby móc porównywać. Ten sam zbiór dla wszystkich training run'ów.

Tutaj fine-tun'ujemy każdy z modelów z 1) zgodnie z tym co ustaliliśmy na początku 4). Na 99% każdy będzie lepszy od wyników z baseline'u. Już tutaj można dać pierwsze porównania. Jak metryki zależą od serii modelu, liczby parametrów, zajmowanego VRAMu, kwantyzacji, itp.

Tutaj też mam nadzieję, że wystąpi następujące zjawisko (wydaje mi się całkiem logiczne, że tak powinno być), na przykładzie:
Powiedzmy, że jednym z modelów z 1) jest llama3.1:8b. Po zfine-tun'owaniu tego modelu (do klasyfikatora/detektora) powinno się okazać, że jest on najlepszy w detekcji tekstów właśne z llama3.1:8b.

Generalnie: Ustalony LLM po zfine-tun'owaniu jest najlepszy w detekcji tekstów generowanych wlasnie przez tego samego LLMa (tylko przed zfinetunowaniem, jak byl jeszcze generatorem)


## Stąd pojawia, się pomysł na drugą serię fine-tun'ingu:

Postępowanie będzie analogiczne dla każdego modelu z 1)

Będziemy fine-tunowac wybrany model jako detektor - czy tekst został wygenerowany przez ten ustalony model czy nie. Dalej jest to klasyfikacja binarna tekstu, taka sama jak przy pierwszej serii. 

Dataset składa się z tekstów wygenerowanych przez ten ustalony model (label 1), losowej probki z reszty tekstów (label 0). 

Efektem drugiej serii trenowanie, będzie komplet klasyfikatorów decydujących czy wybrany tekst został wygenerowany przez ustalony model czy nie.

W kontekście detekcji czy tekst został wygenerowany przez AI czy napisany przez człowieka, można powiedzieć, że jezli wszystkie detektory stwierdzą, że tekst nie został wygenerowany przez odpowiadający im model, to został on napisany przez człowieka. 

Porównania podobne jak przy pierwszej serii - jak metryki zależą od serii modelu, liczby parametrów, zajmowanego VRAMu, kwantyzacji, itp.

Taki "ensamble" modelów powinien być bardziej skuteczny niż modele z pierszej serii fine-tuningu. Problem tego rozwiązania polega na tym, że w praktyce modelów generatywnych jest bardzo dużo, wiąże się to z bardzo dużą liczbą klasyfikatorów potrzebnych do wytrenowania. Dla przykładu, modelów z serii llama3.1 jest 44 (licząc 3 rozmiary, i wszystkie kwantyzacje, źródło: https://ollama.com/library/llama3.1/tags). 

## Stąd pomysł na trzecią, ostatnią serię trenowania:

W zasadzie, tutaj się zastanawiam czy nie możemy pominąć drugiej serii i przejść z 1 od razu do 3., ponieważ druga i trzecia są bardzo podobne. Ale wydaje mi się, że tak jak jest teraz, to proces myślowy jest lepiej pokazany.

Postępowanie będzie analogiczne dla każdego modelu z 1)

Będziemy fine-tunowac wybrany model jako detektor - czy tekst został wygenerowany przez ustaloną serię modelów czy nie. Dalej jest to klasyfikacja binarna tekstu, taka sama jak przy pierwszej serii. 

Dataset składa się z tekstów wygenerowanych przez ustaloną serię modelów (label 1), losowej próbki z reszty tekstów (label 0). 

Przykładowo wśród modelów z 1) mamy modele z serii: llama3.1, llama.3.2, phi3.5, qwen2 i qwen2.5. Wszystkie modele z serii llama 3.1, będą trenowane jako klasyfikator czy tekst jest z (któregokolwiek modelu z serii)llama3.1 czy nie. Analogicznie dla wszystkich serii. 

Najprawdopodobniej wyjdzie tak, że dla każdej serii średnio najlepszy będzie największy model z tej ustalonej serii wybrany przez nas, ale zobaczymy...

Taki "ensamble" modelów powinien być trochę mniej skuteczny niż "ensamble" z drugiej serii fine-tuningu. Natomiast będzie się składał, z dużo mniejszej liczby modelów. W praktyce, ze względu na liczbę modelów generatywnych, tylko takie podejście jest wykonalne. Niewielka strata na jakości predykcji, jest warta znacznego zmniejszania kosztów obliczeniowych. Ten "ensamble" będzie ostatnim przygotowanym przez nas detektorem.
Porównania podobne jak przy pierwszej serii - jak metryki zależą od serii modelu, liczby parametrów, zajmowanego VRAMu, kwantyzacji, itp.

## Przemyślenia:

Ten ostatni "ensamble" można podłączyć do prostej aplikacji (np. w streamlicie), tak aby było dostępne z UI.

Tutaj ta końcówka kojarzy mi się trochę z feature selection. Tak jak przy feature selection wybieramy jak najmniejszy podzbiór zmiennych aby osiągnąć jak najlepszą jakość predykcji. Tak tutaj chcemy wybrać jak najmniejszy podzbiór modeli do zfine-tunowania, aby osiągnąć jak najlepszą jakość predykcji. Być może można tutaj coś więcej pod tym kontem spróbować, ale to na razie takie luźnie myśli...

Wydaje mi się, że można też liczyć (przy każdej serii fine-tunowania) metryki na jakimś benchmarku, tak aby można było porównywać z literaturą.

Jedyny problem jest taki, że to podejścia działa tylko do modelów lokalnych. Jeżeli chodzi o modele dostępne tylko przez API to jeszcze troszkę muszę się zastanowić jak do tego podejść...


