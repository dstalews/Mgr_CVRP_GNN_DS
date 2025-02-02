Niniejsze repozytorium zawiera pliki utworzone i wykorzystane w trakcie pisania mojej pracy magisterskiej pt. "Zastosowanie grafowych sieci neuronowych do problemu wyznaczania tras z ograniczoną ładownością".

Katalogi:
-
- datasets - zawiera zbiory treningowe, testowe oraz walidacyjne dla każdej wielkości problemu (13,25,49)
- helpers - zawiera pliki z sieciami neuronowymi, algorytmami (przeszykiwanie wiązkowego, przeszukiwanie drzewa Monte-Carlo itp.) oraz inne skrypty i struktury wspomagające generację wyników
- models - zawiera wyszkolone modele zastosowane do generacji wyników końcowych
- plots - zawiera pliki .png z przykładowymi trasami
- plots_and_data - zawiera resztę wyników wygenerowanych w trakcie pisania pracy

Główne pliki:
-
- main.py - trening sieci
- main_load.py - wczytywanie wyszkolonego modelu i generacja rozwiązań końcowych
- main_just_plots.py - wczytywanie wyszkolonego modelu i generacja wybranych tras
