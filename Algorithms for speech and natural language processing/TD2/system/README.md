# PCFG & CKY

Implémentation d'un parser probabiliste pour l’etiquetage morpho-syntaxique

## Installation

Ce parser tourne sous python >= 3, utilisez pip pour installer les dépendances
```
pip3 install -r requirements.txt
```
## Usage

Utilisez le script `run.sh` pour parser des phrases. 
```
./run.sh
```
Il est à noter que ce script utilise la commande `python3`, si vous utiliser `python` à la place, merci de le modifier dans le script. Le script prend une ligne à la fois, chaque ligne est composée de tokens séparés par des espaces, et renvoie le parsing de ces derniers. Exemple:
```
Please enter phrase to parse!
>>> je suis un garçon .
Took 0s
( SENT ( VN ( CLS je ) ( V suis ) ) ( NP ( DET un ) ( NC garçon ) ) ( PONCT . ) )
>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<

Please enter phrase to parse!
>>> Affaire des HLM de Paris
Took 0s
( SENT ( NC Affaire ) ( PP ( P+D des ) ( NP ( NC HLM ) ( PP ( P de ) ( NP Paris ) ) ) ) )
>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<

Please enter phrase to parse!
>>> 
```
