# Welcome to DSLR

### Get dependency
```bash
./install.py
```

### Describe

```bash
./describe.py datasets/dataset_train.csv
```


## 2.2 Corelation de Pearson

To launch : 
```bash
./scatter_plot.py datasets/dataset_train.csv
```

La corelation de Pearson nous permet de voir sur une echelle de -1 (inverse) - 0 (aucun) - 1 (pareil) a quelle points deux variables d'un jeux de donne se resemble.

La formule doit etre applique entre toutes les combinaisons des variables de notre dataframe: 

$$
r = \frac{\sum_{i=1}^{n} (x_i - \overline{x}) (y_i - \overline{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2 \sum_{i=1}^{n} (y_i - \overline{y})^2}}
$$

Pourquoi mettre 0 au diagonal de la matrice :
Extraction de Caractéristiques ou de Relations Spécifiques

Si vous cherchez à identifier les paires de variables ayant la corrélation la plus élevée ou la plus basse, remplir la diagonale avec des zéros peut empêcher que les corrélations parfaites (de 1.0) des variables avec elles-mêmes ne faussent votre analyse ou vos algorithmes de sélection de caractéristiques.


## 3 Train et Predict

## Train

Avant de vouloir faire une prediction nous devont train notre algorithm pour avoir des poids.   
Poids qui nous servirons lors de la determination pour nos prediction.   

```
Explication math need here
```

## Usage Train
```bash
./logreg_train.py datasets/dataset_train.csv
```
```
Option :
-h : show information 
-show arg1 ... argn : list des maisons pour lequelles nous verrons le cost_history
-gradient arg : pour changer le type de descente de gradient que nous fesont
```

## Predict

Une fois que nous avons nos poids, avec un fichier similaire a celui de dataset_train.csv,   
Mais sans les 'Hogwarts House' definit pour les eleves, nous allons utiliser notre fichier de poids et les notes des eleves pour determiner a quelle maison ils appartiennent.

Pour cela nous normalisons les donnees et les preparons de la meme facons qui est faite dans le programme de Train 