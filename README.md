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

La corelation de Pearson nous permet de voir sur une echelle de -1 (inverse) - 0 (aucun) - 1 (pareil) a quelle points deux variables d'un jeux de donne se resemble.

La formule doit etre applique entre toutes les combinaisons des variables de notre dataframe: 

$$
r = \frac{\sum_{i=1}^{n} (x_i - \overline{x}) (y_i - \overline{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2 \sum_{i=1}^{n} (y_i - \overline{y})^2}}
$$

Pourquoi mettre 0 au diagonal de la matrice :
Extraction de Caractéristiques ou de Relations Spécifiques

Si vous cherchez à identifier les paires de variables ayant la corrélation la plus élevée ou la plus basse, remplir la diagonale avec des zéros peut empêcher que les corrélations parfaites (de 1.0) des variables avec elles-mêmes ne faussent votre analyse ou vos algorithmes de sélection de caractéristiques.