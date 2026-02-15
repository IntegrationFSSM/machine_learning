# Projet Titanic - MLOps & Machine Learning

## Description
Ce projet vise à prédire la survie des passagers du Titanic en utilisant plusieurs modèles de Machine Learning (Régression Logistique, Random Forest, SVM) et en intégrant des pratiques MLOps avec **MLflow** pour le suivi des expériences.

**Dépôt GitHub** : [https://github.com/IntegrationFSSM/machine_learning](https://github.com/IntegrationFSSM/machine_learning)

## Structure du Projet
```
PROJET-TITANIC/
├── data/                  # Données (téléchargées automatiquement ou placées ici)
├── src/                   # Code source
│   ├── data_loader.py     # Chargement des données
│   ├── preprocess.py      # Nettoyage et préparation
│   ├── train.py           # Entraînement et tracking MLflow
│   └── evaluate.py        # Évaluation et génération de graphiques
├── report/                # Rapport LaTeX et images générées
│   ├── images/            # Graphiques sauvegardés
│   └── rapport.tex        # Source LaTeX du rapport
├── main.py                # Script principal d'exécution
├── requirements.txt       # Dépendances Python
└── README.md              # Documentation
```

## Installation
1.  Cloner ce dépôt.
2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Pour exécuter le pipeline complet (chargement, préprocessing, entraînement, évaluation) :
```bash
python main.py
```

## Suivi MLflow
Pour visualiser les expériences et comparer les modèles :
```bash
mlflow ui
```
Ouvrir ensuite `http://127.0.0.1:5000` dans le navigateur.

## Auteur
Projet réalisé dans le cadre du Master / Projet Universitaire.
