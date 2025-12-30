# Projet: Classification Binaire et Entropie Croisée

## Description
Ce projet implémente un réseau de neurones peu profond pour la classification binaire.
L'objectif est d'étudier empiriquement la relation entre la vraisemblance (likelihood) 
et la negative log-likelihood (entropie croisée binaire).

## Architecture du réseau
- **Entrée**: scalaire x ∈ [0, 1]
- **Couche cachée**: 3 neurones avec activation ReLU
- **Couche de sortie**: 1 neurone linéaire + fonction sigmoid

## Structure du projet
```
projet_classification/
├── model.py              # Définition du réseau de neurones
├── metrics.py            # Calcul des métriques (likelihood, NLL)
├── visualization.py      # Fonctions de visualisation
├── main.py               # Script principal
├── requirements.txt      # Dépendances Python
└── README.md            # Ce fichier
```

## Installation

### 1. Créer un environnement virtuel (recommandé)
```bash
python3 -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Exécution

### Lancer le projet complet
```bash
python main.py
```

### Résultats générés
Le script génère automatiquement:
1. `sigmoid_output.png` - Courbe de probabilité P(y=1|x) avec les données
2. `likelihood_nll_variation.png` - Variation du likelihood et NLL en fonction de β₁

## Explications théoriques

### Forward Pass
Pour une entrée x:
1. Couche cachée: `z_hidden = Ω₀ * x + β₀`
2. Activation: `h = ReLU(z_hidden)`
3. Sortie: `f(x) = Ω₁ᵀh + β₁`
4. Probabilité: `P(y=1|x) = σ(f(x))` où σ est la fonction sigmoid

### Likelihood
La vraisemblance mesure la probabilité des données observées:
```
L = ∏ P(yi|xi)
```

### Negative Log-Likelihood (Binary Cross-Entropy)
```
NLL = -Σ [yi * log(P(yi=1|xi)) + (1-yi) * log(1-P(yi=1|xi))]
```

### Relation
Maximiser le likelihood ⟺ Minimiser le NLL

## Résultats attendus
Le projet montre que:
- Le likelihood est maximisé pour une valeur optimale de β₁
- Le NLL est minimisé exactement à la même valeur
- Ces deux métriques sont mathématiquement équivalentes

## Auteur
Projet d'Intelligence Artificielle - USTHB 2025/2026
