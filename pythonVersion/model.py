"""
Module contenant l'architecture du réseau de neurones
Réseau peu profond: 1 couche cachée (3 neurones ReLU) + 1 couche de sortie
"""
import numpy as np


class ShallowNeuralNetwork:
    """
    Réseau de neurones peu profond pour classification binaire

    Architecture:
    - Entrée: scalaire x ∈ [0, 1]
    - Couche cachée: 3 neurones avec activation ReLU
    - Couche de sortie: 1 neurone linéaire + sigmoid pour probabilité
    """

    def __init__(self, beta0, omega0, beta1, omega1):
        """
        Initialise le réseau avec les paramètres donnés

        Args:
            beta0: biais de la couche cachée (3 valeurs)
            omega0: poids de la couche cachée (3 valeurs)
            beta1: biais de la couche de sortie (1 valeur)
            omega1: poids de la couche de sortie (3 valeurs)
        """
        self.beta0 = np.array(beta0)
        self.omega0 = np.array(omega0)
        self.beta1 = beta1
        self.omega1 = np.array(omega1)

    def relu(self, z):
        """Fonction d'activation ReLU: max(0, z)"""
        return np.maximum(0, z)

    def sigmoid(self, z):
        """Fonction sigmoid: 1 / (1 + exp(-z))"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward_pass(self, x):
        """
        Propagation avant dans le réseau

        Args:
            x: entrée(s) scalaire(s)

        Returns:
            f: sortie brute du réseau (avant sigmoid)
            h: activations de la couche cachée
        """
        if isinstance(x, np.ndarray) and x.ndim == 1 and len(x) > 1:
            z_hidden = np.outer(x, self.omega0) + self.beta0
            h = self.relu(z_hidden)
            f = np.dot(h, self.omega1) + self.beta1
        else:
            z_hidden = self.omega0 * x + self.beta0
            h = self.relu(z_hidden)
            f = np.dot(self.omega1, h) + self.beta1

        return f, h

    def predict_probability(self, x):
        """
        Prédit P(y=1|x) en appliquant sigmoid à la sortie

        Args:
            x: entrée(s) scalaire(s)

        Returns:
            probabilité(s) que y = 1
        """
        f, _ = self.forward_pass(x)
        return self.sigmoid(f)

    def set_beta1(self, new_beta1):
        """Modifie le biais de sortie beta1"""
        self.beta1 = new_beta1
