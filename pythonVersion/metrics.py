"""
Module pour calculer les métriques: likelihood et negative log-likelihood
"""

import numpy as np


def compute_likelihood(model, x_train, y_train):
    """
    Calcule la vraisemblance (likelihood) du modèle sur les données

    Likelihood = ∏ P(yi|xi) où P est la probabilité prédite

    Args:
        model: instance de ShallowNeuralNetwork
        x_train: données d'entrée
        y_train: labels binaires (0 ou 1)

    Returns:
        likelihood: produit des probabilités
    """
    probabilities = model.predict_probability(x_train)

    likelihood = 1.0
    for i in range(len(y_train)):
        p = probabilities[i]
        if y_train[i] == 1:
            likelihood *= p
        else:
            likelihood *= 1 - p

    return likelihood


def compute_negative_log_likelihood(model, x_train, y_train):
    """
    Calcule la negative log-likelihood (NLL) = Binary Cross-Entropy

    NLL = -Σ [yi * log(p) + (1-yi) * log(1-p)]

    Args:
        model: instance de ShallowNeuralNetwork
        x_train: données d'entrée
        y_train: labels binaires (0 ou 1)

    Returns:
        nll: negative log-likelihood
    """
    probabilities = model.predict_probability(x_train)

    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    nll = -np.sum(
        y_train * np.log(probabilities) + (1 - y_train) * np.log(1 - probabilities)
    )

    return nll


def compute_metrics_for_beta1_range(model, x_train, y_train, beta1_range):
    """
    Calcule likelihood et NLL pour une gamme de valeurs de beta1

    Args:
        model: instance de ShallowNeuralNetwork
        x_train: données d'entrée
        y_train: labels binaires
        beta1_range: array de valeurs de beta1 à tester

    Returns:
        likelihoods: array des likelihoods
        nlls: array des NLL
    """
    original_beta1 = model.beta1

    likelihoods = []
    nlls = []

    for b1 in beta1_range:
        model.set_beta1(b1)
        lik = compute_likelihood(model, x_train, y_train)
        nll = compute_negative_log_likelihood(model, x_train, y_train)
        likelihoods.append(lik)
        nlls.append(nll)

    model.set_beta1(original_beta1)

    return np.array(likelihoods), np.array(nlls)
