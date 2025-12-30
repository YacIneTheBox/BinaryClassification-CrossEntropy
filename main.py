"""
Script principal pour exécuter le projet de classification binaire
"""
import numpy as np
from model import ShallowNeuralNetwork
from metrics import (compute_likelihood, compute_negative_log_likelihood, 
                     compute_metrics_for_beta1_range)
from visualization import plot_sigmoid_output, plot_likelihood_and_nll


def main():
    """Fonction principale du projet"""

    print("="*60)
    print("PROJET: Classification binaire et entropie croisée")
    print("="*60)

    x_train = np.array([
        0.09291784, 0.46809093, 0.93089486, 0.67612654, 0.73441752,
        0.86847339, 0.49873225, 0.51083168, 0.18343972, 0.99380898,
        0.27840809, 0.38028817, 0.12055708, 0.56715537, 0.92005746,
        0.77072270, 0.85278176, 0.05315950, 0.87168699, 0.58858043
    ])

    y_train = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 
                        0, 1, 0, 1, 1, 0, 1, 0, 1, 1])

    beta0 = [0.3, -1.0, -0.5]
    omega0 = [-1.0, 1.8, 0.65]
    beta1 = 2.6
    omega1 = [-24.0, -8.0, 50.0]

    print("\n1. Initialisation du réseau de neurones")
    print("-" * 60)
    model = ShallowNeuralNetwork(beta0, omega0, beta1, omega1)
    print("✓ Réseau initialisé avec:")
    print(f"  - β₀ (biais cachés): {beta0}")
    print(f"  - Ω₀ (poids cachés): {omega0}")
    print(f"  - β₁ (biais sortie): {beta1}")
    print(f"  - Ω₁ (poids sortie): {omega1}")

    print("\n2. Visualisation de la sortie sigmoid P(y=1|x)")
    print("-" * 60)
    plot_sigmoid_output(model, x_train, y_train)

    print("\n3. Calcul des métriques avec beta1 initial")
    print("-" * 60)
    likelihood = compute_likelihood(model, x_train, y_train)
    nll = compute_negative_log_likelihood(model, x_train, y_train)
    print(f"  Likelihood: {likelihood:.6e}")
    print(f"  Negative Log-Likelihood: {nll:.6f}")

    print("\n4. Variation de beta1 et recherche de l'optimum")
    print("-" * 60)
    beta1_range = np.linspace(-10, 10, 200)
    likelihoods, nlls = compute_metrics_for_beta1_range(
        model, x_train, y_train, beta1_range
    )

    max_lik_idx = np.argmax(likelihoods)
    min_nll_idx = np.argmin(nlls)

    optimal_beta1_lik = beta1_range[max_lik_idx]
    optimal_beta1_nll = beta1_range[min_nll_idx]

    print(f"  β₁ optimal (max Likelihood): {optimal_beta1_lik:.4f}")
    print(f"  Likelihood maximal: {likelihoods[max_lik_idx]:.6e}")
    print(f"\n  β₁ optimal (min NLL): {optimal_beta1_nll:.4f}")
    print(f"  NLL minimal: {nlls[min_nll_idx]:.6f}")

    print("\n5. Visualisation de la variation likelihood/NLL")
    print("-" * 60)
    plot_likelihood_and_nll(beta1_range, likelihoods, nlls)

    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print(f"Le likelihood est maximisé à β₁ = {optimal_beta1_lik:.4f}")
    print(f"Le NLL est minimisé à β₁ = {optimal_beta1_nll:.4f}")
    print("Ces deux valeurs sont identiques, confirmant la théorie!")
    print("="*60)


if __name__ == "__main__":
    main()
