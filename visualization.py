"""
Module pour visualiser les résultats
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_sigmoid_output(model, x_train, y_train, output_file='sigmoid_output.png'):
    """
    Trace la courbe de probabilité P(y=1|x) et les données d'entraînement

    Args:
        model: instance de ShallowNeuralNetwork
        x_train: données d'entrée
        y_train: labels binaires
        output_file: nom du fichier de sortie
    """
    x_range = np.arange(0, 1.01, 0.01)
    prob_values = model.predict_probability(x_range)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, prob_values, 'b-', linewidth=2, 
             label='P(y=1|x) = σ(f(x))')
    plt.scatter(x_train, y_train, c='black', s=100, zorder=5, 
                edgecolors='black', linewidths=1.5, 
                label='Données d'entraînement')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probabilité P(y=1|x)', fontsize=12)
    plt.title('Sortie du réseau après sigmoid', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✓ Graphique sauvegardé: {output_file}")


def plot_likelihood_and_nll(beta1_range, likelihoods, nlls, 
                            output_file='likelihood_nll_variation.png'):
    """
    Trace les courbes de likelihood et NLL en fonction de beta1

    Args:
        beta1_range: array des valeurs de beta1
        likelihoods: array des likelihoods
        nlls: array des NLL
        output_file: nom du fichier de sortie
    """
    max_lik_idx = np.argmax(likelihoods)
    min_nll_idx = np.argmin(nlls)
    optimal_beta1_lik = beta1_range[max_lik_idx]
    optimal_beta1_nll = beta1_range[min_nll_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(beta1_range, likelihoods, 'g-', linewidth=2)
    ax1.axvline(optimal_beta1_lik, color='r', linestyle='--', 
                linewidth=2, label=f'Max à β₁={optimal_beta1_lik:.4f}')
    ax1.scatter([optimal_beta1_lik], [likelihoods[max_lik_idx]], 
                c='red', s=150, zorder=5)
    ax1.set_xlabel('β₁ (biais de sortie)', fontsize=12)
    ax1.set_ylabel('Likelihood', fontsize=12)
    ax1.set_title('Variation du Likelihood en fonction de β₁', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.plot(beta1_range, nlls, 'r-', linewidth=2)
    ax2.axvline(optimal_beta1_nll, color='g', linestyle='--', 
                linewidth=2, label=f'Min à β₁={optimal_beta1_nll:.4f}')
    ax2.scatter([optimal_beta1_nll], [nlls[min_nll_idx]], 
                c='green', s=150, zorder=5)
    ax2.set_xlabel('β₁ (biais de sortie)', fontsize=12)
    ax2.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax2.set_title('Variation du Negative Log-Likelihood en fonction de β₁', 
                  fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✓ Graphique sauvegardé: {output_file}")

    return optimal_beta1_lik, optimal_beta1_nll
