"""
================================================================================
                    EXERCICE 3: ENTRAINEMENT D'UN MLP
================================================================================
                        Master 2 Informatique - Introduction IA
================================================================================

OBJECTIF :
Entrainer un reseau de neurones (MLP) sur le dataset "Moons" en utilisant
notre moteur autograd maison.

TODO: Implementer la fonction hinge_loss(y, y_preds)

    Formule mathematique (pour chaque exemple i):

        Loss_i = max(0, 1 - y_i * y_pred_i)

    Ou:
        - y_i        : label cible (-1 ou +1)
        - y_pred_i   : prediction du modele
        - max(0, x)  : fonction ReLU disponible via .relu() sur un objet Value

    Retourne une liste de Value contenant la loss pour chaque exemple.

Lancer ce script : python exo3_mlp_training.py
================================================================================
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from micrograd.engine import Value
from micrograd.nn import MLP


def hinge_loss(y, y_preds):
    losses = []
    
    for yi, y_pred in zip(y, y_preds):
        margin = yi * y_pred
        loss_i = (1 - margin).sigmoid()
        losses.append(loss_i)
    
    return losses


# =============================================================================
# PARTIE 3: ENTRAINEMENT D'UN MLP (Classification)
# =============================================================================
print("\n" + "=" * 80)
print(" PARTIE 3: Entrainement sur le dataset 'Moons'")
print("=" * 80)

# 1. Creation du Dataset
np.random.seed(1337)
random.seed(1337)
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Transformation des labels 0/1 en -1/+1 pour le SVM

print(f"Dataset : {len(X)} points. Entrees 2D (x1, x2). Sortie binaire (-1 ou +1).")

# 2. Initialisation du modele
# 2 entrees -> 16 neurones -> 16 neurones -> 1 sortie
model = MLP(nin=2, nouts=[16, 16, 1])
print(f"Modele : {model}")
print(f"Nombre de parametres (poids + biais) : {len(model.parameters())}")

# 3. Boucle d'optimisation
epochs = 100
learning_rate_init = 1.0  # Taux d'apprentissage initial (eta)

print("\nDemarrage de la Descente de Gradient...")

for k in range(epochs):

    # --- A. FORWARD PASS ---
    # On convertit les donnees numpy en objets Value
    inputs = [list(map(Value, x_row)) for x_row in X]

    # On passe tout le dataset dans le modele
    # map(model, inputs) execute model(x) pour chaque ligne x
    y_preds = list(map(model, inputs))

    # --- B. CALCUL DE LA LOSS (Max Margin / SVM Loss) ---
    # Loss_i = max(0, 1 - y_target * y_pred)
    # L'objectif est que (y_target * y_pred) soit > 1 (bonne classification avec marge)
    losses = hinge_loss(y, y_preds)

    data_loss = sum(losses) * (1.0 / len(losses))  # Moyenne

    # Regularisation L2 (optionnelle, pour eviter les poids trop grands)
    # Reg = alpha * somme(w^2)
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))

    total_loss = data_loss + reg_loss

    # Calcul de la precision (Accuracy) pour le suivi
    accuracy = [(yi > 0) == (score.data > 0) for yi, score in zip(y, y_preds)]
    acc = sum(accuracy) / len(accuracy)

    # --- C. ZERO GRAD ---
    # CRUCIAL : On remet les gradients a zero avant de calculer les nouveaux.
    # Sinon, ils s'accumuleraient avec ceux de l'etape precedente (+=).
    model.zero_grad()

    # --- D. BACKWARD PASS ---
    # C'est ici que la magie opere : propagation de "l'urgence"
    total_loss.backward()

    # --- E. UPDATE (Descente de Gradient) ---
    # w = w - eta * grad

    # Learning rate decroissant (le pas devient plus petit a la fin)
    eta = learning_rate_init * (1.0 - 0.9 * k / epochs)

    for p in model.parameters():
        p.data -= eta * p.grad

    if k % 10 == 0:
        print(f"Etape {k:3d} | Loss: {total_loss.data:.4f} | Precision: {acc * 100:.1f}% | LR: {eta:.4f}")

# Fin de l'entrainement
print(f"\nFIN. Loss finale : {total_loss.data:.4f}, Precision : {acc * 100:.1f}%")

# =============================================================================
# VISUALISATION
# =============================================================================
print("Generation du graphique de decision...")

h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='black')
plt.title(f"Frontiere de decision (Micrograd)\nLoss: {total_loss.data:.4f}, Acc: {acc * 100:.1f}%")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("decision_boundary.png")
print("Graphique sauvegarde sous 'decision_boundary.png'")
