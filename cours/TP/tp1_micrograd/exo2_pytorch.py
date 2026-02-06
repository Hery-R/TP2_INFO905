"""
================================================================================
                    EXERCICE 2: COMPARAISON AVEC PYTORCH
================================================================================
                        Master 2 Informatique - Introduction IA
================================================================================

OBJECTIF :
Valider que notre moteur autograd calcule les memes gradients que PyTorch.
Il y a une erreur ici, trouvez-la!

Lancer ce script : python exo2_pytorch.py
================================================================================
"""

import torch

from micrograd.engine import Value

# =============================================================================
# PARTIE 2: COMPARAISON AVEC PYTORCH
# =============================================================================
print("\n" + "=" * 80)
print(" PARTIE 2: Validation Industrielle (vs PyTorch)")
print("=" * 80)

print("On va comparer le calcul de gradient sur une expression complexe.")
print("Expression : L = (a * b + 1)^2")

# --- MICROGRAD ---
a_mg = Value(2.0)
b_mg = Value(3.0)
L_mg = (a_mg * b_mg + 1) ** 2
L_mg.backward()

# --- PYTORCH ---
a_pt = torch.tensor([2.0], requires_grad=True)
b_pt = torch.tensor([3.0], requires_grad=True)
L_pt = (a_pt * b_pt + 1) ** 2
L_pt.backward()

print(f"\nResultats :")
print(f"Micrograd :: L={L_mg.data:.4f} | dL/da={a_mg.grad:.4f} | dL/db={b_mg.grad:.4f}")
print(f"PyTorch   :: L={L_pt.item():.4f} | dL/da={a_pt.grad.item():.4f} | dL/db={b_pt.grad.item():.4f}")

if abs(a_mg.grad - a_pt.grad.item()) < 1e-4:
    print("\n SUCCES : Votre moteur calcule exactement comme PyTorch !")
else:
    print("\n ERREUR : Divergence detectee.")