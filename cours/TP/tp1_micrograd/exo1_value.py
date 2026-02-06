"""
================================================================================
                    EXERCICE 1: DECOUVERTE DE LA CLASSE VALUE
================================================================================
                        Master 2 Informatique - Introduction IA
================================================================================

OBJECTIF :
Comprendre physiquement ce qu'est le gradient : un "Signal d'Urgence" qui remonte
le temps pour indiquer comment corriger les erreurs.

TODO Il y a des erreur de calcule dans cours/TP/tp1_micrograd/micrograd/engine.py !!

Lancer ce script : python exo1_value.py
================================================================================
"""

from micrograd.engine import Value

# =============================================================================
# PARTIE 1: DECOUVERTE DE LA CLASSE VALUE
# =============================================================================
print("\n" + "=" * 80)
print(" PARTIE 1: Le Gradient comme Sensibilite")
print("=" * 80)

print("Objectif : Calculer dL/da et dL/db pour l'expression L = a * b + c")

# 1. Definition des entrees
a = Value(2.0, _op_symbol='a')
b = Value(-3.0, _op_symbol='b')
c = Value(10.0, _op_symbol='c')

# 2. Construction du graphe (Forward Pass)
e = a * b  # e = -6
L = e + c  # L = 4

print(f"\nForward Pass :")
print(f"  a={a.data}, b={b.data}, c={c.data}")
print(f"  e = a * b = {e.data}")
print(f"  L = e + c = {L.data} (Notre 'Loss' finale)")

# 3. Calcul des gradients (Backward Pass)
# On initialise l'urgence a la fin : dL/dL = 1
L.backward()

print(f"\nBackward Pass (Resultats) :")
print(f"  dL/dL = {L.grad} (L'urgence initiale)")
print(f"  dL/dc = {c.grad} (L depend lineairement de c -> grad=1)")
print(f"  dL/de = {e.grad} (L depend lineairement de e -> grad=1)")
print(f"  dL/da = {a.grad} (Devrait etre b = -3.0)")
print(f"  dL/db = {b.grad} (Devrait etre a = 2.0)")

print("\nAnalyse :")
print("Si dL/da = -3, cela signifie que si j'augmente 'a' un tout petit peu,")
print("L va diminuer fortement (car le gradient est negatif).")