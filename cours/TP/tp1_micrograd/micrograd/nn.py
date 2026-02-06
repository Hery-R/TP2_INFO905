import random

from micrograd.engine import Value


# ==============================================================================
#                           CLASSE DE BASE: MODULE
# ==============================================================================

class Module:
    """Classe parente pour gérer les paramètres et les gradients."""

    def zero_grad(self):
        """Remet à zéro le gradient de tous les paramètres (w, b)."""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """Retourne la liste des paramètres [w, b] à optimiser."""
        return []


# ==============================================================================
#                               NEURONE (Neuron)
# ==============================================================================

class Neuron(Module):
    """
    Un neurone artificiel unique.
    
    Notations Mathématiques :
    -------------------------
    1. Pré-activation (z) : z = (Σ w_i * x_i) + b
    2. Activation (a)     : a = ReLU(z)
    """

    def __init__(self, nin, nonlin=True):
        """
        Args:
            nin (int): Nombre d'entrées (Number IN) -> dimension de x
            nonlin (bool): Si True, applique la non-linéarité (ReLU)
        """
        # w : Vecteur de poids (initialisés aléatoirement)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

        # b : Biais (initialisé à 0)
        self.b = Value(0.0)

        self.nonlin = nonlin

    def __call__(self, x):
        """
        Forward Pass : x -> z -> a
        """
        # 1. Calcul de la pré-activation (z)
        # z = w * x + b
        # On utilise zip pour appairer chaque w_i avec son x_i
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # 2. Calcul de l'activation (a)
        if self.nonlin:
            return act.relu()
        else:
            return act

    def parameters(self):
        # Retourne les paramètres [w1, w2, ..., b]
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


# ==============================================================================
#                                COUCHE (Layer)
# ==============================================================================

class Layer(Module):
    """
    Une couche de neurones indépendants.
    
    Structure :
    Input x -> [Neurone 1, Neurone 2, ...] -> Output y (vecteur)
    """

    def __init__(self, nin, nout, **kwargs):
        """
        Args:
            nin (int): Nombre d'entrées venant de la couche précédente
            nout (int): Nombre de neurones dans cette couche (Number OUT)
        """
        # Création de la liste des neurones
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        Calcule la sortie de chaque neurone.
        Si la couche a plusieurs neurones, retourne un vecteur (liste).
        Si un seul neurone, retourne un scalaire.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Récupère les params (w, b) de tous les neurones de la couche
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


# ==============================================================================
#                         MLP (Multi-Layer Perceptron)
# ==============================================================================

class MLP(Module):
    """
    Le réseau complet. Une séquence de couches.
    """

    def __init__(self, nin, nouts):
        """
        Args:
            nin (int): Nombre d'entrées du réseau
            nouts (list): Liste des tailles de chaque couche
                          ex: [4, 4, 1] signifie 2 couches cachées de 4 et 1 sortie
        """
        sz = [nin] + nouts  # ex: [3, 4, 4, 1]

        self.layers = []

        for i in range(len(nouts)):
            # Création de la couche i
            # Elle prend sz[i] entrées et produit sz[i+1] sorties

            # La dernière couche est souvent linéaire (pas de ReLU)
            est_derniere = (i == len(nouts) - 1)

            self.layers.append(
                    Layer(sz[i], sz[i + 1], nonlin=not est_derniere)
                    )

    def __call__(self, x):
        """
        Forward Pass séquentiel.
        L'entrée x traverse chaque couche successivement.
        x -> Layer1 -> Layer2 -> ... -> y
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# ==============================================================================
#                           EXEMPLE D'EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # 1. Création du réseau
    # Entrée (3) -> Cachée (4) -> Cachée (4) -> Sortie (1)
    x_input = [2.0, 3.0, -1.0]  # Un vecteur d'entrée x
    model = MLP(nin=3, nouts=[4, 4, 1])

    # 2. Forward Pass (Prédiction)
    y_pred = model(x_input)
    print(f"Prédiction (y_pred) : {y_pred.data:.4f}")

    # 3. Calcul de la Loss (Erreur quadratique)
    y_target = 1.0  # La vraie valeur attendue
    loss = (y_pred - y_target) ** 2
    print(f"Perte (Loss)        : {loss.data:.4f}")

    # 4. Backward Pass (Calcul des gradients)
    model.zero_grad()  # Toujours remettre à zéro avant !
    loss.backward()

    print("\n--- Analyse d'un neurone de la 1ère couche ---")
    premier_neurone = model.layers[0].neurons[0]
    print(f"Poids w[0] : valeur={premier_neurone.w[0].data:.4f}, grad={premier_neurone.w[0].grad:.4f}")
    print(f"Si j'augmente w[0], la Loss va {'augmenter' if premier_neurone.w[0].grad > 0 else 'diminuer'}.")
