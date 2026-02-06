class Value:
    """
    Noeud du graphe de calcul.
    Stocke une valeur (data) et son gradient (grad).
    """

    def __init__(self, data, _sources=(), _op_symbol=''):
        self.data = data
        self.grad = 0.0  # dL/d(self)

        # Structure du graphe
        self._backward = lambda: None
        self._sources = set(_sources)  # Les variables utilisées pour créer celle-ci
        self._op_symbol = _op_symbol  # Symbole de l'opération (+, *, ReLU...)

    # ==========================================================================
    # 1. ADDITION : z = x + y
    # ==========================================================================
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        # Forward : z = x + y
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # --- MATHÉMATIQUES ---
            # Formule : dL/dx = (dL/dz) * (dz/dx)
            # Sachant que z = x + y, alors dz/dx = 1

            grad_externe = out.grad  # dL/dz (vient de la couche suivante)

            derivee_locale_self = 1.0  # dz/dx
            derivee_locale_other = 1.0  # dz/dy

            # Application de la Chain Rule
            self.grad += grad_externe * derivee_locale_self
            other.grad += grad_externe * derivee_locale_other

        out._backward = _backward
        return out

    # ==========================================================================
    # 2. MULTIPLICATION : z = x * y
    # ==========================================================================
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        # Forward : z = x * y
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # --- MATHÉMATIQUES ---
            # Formule : dL/dx = (dL/dz) * (dz/dx)
            # Sachant que z = x * y, alors dz/dx = y et dz/dy = x

            grad_externe = out.grad  # dL/dz

            derivee_locale_self = other.data  # dz/dx = y
            derivee_locale_other = self.data   # dz/dy = x

            # Application de la Chain Rule
            self.grad += grad_externe * derivee_locale_self
            other.grad += grad_externe * derivee_locale_other

        out._backward = _backward
        return out

    # ==========================================================================
    # 3. PUISSANCE : z = x^n
    # ==========================================================================
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "L'exposant doit être un scalaire"

        # Forward : z = x^n
        out = Value(self.data ** exponent, (self,), f'^{exponent}')

        def _backward():
            # --- MATHÉMATIQUES ---
            # Sachant que z = x^n, alors dz/dx = n * x^(n-1)

            grad_externe = out.grad  # dL/dz

            # dz/dx = n * x^(n-1)
            derivee_locale = exponent * (self.data ** (exponent - 1))

            # Chain Rule
            self.grad += grad_externe * derivee_locale

        out._backward = _backward
        return out

    # ==========================================================================
    # 4. ACTIVATION ReLU : z = max(0, x)
    # ==========================================================================
    def relu(self):
        # Forward
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # --- MATHÉMATIQUES ---
            # dz/dx = 1 si x > 0, sinon 0

            grad_externe = out.grad

            # Indicateur : 1 si actif, 0 si inactif
            derivee_locale = 1.0 if out.data > 0 else 0.0

            self.grad += grad_externe * derivee_locale

        out._backward = _backward
        return out

    # ==========================================================================
    # 5. ACTIVATION Sigmoid : z = 1 / (1 + e^(-x))
    # ==========================================================================
    def sigmoid(self):
        import math
        
        # Forward : z = 1 / (1 + e^(-x))
        sig = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(sig, (self,), 'Sigmoid')

        def _backward():
            # --- MATHÉMATIQUES ---
            # d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))

            grad_externe = out.grad

            # La dérivée de sigmoid est sig * (1 - sig)
            derivee_locale = out.data * (1 - out.data)

            self.grad += grad_externe * derivee_locale

        out._backward = _backward
        return out

    # ==========================================================================
    # MOTEUR DE RÉTROPROPAGATION (Backpropagation Engine)
    # ==========================================================================
    def backward(self):
        """
        Orchestre le calcul de tous les gradients en partant de la fin (self).
        """
        # 1. Tri Topologique pour s'assurer que les dépendances sont respectées
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for source in v._sources:
                    build_topo(source)
                topo.append(v)

        build_topo(self)

        # 2. Initialisation : Le gradient de la sortie par rapport à elle-même est 1
        # dL/dL = 1
        self.grad = 1.0

        # 3. Exécution inverse (de la fin vers le début)
        for node in reversed(topo):
            node._backward()

    # --- Opérations utilitaires (Négation, Soustraction, etc.) ---
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
