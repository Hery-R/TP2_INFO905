import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reseaux de Neurones : De la Theorie a la Pratique

    ## Master 2 Informatique - Introduction a l'IA

    ---

    ### Plan du cours

    1. **Le Neurone Artificiel** - Du biologique au mathematique
    2. **Fonctions d'Activation** - Non-linearite et expressivite
    3. **Descente de Gradient** - Optimisation et apprentissage
    4. **Backpropagation** - La regle de la chaine en action
    5. **Multi-Layer Perceptron (MLP)** - Architecture profonde
    6. **Demo Interactive** - Micrograd en pratique
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Le Neurone Artificiel

    ### Du biologique au mathematique

    Le neurone biologique recoit des signaux electriques via ses **dendrites**,
    les integre dans le **corps cellulaire**, et si le signal depasse un seuil,
    transmet une impulsion via son **axone**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Le modele mathematique

    Le neurone artificiel (perceptron) imite ce comportement :

    $$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^T \mathbf{x} + b)$$

    Ou :
    - $\mathbf{x} = (x_1, x_2, ..., x_n)$ : vecteur d'entree (les "dendrites")
    - $\mathbf{w} = (w_1, w_2, ..., w_n)$ : poids synaptiques (force des connexions)
    - $b$ : biais (seuil d'activation)
    - $f$ : fonction d'activation (decision de "tirer" ou non)
    - $y$ : sortie du neurone
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🕹️ Labo Interactif : La Frontière de Décision

    Jouez avec les paramètres pour comprendre leur rôle géométrique :
    - **$w_1$ et $w_2$** : Ils déterminent l'**angle** (la rotation) de la ligne.
    - **$b$ (Biais)** : Il détermine la **position** (le décalage) de la ligne par rapport à l'origine.
    """)
    return


@app.cell
def _(mo):
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_interactive_neuron(w1, w2, b):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Données linéairement séparables (Génération fixe via seed)
        np.random.seed(42)
        class_0 = np.random.randn(50, 2) + np.array([2, 2])
        class_1 = np.random.randn(50, 2) + np.array([-2, -2])

        # Affichage des points
        ax.scatter(class_0[:, 0], class_0[:, 1], c='blue', label='Classe 0', alpha=0.7, edgecolors='k')
        ax.scatter(class_1[:, 0], class_1[:, 1], c='red', label='Classe 1', alpha=0.7, edgecolors='k')

        # --- Calcul de la Frontière de décision ---
        # Equation : w1*x1 + w2*x2 + b = 0
        # Donc : x2 = -(w1*x1 + b) / w2

        x_line = np.linspace(-6, 6, 100)

        if w2 != 0:
            y_line = -(w1 * x_line + b) / w2
            label = f'{w1} $x_1$ + {w2} $x_2$ + {b} = 0'
            ax.plot(x_line, y_line, 'g-', linewidth=3, label=label)

            # Coloration des zones (fond)
            # Pour visualiser la zone "Classe 0" vs "Classe 1"
            y_min, y_max = -6, 6
            ax.fill_between(x_line, y_line, y_max if w2 > 0 else y_min, color='blue', alpha=0.1)
            ax.fill_between(x_line, y_line, y_min if w2 > 0 else y_max, color='red', alpha=0.1)

        else:
            # Cas particulier ligne verticale (w2 = 0)
            x_vert = -b / w1 if w1 != 0 else 0
            ax.axvline(x=x_vert, color='g', linewidth=3, label=f'Frontière Verticale ($x_1$={x_vert:.2f})')

        # Paramètres graphiques
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('$x_1$ (Entrée 1)')
        ax.set_ylabel('$x_2$ (Entrée 2)')
        ax.legend(loc='upper right')
        ax.set_title(f'Visualisation : $w_1={w1}, w_2={w2}, b={b}$')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    # Création des sliders
    w1_slider = mo.ui.slider(start=-5.0, stop=5.0, step=0.1, value=1.0, label="Poids w1")
    w2_slider = mo.ui.slider(start=-5.0, stop=5.0, step=0.1, value=1.0, label="Poids w2")
    b_slider = mo.ui.slider(start=-10.0, stop=10.0, step=0.5, value=0.0, label="Biais b")
    return b_slider, np, plot_interactive_neuron, plt, w1_slider, w2_slider


@app.cell
def _(b_slider, mo, w1_slider, w2_slider):
    # Affichage groupé
    mo.vstack(
            [
                mo.md("**Paramètres du Neurone**"),
                w1_slider,
                w2_slider,
                b_slider
                ]
            )
    return


@app.cell
def _(b_slider, plot_interactive_neuron, w1_slider, w2_slider):
    # Appel de la fonction avec les valeurs actuelles des sliders
    plot_interactive_neuron(w1_slider.value, w2_slider.value, b_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Fonctions d'Activation

    ### Pourquoi avons-nous besoin de non-linearite ?

    Sans fonction d'activation non-lineaire, un reseau de N couches
    est equivalent a une seule transformation lineaire :

    $$f(g(x)) = W_2(W_1 x) = (W_2 W_1) x = W' x$$

    La non-linearite permet d'apprendre des frontieres de decision complexes !
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    def plot_activations():
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        x = np.linspace(-5, 5, 200)

        # Sigmoid
        sigmoid = 1 / (1 + np.exp(-x))
        axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2)
        axes[0, 0].set_title('Sigmoid: $\\sigma(x) = \\frac{1}{1+e^{-x}}$')
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 0].grid(True, alpha=0.3)

        # Tanh
        tanh = np.tanh(x)
        axes[0, 1].plot(x, tanh, 'g-', linewidth=2)
        axes[0, 1].set_title('Tanh: $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$')
        axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        # ReLU
        relu = np.maximum(0, x)
        axes[0, 2].plot(x, relu, 'r-', linewidth=2)
        axes[0, 2].set_title('ReLU: $\\max(0, x)$')
        axes[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 2].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 2].grid(True, alpha=0.3)

        # Derivees
        sigmoid_grad = sigmoid * (1 - sigmoid)
        axes[1, 0].plot(x, sigmoid_grad, 'b--', linewidth=2)
        axes[1, 0].set_title("Derivee Sigmoid: $\\sigma'(x) = \\sigma(x)(1-\\sigma(x))$")
        axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(-0.1, 0.5)

        tanh_grad = 1 - tanh ** 2
        axes[1, 1].plot(x, tanh_grad, 'g--', linewidth=2)
        axes[1, 1].set_title("Derivee Tanh: $1 - \\tanh^2(x)$")
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        relu_grad = (x > 0).astype(float)
        axes[1, 2].plot(x, relu_grad, 'r--', linewidth=2)
        axes[1, 2].set_title("Derivee ReLU: $\\mathbb{1}_{x>0}$")
        axes[1, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 2].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(-0.1, 1.5)

        plt.tight_layout()
        return fig

    mo.md("### Les fonctions d'activation courantes et leurs derivees")
    return (plot_activations,)


@app.cell
def _(plot_activations):
    plot_activations()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparaison des fonctions d'activation

    | Fonction | Avantages | Inconvenients |
    |----------|-----------|---------------|
    | **Sigmoid** | Sortie bornee [0,1], probabiliste | Vanishing gradient, pas centree sur 0 |
    | **Tanh** | Centree sur 0, sortie [-1,1] | Vanishing gradient |
    | **ReLU** | Simple, pas de vanishing gradient | "Dying ReLU" (neurones morts) |
    | **Leaky ReLU** | Evite les neurones morts | Hyperparametre supplementaire |
    | **GELU/SiLU** | Smooth, performant en pratique | Plus couteux a calculer |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Descente de Gradient

    ### Le probleme d'optimisation

    Objectif : trouver les poids $\mathbf{w}$ qui minimisent une fonction de cout $\mathcal{L}$

    $$\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{w})$$

    **Idee intuitive** : Imaginez-vous sur une montagne dans le brouillard.
    Pour descendre, vous tatez le sol autour de vous et avancez dans la direction
    qui descend le plus.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📉 L'algorithme de Descente de Gradient

    L'objectif est de minimiser la fonction de coût $\mathcal{L}(\mathbf{w})$.

    1. **Initialisation** : Choisir un vecteur de poids initial $\mathbf{w}^{(0)}$ (aléatoire).
    2. **Itération** : Répéter jusqu'à la convergence :
       - Calculer le gradient : $\nabla_{\mathbf{w}} \mathcal{L}$
       - Mettre à jour les poids :
    $\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} \mathcal{L}$

    ---

    #### 📝 Résumé des notations

    | Symbole | Terme | Signification |
    | :---: | :--- | :--- |
    | $\mathbf{w}$ | **Vecteur de poids** | Les paramètres du modèle que l'on cherche à optimiser. |
    | $\eta$ | **Taux d'apprentissage** | (*Learning rate*) Contrôle la taille du pas à chaque itération. |
    | $\mathcal{L}$ | **Fonction de coût** | (*Loss function*) Mesure l'erreur entre la prédiction et la réalité. |
    | $\nabla$ | **Gradient** | Vecteur indiquant la direction de la pente la plus raide. |
    | $\partial$ | **Dérivée partielle** | Variation de l'erreur par rapport à un seul poids spécifique. |
    """)
    return


@app.cell
def _(mo, np, plt):
    def plot_gradient_descent():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # =====================================================================
        # PARTIE 1 : Descente de gradient en 1D (un seul poids)
        # =====================================================================

        # --- ETAPE 1 : Definir la fonction de perte ---
        # On utilise L(w) = w^2, une parabole avec minimum en w=0
        # C'est une simplification pedagogique (en vrai : MSE, Cross-Entropy...)
        w_values = np.linspace(-3, 3, 100)
        loss_values = w_values ** 2  # Loss = w^2

        axes[0].plot(
                w_values, loss_values, 'b-', linewidth=2,
                label='$\\mathcal{L}(w) = w^2$ (fonction de perte)'
                )

        # --- ETAPE 2 : Initialiser les hyperparametres ---
        learning_rate = 0.3  # eta : pas d'apprentissage
        n_iterations = 10  # nombre d'iterations
        w_initial = 2.5  # point de depart (choisi arbitrairement)

        # --- ETAPE 3 : Executer la descente de gradient ---
        weights_history = [w_initial]

        for step in range(n_iterations):
            w_current = weights_history[-1]

            # Calcul du gradient : dL/dw = 2w (derivee de w^2)
            gradient = 2 * w_current

            # Mise a jour : w_new = w_old - lr * gradient
            # Pseudo Stochastic Gradient Descent (SGD),
            w_new = w_current - learning_rate * gradient
            weights_history.append(w_new)

        # Calculer la loss pour chaque poids visite
        loss_history = [w ** 2 for w in weights_history]

        # Afficher la trajectoire
        axes[0].plot(
                weights_history, loss_history, 'ro-', markersize=8,
                label=f'Trajectoire GD ($\\eta={learning_rate}$)'
                )

        for i in range(len(weights_history) - 1):
            axes[0].annotate(
                    '', xy=(weights_history[i + 1], loss_history[i + 1]),
                    xytext=(weights_history[i], loss_history[i]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
                    )

        axes[0].set_xlabel('Poids $w$')
        axes[0].set_ylabel('Perte $\\mathcal{L}(w)$')
        axes[0].set_title('Descente de gradient 1D : minimiser la perte')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Fonction de perte 2D : L(w1, w2) = w1^2 + 2*w2^2
        w1 = np.linspace(-3, 3, 100)
        w2 = np.linspace(-3, 3, 100)
        W1, W2 = np.meshgrid(w1, w2)
        L_2d = W1 ** 2 + 2 * W2 ** 2

        # W1 position X, W2 position Y
        axes[1].contour(W1, W2, L_2d, levels=20, cmap='viridis')
        axes[1].set_xlabel('Poids $w_1$')
        axes[1].set_ylabel('Poids $w_2$')

        # Trajectoire 2D
        lr = 0.2
        pos = [np.array([2.5, 2.0])]  # poids initiaux
        for _ in range(15):
            grad = np.array([2 * pos[-1][0], 4 * pos[-1][1]])  # [dL/dw1, dL/dw2]
            pos.append(pos[-1] - lr * grad)

        pos = np.array(pos)
        axes[1].plot(pos[:, 0], pos[:, 1], 'ro-', markersize=6, label=f'GD ($\\eta={lr}$)')
        axes[1].scatter([0], [0], c='green', s=100, marker='*', label='Minimum global')
        axes[1].set_title('Descente de gradient 2D : $\\mathcal{L}(w_1, w_2) = w_1^2 + 2w_2^2$')
        axes[1].legend()

        plt.tight_layout()
        return fig

    mo.md("### Visualisation de la descente de gradient")
    return (plot_gradient_descent,)


@app.cell
def _(plot_gradient_descent):
    plot_gradient_descent()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 🕹️ Labo Interactif : Jouez avec le Learning Rate !

    Essayez de modifier le **taux d'apprentissage** ($\eta$) ci-dessous.
    - **$\eta < 0.1$** : Descente lente (prudent mais long).
    - **$\eta \approx 0.3$** : Convergence rapide.
    - **$\eta > 0.9$** : Oscillations.
    - **$\eta > 1.0$** : Explosion (Divergence) !
    """)
    return


@app.cell
def _(mo):
    # Les Widgets
    lr_slider = mo.ui.slider(start=0.01, stop=1.2, step=0.01, value=0.1, label="Learning Rate (η)")
    step_slider = mo.ui.slider(start=1, stop=50, step=1, value=10, label="Nombre de pas")

    # On regroupe pour l'affichage
    mo.hstack([lr_slider, step_slider], justify="center")
    return lr_slider, step_slider


@app.cell
def _(lr_slider, mo, np, plt, step_slider):
    # La fonction réactive
    def interact_gradient_descent(lr, steps):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Fonction L(w) = w^2
        w_range = np.linspace(-3, 3, 200)
        loss = w_range ** 2
        ax.plot(w_range, loss, 'b-', alpha=0.3, label='Perte $\mathcal{L}(w) = w^2$')

        # Simulation
        w = 2.5  # Point de départ fixe pour bien comparer
        path = [w]
        for _ in range(steps):
            grad = 2 * w
            w = w - lr * grad
            path.append(w)
            # Stop si ça explose trop pour le graphe
            if abs(w) > 5:
                break

        path = np.array(path)
        loss_path = path ** 2

        # Dessin
        ax.plot(path, loss_path, 'ro-', label=f'Trajectoire ($\eta={lr}$)')

        # Flèches
        for i in range(len(path) - 1):
            if abs(path[i]) < 4 and abs(path[i + 1]) < 4:  # Garder propre
                ax.annotate(
                        '', xy=(path[i + 1], loss_path[i + 1]), xytext=(path[i], loss_path[i]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
                        )

        ax.set_title(f"Descente de Gradient : {len(path) - 1} itérations")
        ax.set_ylim(-0.5, 10)
        ax.set_xlim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig

    # Appel réactif
    mo.vstack(
            [
                mo.md(f"**Taux d'apprentissage actuel : {lr_slider.value}**"),
                interact_gradient_descent(lr_slider.value, step_slider.value)
                ]
            )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. 🧠 Backpropagation : La Règle de la Chaîne

    ### Le problème fondamental

    Comment savoir quel poids modifier (et de combien) dans un réseau de neurones géant ?

    **Situation :** Imaginez un réseau avec 1 million de poids. La prédiction est fausse.
    - Qui est responsable ?
    - Le poids de la couche 1 ? De la couche 50 ? Tous un peu ?

    C'est le problème d'**attribution de crédit** (ou de blâme) : distribuer la responsabilité de l'erreur à chaque paramètre du réseau.

    ---

    ### La solution : La Règle de la Chaîne (*Chain Rule*)

    #### 1. Version mathématique simple

    Si on a une composition de fonctions $y = f(g(x))$, alors :

    $$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

    **Exemple concret :** Soit $y = (2x + 1)^3$

    On pose $g(x) = 2x + 1$ et $f(g) = g^3$, donc $y = f(g(x))$.

    | Étape | Calcul | Résultat |
    | :---: | :--- | :--- |
    | 1 | $\frac{dg}{dx} = \frac{d(2x+1)}{dx}$ | $2$ |
    | 2 | $\frac{dy}{dg} = \frac{d(g^3)}{dg}$ | $3g^2$ |
    | 3 | $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$ | $3g^2 \cdot 2 = 6(2x+1)^2$ |

    ---

    #### 2. Extension à plusieurs étapes

    Pour une chaîne plus longue $y = f(g(h(x)))$ :

    $$\frac{dy}{dx} = \frac{dy}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

    C'est comme une **chaîne de transmission** : chaque maillon multiplie le signal.

    ---

    #### 3. L'intuition : Le système de communication

    Pensez à un **jeu du téléphone** dans une entreprise :

    | Analogie | Mathématiques | Signification |
    | :--- | :--- | :--- |
    | Le PDG annonce un problème | $\frac{\partial L}{\partial L} = 1$ | Signal d'erreur initial |
    | Chaque manager relaie le message | $\times$ dérivée locale | Amplification ou atténuation |
    | L'employé reçoit sa part de responsabilité | $\frac{\partial L}{\partial w}$ | Gradient final du poids |

    ---

    #### 4. La formule fondamentale

    $$
    \boxed{\text{Gradient d'un noeud} = \underbrace{\text{Gradient reçu}}_{\text{"Urgence" venant de l'aval}} \times \underbrace{\text{Dérivée locale}}_{\text{"Influence" calculée ici}}}
    $$

    ---

    #### 5. Tableau récapitulatif des termes

    | Terme | Nom | Rôle | Exemple |
    | :--- | :--- | :--- | :--- |
    | $\frac{\partial \mathcal{L}}{\partial \text{sortie}}$ | **Gradient Externe** | *"L'Urgence"* : À quel point la perte veut que ma sortie change | Si = 10, la sortie doit beaucoup changer |
    | $\frac{\partial \text{sortie}}{\partial \text{entrée}}$ | **Dérivée Locale** | *"L'Influence"* : Si je bouge mon entrée, est-ce que ça impacte ma sortie ? | Si = 0, l'entrée n'a aucun effet |
    | $\frac{\partial \mathcal{L}}{\partial \text{entrée}}$ | **Gradient Propagé** | Le gradient à transmettre aux noeuds précédents | Produit des deux termes ci-dessus |

    ---

    #### 6. Cas particuliers importants

    | Situation | Dérivée locale | Conséquence |
    | :--- | :---: | :--- |
    | ReLU avec entrée < 0 | $0$ | Gradient bloqué ("neurone mort") |
    | Sigmoid saturée (entrée très grande) | $\approx 0$ | Gradient qui s'évanouit |
    | Connexion résiduelle ($y = x + f(x)$) | $1 + f'(x)$ | Gradient préservé (au moins 1) |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 🔍 Backpropagation en Action

    Prenons un exemple concret : $L = (w \cdot x - y)^2$

    ---

    **1. Forward pass** (On calcule le résultat étape par étape) :

    | Étape | Opération | Description |
    | :---: | :--- | :--- |
    | 1 | $z = w \cdot x$ | Multiplication du poids par l'entrée |
    | 2 | $a = z - y$ | Différence entre la prédiction et la cible |
    | 3 | $L = a^2$ | Erreur quadratique (toujours positive) |

    *Exemple numérique :* Si $w=2$, $x=3$, $y=5$ :
    - $z = 2 \times 3 = 6$
    - $a = 6 - 5 = 1$
    - $L = 1^2 = 1$

    ---

    **2. Backward pass** (On remonte le signal d'erreur) :

    On part de la fin et on applique la règle : **grad = urgence × influence**

    ---

    * **Étape 0 (Initialisation) :**
        * On démarre avec $\frac{\partial L}{\partial L} = 1$ (la perte s'influence elle-même à 100%)
        * C'est le "signal d'erreur" initial qu'on va propager vers l'arrière

    ---

    * **Étape 1 (Noeud $L = a^2$) → On cherche $\frac{\partial L}{\partial a}$ :**
        * **Question :** Comment $a$ influence-t-il $L$ ?
        * **Fonction locale :** $L = a^2$
        * **Dérivée locale :** $\frac{\partial L}{\partial a} = 2a$ (règle de puissance)
        * **Calcul :** Urgence reçue × Influence locale = $1 \times 2a$
        * **Résultat :** $\boxed{\frac{\partial L}{\partial a} = 2a}$
        * *Exemple :* Si $a=1$, alors $\frac{\partial L}{\partial a} = 2$

    ---

    * **Étape 2 (Noeud $a = z - y$) → On cherche $\frac{\partial L}{\partial z}$ :**
        * **Question :** Comment $z$ influence-t-il $a$ ?
        * **Fonction locale :** $a = z - y$
        * **Dérivée locale :** $\frac{\partial a}{\partial z} = 1$ (coefficient devant $z$)
        * **Calcul :** Urgence reçue (de $a$) × Influence locale = $2a \times 1$
        * **Résultat :** $\boxed{\frac{\partial L}{\partial z} = 2a}$
        * *Exemple :* Si $a=1$, alors $\frac{\partial L}{\partial z} = 2$

    ---

    * **Étape 2bis (Noeud $a = z - y$) → On cherche aussi $\frac{\partial L}{\partial y}$ :**
        * **Question :** Comment $y$ influence-t-il $a$ ?
        * **Fonction locale :** $a = z - y$
        * **Dérivée locale :** $\frac{\partial a}{\partial y} = -1$ (coefficient devant $y$)
        * **Calcul :** Urgence reçue (de $a$) × Influence locale = $2a \times (-1)$
        * **Résultat :** $\boxed{\frac{\partial L}{\partial y} = -2a}$
        * *Interprétation :* Le signe négatif signifie que si on augmente $y$, la perte diminue (et vice-versa)

    ---

    * **Étape 3 (Noeud $z = w \cdot x$) → On cherche $\frac{\partial L}{\partial w}$ :**
        * **Question :** Comment $w$ influence-t-il $z$ ?
        * **Fonction locale :** $z = w \cdot x$
        * **Dérivée locale :** $\frac{\partial z}{\partial w} = x$ (on dérive par rapport à $w$, donc $x$ est une constante)
        * **Calcul :** Urgence reçue (de $z$) × Influence locale = $2a \times x$
        * **Résultat :** $\boxed{\frac{\partial L}{\partial w} = 2ax = 2x(wx - y)}$
        * *Exemple :* Si $a=1$ et $x=3$, alors $\frac{\partial L}{\partial w} = 6$
        * *Interprétation :* Plus $x$ est grand, plus $w$ a d'influence sur la perte !

    ---

    * **Étape 3bis (Noeud $z = w \cdot x$) → On cherche aussi $\frac{\partial L}{\partial x}$ :**
        * **Question :** Comment $x$ influence-t-il $z$ ?
        * **Fonction locale :** $z = w \cdot x$
        * **Dérivée locale :** $\frac{\partial z}{\partial x} = w$ (on dérive par rapport à $x$, donc $w$ est une constante)
        * **Calcul :** Urgence reçue (de $z$) × Influence locale = $2a \times w$
        * **Résultat :** $\boxed{\frac{\partial L}{\partial x} = 2aw = 2w(wx - y)}$

    ---

    **3. Résumé des gradients :**

    | Variable | Gradient | Rôle |
    | :---: | :--- | :--- |
    | $w$ | $\frac{\partial L}{\partial w} = 2x(wx-y)$ | Utilisé pour mettre à jour le poids |
    | $x$ | $\frac{\partial L}{\partial x} = 2w(wx-y)$ | Propagé vers les couches précédentes |
    | $y$ | $\frac{\partial L}{\partial y} = -2(wx-y)$ | La cible (pas de mise à jour) |

    **Mise à jour du poids :** $w_{\text{nouveau}} = w - \eta \cdot \frac{\partial L}{\partial w}$

    où $\eta$ est le *learning rate* (taux d'apprentissage).
    """)
    return


@app.cell
def _(mo, plt):
    def plot_computational_graph_improved():
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')

        # Définition des positions
        nodes = {
            'x':  (1, 3.5), 'w': (1, 1.5),  # Entrées
            '*':  (3, 2.5),  # Opération z
            '-':  (5, 2.5),  # Opération a (avec y implicite pour simplifier)
            '^2': (7, 2.5),  # Opération L
            'L':  (9, 2.5)  # Sortie
            }

        # --- DESSIN DU FORWARD (Bleu) ---
        arrows_fwd = [('x', '*'), ('w', '*'), ('*', '-'), ('-', '^2'), ('^2', 'L')]

        for start, end in arrows_fwd:
            x1, y1 = nodes[start]
            x2, y2 = nodes[end]
            # Décalage léger vers le haut pour les flèches bleues
            ax.annotate(
                    '', xy=(x2 - 0.3, y2 + 0.1), xytext=(x1 + 0.3, y1 + 0.1),
                    arrowprops=dict(arrowstyle='->', color='#3498db', lw=2)
                    )

        # --- DESSIN DU BACKWARD (Rouge) ---
        # On inverse les flèches
        for start, end in arrows_fwd:
            x1, y1 = nodes[end]  # Inversé
            x2, y2 = nodes[start]  # Inversé

            # Courbure (connectionstyle) pour séparer visuellement le retour
            ax.annotate(
                    '', xy=(x2 + 0.3, y2 - 0.1), xytext=(x1 - 0.3, y1 - 0.1),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2, ls='--')
                    )

        # --- NOEUDS ---
        for name, (x, y) in nodes.items():
            color = 'lightgreen' if name == 'L' else 'white'
            circle = plt.Circle((x, y), 0.4, facecolor=color, edgecolor='black', linewidth=1.5, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold', zorder=11)

        # --- ANNOTATIONS PEDAGOGIQUES ---

        # Sur la flèche de retour de w
        ax.text(
                2, 1.2, r"$\frac{\partial L}{\partial w} = \text{Grad}_z \times x$",
                color='#c0392b', fontsize=11, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )

        # Sur la flèche de retour de x
        ax.text(
                2, 3.8, r"$\frac{\partial L}{\partial x} = \text{Grad}_z \times w$",
                color='#c0392b', fontsize=11, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )

        # Légende
        ax.text(
                5, 4.5, "Forward Pass (Calcul de la valeur)", color='#3498db', fontsize=14, ha='center',
                fontweight='bold'
                )
        ax.text(
                5, 0.5, "Backward Pass (Transport de l'urgence)", color='#e74c3c', fontsize=14, ha='center',
                fontweight='bold'
                )

        ax.set_title('Le "Jeu du Téléphone" des Gradients', fontsize=16)
        plt.tight_layout()
        return fig

    mo.md("### 🎨 Visualisation du flux de gradient")
    return (plot_computational_graph_improved,)


@app.cell
def _(plot_computational_graph_improved):
    plot_computational_graph_improved()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ⚠️ Note cruciale : L'Accumulation (`+=`)

    Que se passe-t-il si une variable est utilisée **plusieurs fois** ?

    Imaginez une rivière avec deux affluents qui se rejoignent. Le débit final est la **somme** des deux.

    * Mathématiquement : C'est la **Règle de la Chaîne Multivariée**.
    * Code : C'est pour ça qu'on utilise `self.grad += ...` et non `=`.

    Si $x$ influence la perte via deux chemins différents, on doit additionner l'urgence venant des deux chemins pour connaître l'urgence totale sur $x$.
    """)
    return


@app.cell
def _(mo, np, plt):
    def plot_vanishing_gradient():
        # Simulation : On remonte le temps sur 50 étapes
        steps = np.arange(50)

        # Cas 1 : Dérivée locale < 1 (ex: fonction Tanh ou Sigmoid saturée)
        grad_vanish = 1.0 * (0.8 ** steps)

        # Cas 2 : Dérivée locale > 1 (Exploding Gradient - l'inverse, tout explose)
        grad_explode = 1.0 * (1.1 ** steps)

        # Cas 3 : Idéal (LSTM / Residual connection)
        grad_stable = np.ones(50)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, grad_vanish, label='Vanishing (facteur 0.8)', color='red', linewidth=3)
        ax.plot(steps, grad_explode, label='Exploding (facteur 1.1)', color='orange', linestyle='--')
        ax.plot(steps, grad_stable, label='Idéal (facteur 1.0)', color='green', linestyle=':')

        ax.set_title("Pourquoi les RNN oublient le début du contexte ?")
        ax.set_xlabel("Nombre d'étapes en arrière (Backpropagation through Time)")
        ax.set_ylabel("Force du signal d'urgence (Gradient)")
        ax.set_ylim(0, 2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotation
        ax.annotate(
                'Le signal devient nul ici !\nLe début du réseau ne change pas.',
                xy=(20, 0.01), xytext=(25, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05)
                )

        return fig

    mo.md("### 📉 Visualisation : La mort du gradient")
    return (plot_vanishing_gradient,)


@app.cell
def _(plot_vanishing_gradient):
    plot_vanishing_gradient()
    return


if __name__ == "__main__":
    app.run()
