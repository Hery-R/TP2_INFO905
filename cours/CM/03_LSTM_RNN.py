import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
            # Reseaux de Neurones Recurrents (RNN) et LSTM
        
            Les **RNN** et **LSTM** sont des architectures concues pour traiter des **donnees sequentielles** : texte, series temporelles, audio, video...
        
            ## Pourquoi avons-nous besoin de RNN ?
        
            Les reseaux feedforward ont deux limitations majeures :
        
            1. **Entree de taille fixe** : impossible de traiter des sequences de longueur variable
            2. **Pas de memoire** : chaque entree est traitee independamment
        
            **Exemple** : Pour completer *"Les nuages sont dans le ___"*, on a besoin du **contexte** !
        
            ## L'idee cle
        
            Un RNN maintient un **etat cache** (hidden state) $h_t$ qui accumule l'information :
            - A chaque pas $t$ : le reseau recoit l'entree $x_t$ ET l'etat precedent $h_{t-1}$
            - L'etat cache agit comme une **memoire** des entrees passees
            """
        )
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pathlib
    torch.cuda.is_available()
    return F, go, make_subplots, nn, np, pathlib, torch


@app.cell
def _(pathlib):
    abs_path_asset = pathlib.Path(__file__).parent.resolve() / "asset"
    return (abs_path_asset,)


@app.cell
def _(mo):
    # 1. Create a slider to choose the image index
    step = mo.ui.slider(start=1, stop=5, step=1, label="Select RNN Step")
    step
    return (step,)


@app.cell
def _(abs_path_asset, mo, step):
    # 3. Use the slider's value to dynamically pick the file
    image_filename = f"rnn0{step.value}.gif"

    mo.image(
            src=abs_path_asset / image_filename,
            alt=f"RNN Step {step.value}",
            rounded=True,
            caption=f"Displaying {image_filename}"
            )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ---
        
            # 1. RNN Simple : Theorie et Implementation
        
            ## Equations fondamentales
        
            A chaque pas de temps $t$ :
        
            $$\boxed{h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)}$$
        
            $$y_t = W_{hy} \cdot h_t + b_y$$
        
            | Symbole | Dimension | Description |
            |---------|-----------|-------------|
            | $x_t$ | `(input_size,)` | Entree au temps $t$ |
            | $h_t$ | `(hidden_size,)` | Etat cache au temps $t$ (la "memoire") |
            | $W_{xh}$ | `(hidden_size, input_size)` | Poids entree → cache |
            | $W_{hh}$ | `(hidden_size, hidden_size)` | Poids cache → cache (recurrence) |
            | $b_h$ | `(hidden_size,)` | Biais |
            """
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Implementation manuelle vs PyTorch
        
            Voyons comment implementer la formule $h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$ :
            """
        )
    return


@app.cell
def _(nn, torch):
    # === IMPLEMENTATION MANUELLE D'UN PAS RNN ===

    # Dimensions
    input_size = 4  # Taille de x_t
    hidden_size = 3  # Taille de h_t

    # Initialisation des poids (comme PyTorch le ferait)
    torch.manual_seed(42)
    W_xh = torch.randn(hidden_size, input_size)  # (3, 4)
    W_hh = torch.randn(hidden_size, hidden_size)  # (3, 3)
    b_h = torch.zeros(hidden_size)  # (3,)

    # Entree et etat precedent
    x_t = torch.randn(1, input_size)  # (1, 4) - un echantillon
    h_prev = torch.zeros(1, hidden_size)  # (1, 3) - etat initial nul

    # === FORMULE MANUELLE ===
    # h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
    h_manual = torch.tanh(
            x_t @ W_xh.T +  # (1,4) @ (4,3) = (1,3)
            h_prev @ W_hh.T +  # (1,3) @ (3,3) = (1,3)
            b_h  # broadcast (3,) -> (1,3)
            )

    print("=== Implementation manuelle ===")
    print(f"x_t shape: {x_t.shape}")
    print(f"h_prev shape: {h_prev.shape}")
    print(f"h_t shape: {h_manual.shape}")
    print(f"h_t = {h_manual}")

    # === VERIFICATION AVEC nn.RNNCell ===
    rnn_cell = nn.RNNCell(input_size, hidden_size)

    # Copier nos poids dans la cellule PyTorch
    with torch.no_grad():
        rnn_cell.weight_ih.copy_(W_xh)
        rnn_cell.weight_hh.copy_(W_hh)
        rnn_cell.bias_ih.copy_(b_h)
        rnn_cell.bias_hh.zero_()

    h_pytorch = rnn_cell(x_t, h_prev)

    print("\n=== Verification avec nn.RNNCell ===")
    print(f"h_t PyTorch = {h_pytorch}")
    print(f"Difference: {(h_manual - h_pytorch).abs().max().item():.2e}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Traitement d'une sequence complete
        
            Pour une sequence $[x_1, x_2, ..., x_T]$, on applique la meme formule a chaque pas :
            """
        )
    return


@app.cell
def _(nn, torch):
    # === TRAITEMENT D'UNE SEQUENCE ===

    seq_len = 5
    input_dim = 4
    hidden_dim = 3

    torch.manual_seed(123)

    # Sequence d'entree : 5 vecteurs de dimension 4
    sequence = torch.randn(seq_len, input_dim)
    print(f"Sequence shape: {sequence.shape}  # (seq_len, input_size)")

    # RNN Cell
    rnn = nn.RNNCell(input_dim, hidden_dim)

    # Etat initial
    h = torch.zeros(1, hidden_dim)
    print(f"h_0 = {h.squeeze().tolist()}")

    # Parcourir la sequence
    hidden_states = [h.squeeze().clone()]

    for t in range(seq_len):
        x = sequence[t].unsqueeze(0)  # (1, input_dim)
        h = rnn(x, h)  # Formule: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
        hidden_states.append(h.squeeze().clone())
        print(f"t={t + 1}: x_t = {x.squeeze()[:2].tolist()}... -> h_t = {h.squeeze().tolist()}")

    print(f"\nL'etat final h_{seq_len} encode l'information de TOUTE la sequence!")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Visualisation : Evolution de l'etat cache
            """
        )
    return


@app.cell
def _(mo):
    sequence_input = mo.ui.text(
            value="le chat mange la souris",
            label="Sequence d'entree (mots separes par espaces)"
            )
    hidden_size_demo = mo.ui.slider(
            start=2, stop=16, step=2, value=4,
            label="Taille de l'etat cache"
            )
    mo.hstack([sequence_input, hidden_size_demo])
    return hidden_size_demo, sequence_input


@app.cell
def _(go, hidden_size_demo, make_subplots, mo, nn, np, sequence_input, torch):
    words = sequence_input.value.split()
    vocab = {word: i for i, word in enumerate(set(words))}
    vocab_size = len(vocab)

    embedding_dim = 8
    hidden_dim_viz = hidden_size_demo.value

    torch.manual_seed(42)
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    rnn_cell_viz = nn.RNNCell(embedding_dim, hidden_dim_viz)

    indices = torch.tensor([vocab[w] for w in words])
    embeddings = embedding_layer(indices)

    hidden_states_viz = []
    h_viz = torch.zeros(1, hidden_dim_viz)
    hidden_states_viz.append(h_viz.detach().numpy().flatten())

    for i in range(len(words)):
        x_viz = embeddings[i].unsqueeze(0)
        h_viz = rnn_cell_viz(x_viz, h_viz)
        hidden_states_viz.append(h_viz.detach().numpy().flatten())

    hidden_states_viz = np.array(hidden_states_viz)

    fig_rnn = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Evolution de chaque dimension h[i]", "Heatmap des etats"),
            column_widths=[0.5, 0.5]
            )

    for dim in range(hidden_dim_viz):
        fig_rnn.add_trace(
                go.Scatter(
                        x=["init"] + words,
                        y=hidden_states_viz[:, dim],
                        mode='lines+markers',
                        name=f'h[{dim}]'
                        ),
                row=1, col=1
                )

    fig_rnn.add_trace(
            go.Heatmap(
                    z=hidden_states_viz.T,
                    x=["init"] + words,
                    y=[f"h[{i}]" for i in range(hidden_dim_viz)],
                    colorscale='RdBu',
                    zmid=0
                    ),
            row=1, col=2
            )

    fig_rnn.update_layout(height=400, title_text="L'etat cache evolue a chaque mot")

    mo.vstack(
            [
                mo.md(f"**Vocabulaire**: {vocab} | **Embedding**: {embedding_dim} | **Hidden**: {hidden_dim_viz}"),
                fig_rnn
                ]
            )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ---
        
            # 2. Le Probleme du Vanishing Gradient
        
            ## Pourquoi les RNN simples echouent sur les longues sequences ?
        
            Lors de la backpropagation, le gradient est multiplie par $W_{hh}$ a **chaque pas de temps** :
        
            $$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} \approx (W_{hh})^{T-1}$$
        
            **Consequence** :
            - Si $\|W_{hh}\| < 1$ : gradient $\to 0$ exponentiellement (**vanishing**)
            - Si $\|W_{hh}\| > 1$ : gradient $\to \infty$ exponentiellement (**exploding**)
        
            Le réseau devient "amnésique" des premiers instants.
        
            > Un Vanishing Gradient (gradient proche de zéro) signifie que le réseau met à jour ses poids uniquement en fonction des erreurs récentes. Concrètement, le début de la séquence n'a plus aucun impact sur l'apprentissage : le réseau devient amnésique aux événements passés.
            """
        )
    return


@app.cell
def _(abs_path_asset, mo):
    mo.image(
            src=abs_path_asset / "rnn_gradiant.png",
            rounded=True,
            )
    return


@app.cell
def _(go, mo, np):
    steps = np.arange(0, 50)

    gradient_0_9 = 0.9 ** steps
    gradient_1_0 = 1.0 ** steps
    gradient_1_1 = 1.1 ** steps

    fig_gradient = go.Figure()
    fig_gradient.add_trace(
        go.Scatter(
            x=steps, y=gradient_0_9, mode='lines',
            name='||W|| = 0.9 (vanishing)', line=dict(color='blue', width=2)
            )
        )
    fig_gradient.add_trace(
        go.Scatter(
            x=steps, y=gradient_1_0, mode='lines',
            name='||W|| = 1.0 (stable)', line=dict(color='green', width=2, dash='dash')
            )
        )
    fig_gradient.add_trace(
        go.Scatter(
            x=steps, y=np.clip(gradient_1_1, 0, 100), mode='lines',
            name='||W|| = 1.1 (exploding)', line=dict(color='red', width=2)
            )
        )

    fig_gradient.update_layout(
            title="Evolution du gradient : (||W||)^t",
            xaxis_title="Nombre de pas t",
            yaxis_title="Magnitude (echelle log)",
            yaxis_type="log",
            height=350
            )

    mo.vstack(
            [
                fig_gradient,
                mo.md("> Apres 50 pas avec $\|W\| = 0.9$, le gradient vaut $0.9^{50} \\approx 0.005$ soit **0.5%** !")
                ]
            )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Demonstration : le gradient disparait
        
            Visualisons le gradient qui remonte dans le temps :
            """
        )
    return


@app.cell
def _(nn, torch):
    def get_gradient_norms(model_type, seq_length):
        torch.manual_seed(42)

        # Dimensions fixes pour la démo
        batch_size = 1
        input_size = 10
        hidden_size = 20

        # 1. Instanciation du modèle
        # batch_first=True est crucial : les tenseurs seront [Batch, Seq, Features]
        if model_type == "RNN":
            model = nn.RNN(input_size, hidden_size, batch_first=True)
        else:
            model = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 2. Création de l'input
        # Shape : [Batch=1, Seq_len=T, Input_size=10]
        x = torch.randn(batch_size, seq_length, input_size, requires_grad=True)

        # 3. Forward Pass
        # output contient les états cachés pour chaque pas de temps
        # Shape output : [Batch=1, Seq_len=T, Hidden_size=20]
        output, _ = model(x)

        # 4. Calcul de la Loss
        # On isole le dernier token de la séquence (le temps T)
        # Shape output[:, -1, :] : [Batch=1, Hidden_size=20]
        last_step_output = output[:, -1, :]

        # La loss est un scalaire
        # Shape loss : [] (Scalaire)
        loss = last_step_output.sum()

        # 5. Backward Pass
        model.zero_grad()
        loss.backward()

        # 6. Inspection du Gradient
        # Le gradient de x a exactement la même forme que x
        # Shape x.grad : [Batch=1, Seq_len=T, Input_size=10]

        # On extrait la norme du vecteur gradient pour chaque pas de temps t
        # x.grad[0, t] est un vecteur de taille [10]
        grads = [x.grad[0, t].norm().item() for t in range(seq_length)]

        return grads

    return (get_gradient_norms,)


@app.cell
def _(mo):
    # Sliders pour contrôler l'expérience
    slider_seq = mo.ui.slider(start=10, stop=100, step=10, value=50, label="Longueur de Séquence")
    slider_seq
    return (slider_seq,)


@app.cell
def _(get_gradient_norms, go, mo, slider_seq):
    def _():
        # 1. Récupération des données (Préfixe viz_)
        viz_seq_len = slider_seq.value

        # On recalcule ou on récupère les gradients
        viz_grads_rnn = get_gradient_norms("RNN", viz_seq_len)
        viz_grads_lstm = get_gradient_norms("LSTM", viz_seq_len)
        viz_steps = list(range(viz_seq_len))

        # 2. Construction de la figure Plotly
        viz_fig = go.Figure()

        # Trace RNN (Rouge - Le "mauvais élève")
        viz_fig.add_trace(
                go.Scatter(
                        x=viz_steps,
                        y=viz_grads_rnn,
                        mode='lines+markers',
                        name='RNN Classique',
                        line=dict(color='#ef553b', width=2),
                        marker=dict(size=6)
                        )
                )

        # Trace LSTM (Vert - Le "bon élève")
        viz_fig.add_trace(
                go.Scatter(
                        x=viz_steps,
                        y=viz_grads_lstm,
                        mode='lines+markers',
                        name='LSTM',
                        line=dict(color='#00cc96', width=2, dash='dash'),
                        marker=dict(size=6, symbol='x')
                        )
                )

        # 3. Mise en page (Layout pédagogique)
        viz_fig.update_layout(
                title=f"<b>Vanishing Gradient</b> (Séquence T={viz_seq_len})",
                xaxis_title="Pas de temps (t)",
                yaxis_title="Norme du Gradient (Log Scale)",
                yaxis_type="log",  # Échelle log indispensable
                hovermode="x unified",
                template="plotly_white",
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

        # Ajout d'une zone rouge "Zone Morte"
        viz_fig.add_hrect(
                y0=0, y1=1e-10,
                fillcolor="red", opacity=0.1, layer="below", line_width=0,
                annotation_text="Zone de Vanishing Gradient (Information Perdue)",
                annotation_position="bottom right"
                )

        # Annotations temporelles pour orienter l'étudiant
        viz_fig.add_annotation(x=0, y=viz_grads_lstm[0], text="⬅ Début (Passé lointain)", showarrow=True, arrowhead=1)
        viz_fig.add_annotation(
            x=viz_seq_len - 1, y=viz_grads_lstm[-1], text="Fin (Présent) ➡", showarrow=True, arrowhead=1
            )

        # 4. Calcul des statistiques
        viz_ratio_rnn = viz_grads_rnn[0] / viz_grads_rnn[-1] if viz_grads_rnn[-1] != 0 else 0
        viz_ratio_lstm = viz_grads_lstm[0] / viz_grads_lstm[-1] if viz_grads_lstm[-1] != 0 else 0

        # 5. Rendu final
        return mo.vstack(
                [
                    mo.md("### 📉 Analyse Dynamique"),
                    viz_fig,
                    mo.callout(
                            mo.md(
                                f"""
                **Diagnostic :**
            
                * **RNN (Ligne Rouge)** : Ratio Début/Fin = **{viz_ratio_rnn:.2e}**.
                    * Le gradient chute drastiquement. Le réseau ne peut pas apprendre les liens entre le début ($t=0$) et la fin ($t={viz_seq_len}$).
                * **LSTM (Ligne Verte)** : Ratio Début/Fin = **{viz_ratio_lstm:.2f}**.
                    * Le gradient reste stable. L'information circule librement du passé vers le présent.
                """
                                ),
                            kind="danger" if viz_ratio_rnn < 1e-3 else "success"
                            )
                    ]
                )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ---
        
            # 3. LSTM : Long Short-Term Memory
        
            **Solution** (Hochreiter & Schmidhuber, 1997) : Ajouter des **portes** (gates) pour controler le flux d'information.
        
            ## Architecture LSTM
        
            Le LSTM introduit un **etat de cellule** $C_t$ separe de l'etat cache $h_t$ :
        
            | Composant | Role |
            |-----------|------|
            | $C_t$ (cell state) | Memoire a long terme - "autoroute" pour le gradient |
            | $h_t$ (hidden state) | Sortie a court terme |
            | $f_t$ (forget gate) | Controle ce qu'on **oublie** de $C_{t-1}$ |
            | $i_t$ (input gate) | Controle ce qu'on **ajoute** a $C_t$ |
            | $o_t$ (output gate) | Controle ce qu'on **produit** dans $h_t$ |
            """
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Equations du LSTM avec code PyTorch
        
            ### Etape 1 : Forget Gate - "Que faut-il oublier ?"
        
            $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
        
            Sortie entre 0 (tout oublier) et 1 (tout garder).
            """
        )
    return


@app.cell
def _(mo, torch):
    # === FORGET GATE (Porte d'oubli) ===
    torch.manual_seed(44)

    # 1. Configuration des Dimensions (Préfixe lstm_)
    lstm_input_sz = 4
    lstm_hidden_sz = 3
    lstm_batch_sz = 1

    torch.manual_seed(42)

    # 2. Création des Tenseurs Inputs
    lstm_x_t = torch.randn(lstm_batch_sz, lstm_input_sz)
    lstm_h_prev = torch.randn(lstm_batch_sz, lstm_hidden_sz)

    # 3. Paramètres de la Forget Gate
    lstm_W_f = torch.randn(lstm_hidden_sz, lstm_input_sz + lstm_hidden_sz)
    lstm_b_f = torch.zeros(lstm_hidden_sz)

    # 4. Opération 1 : Concaténation
    lstm_combined = torch.cat([lstm_h_prev, lstm_x_t], dim=1)

    # 5. Opération 2 : Calcul Linéaire + Activation
    lstm_linear_out = lstm_combined @ lstm_W_f.T + lstm_b_f
    lstm_f_t = torch.sigmoid(lstm_linear_out)

    # === AFFICHAGE ===

    # Visualisation des dimensions
    lstm_dims_info = mo.md(
        f"""
    **Suivi des Dimensions (Shapes) :**

    1.  **Inputs séparés** :
        * `lstm_x_t` : `{tuple(lstm_x_t.shape)}`
        * `lstm_h_prev` : `{tuple(lstm_h_prev.shape)}`
    
    2.  **Concaténation** :
        * `lstm_combined` : `{tuple(lstm_combined.shape)}`
    
    3.  **Vecteur Forget Gate (`lstm_f_t`)** :
        * Résultat : `{tuple(lstm_f_t.shape)}`
    """
        )

    # Interprétation des valeurs
    lstm_vals = lstm_f_t[0].detach().numpy()
    lstm_interpretation = []
    for lstm_i, val in enumerate(lstm_vals):
        action1 = "🟢 GARDER" if val > 0.5 else "🔴 OUBLIER"
        lstm_interpretation.append(
            f"* **Neurone {lstm_i}** (val={val:.2f}) : {action1} {val * 100:.0f}% de la mémoire."
            )

    mo.vstack(
            [
                mo.md("### 🧠 Anatomie d'une Forget Gate"),
                lstm_dims_info,
                mo.md("---"),
                mo.md("**Décision de la porte (f_t) :**"),
                mo.md("\n".join(lstm_interpretation))
                ]
            )
    return lstm_combined, lstm_f_t, lstm_hidden_sz, lstm_input_sz


@app.cell
def _(mo):
    mo.md(
        r"""
            ### Etape 2 : Input Gate - "Quelles nouvelles infos stocker ?"
        
            $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
        
            $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
        
            - $i_t$ : quelles dimensions mettre a jour (0 ou 1)
            - $\tilde{C}_t$ : candidates pour la mise a jour (entre -1 et 1)
            """
        )
    return


@app.cell
def _(lstm_combined, lstm_hidden_sz, lstm_input_sz, mo, torch):
    # === INPUT GATE (Porte d'entrée) ===

    # 1. Poids pour la Porte (Input Gate)
    lstm_W_i = torch.randn(lstm_hidden_sz, lstm_input_sz + lstm_hidden_sz)
    lstm_b_i = torch.zeros(lstm_hidden_sz)

    # 2. Poids pour le Candidat (Cell Candidate)
    lstm_W_C = torch.randn(lstm_hidden_sz, lstm_input_sz + lstm_hidden_sz)
    lstm_b_C = torch.zeros(lstm_hidden_sz)

    # --- CALCULS ---

    # A. La Porte (lstm_i_t) : Sigmoid
    lstm_i_t = torch.sigmoid(lstm_combined @ lstm_W_i.T + lstm_b_i)

    # B. Le Candidat (lstm_C_tilde) : Tanh
    lstm_C_tilde = torch.tanh(lstm_combined @ lstm_W_C.T + lstm_b_C)

    # C. L'Information Finalement Ajoutée
    lstm_added_info = lstm_i_t * lstm_C_tilde

    # --- AFFICHAGE ---

    lstm_lines = []
    for idx1 in range(lstm_hidden_sz):
        gate_val1 = lstm_i_t[0][idx1].item()
        cand_val = lstm_C_tilde[0][idx1].item()
        final_val = lstm_added_info[0][idx1].item()

        status = "🔒 FERMÉ" if gate_val1 < 0.3 else ("🔓 OUVERT" if gate_val1 > 0.7 else "⚠️ MOYEN")

        lstm_lines.append(
                f"* **Neurone {idx1}** : Porte({gate_val1:.2f}) $\\times$ Candidat({cand_val:.2f}) = **{final_val:.2f}** $\\rightarrow$ {status}"
                )

    mo.vstack(
            [
                mo.md("### ✍️ Input Gate : Création et Filtrage"),
                mo.callout(mo.md("\n".join(lstm_lines)), kind="info"),
                mo.md(f"**Shape du résultat ajouté** : `{tuple(lstm_added_info.shape)}`")
                ]
            )
    return lstm_C_tilde, lstm_i_t


@app.cell
def _(mo):
    mo.md(
        r"""
            ### Etape 3 : Mise a jour de l'etat de cellule
        
            $$\boxed{C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t}$$
        
            C'est **LA** formule cle ! Le gradient peut circuler directement via $C_{t-1} \to C_t$.
            """
        )
    return


@app.cell
def _(lstm_C_tilde, lstm_f_t, lstm_hidden_sz, lstm_i_t, mo, torch):
    # === MISE A JOUR DE L'ETAT DE CELLULE ===

    # 1. Simulation de l'état précédent (C_{t-1})
    lstm_C_prev = torch.randn(1, lstm_hidden_sz)

    # 2. Calcul des deux termes
    # Terme A : Vieux souvenirs conservés (Forget * Old)
    lstm_term_forget = lstm_f_t * lstm_C_prev

    # Terme B : Nouveaux souvenirs ajoutés (Input * Candidate)
    lstm_term_input = lstm_i_t * lstm_C_tilde

    # 3. Nouvelle Mémoire (C_t)
    lstm_C_t_1 = lstm_term_forget + lstm_term_input

    # --- AFFICHAGE ---

    lstm_explanation = []
    for lstm_idx2 in range(lstm_hidden_sz):
        old = lstm_C_prev[0][lstm_idx2].item()
        kept = lstm_term_forget[0][lstm_idx2].item()
        added = lstm_term_input[0][lstm_idx2].item()
        new = lstm_C_t_1[0][lstm_idx2].item()

        lstm_explanation.append(
                f"**N{lstm_idx2}** : (Ancien `{old:.2f}` $\\rightarrow$ Gardé `{kept:.2f}`) + (Ajout `{added:.2f}`) = **Nouvel État `{new:.2f}`**"
                )

    mo.vstack(
            [
                mo.md("### 🔄 Mise à jour de la Mémoire ($C_t$)"),
                mo.md(r"$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$"),
                mo.md("\n\n".join(lstm_explanation)),
                mo.callout(
                        mo.md(f"**Résultat final `lstm_C_t_1` shape**: `{tuple(lstm_C_t_1.shape)}`"),
                        kind="success"
                        )
                ]
            )
    return (lstm_C_t_1,)


@app.cell
def _(mo):
    mo.md(
        r"""
            ### Etape 4 : Output Gate - "Que produire en sortie ?"
        
            $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
        
            $$h_t = o_t \odot \tanh(C_t)$$
            """
        )
    return


@app.cell
def _(lstm_C_t_1, lstm_combined, lstm_hidden_sz, lstm_input_sz, mo, torch):
    # === OUTPUT GATE (Porte de Sortie) ===
    # Objectif : Filtrer la mémoire interne (C_t) pour générer l'état caché (h_t).

    # 1. Poids de l'Output Gate
    # Elle regarde le contexte précédent (concaténé) pour décider
    lstm_W_o = torch.randn(lstm_hidden_sz, lstm_input_sz + lstm_hidden_sz)
    lstm_b_o = torch.zeros(lstm_hidden_sz)

    # --- CALCULS ---

    # A. La Porte (lstm_o_t) : "Qu'est-ce qui est pertinent maintenant ?"
    # Sigmoid -> 0 (Cacher) à 1 (Révéler)
    lstm_o_t = torch.sigmoid(lstm_combined @ lstm_W_o.T + lstm_b_o)

    # B. Préparation de la Mémoire
    # On passe la mémoire C_t dans tanh pour la remettre entre -1 et 1
    lstm_memory_tanh = torch.tanh(lstm_C_t_1)

    # C. Calcul final de h_t
    # h_t = o_t * tanh(C_t)
    lstm_h_t = lstm_o_t * lstm_memory_tanh

    # --- AFFICHAGE PÉDAGOGIQUE ---

    # Création d'un tableau explicatif
    output_lines = []
    for idx3 in range(lstm_hidden_sz):
        gate_val3 = lstm_o_t[0][idx3].item()
        mem_val = lstm_memory_tanh[0][idx3].item()
        h_val = lstm_h_t[0][idx3].item()

        # Visualisation
        action3 = "📢 DIRE" if gate_val3 > 0.6 else ("🤫 CHUCHOTER" if gate_val3 > 0.3 else "🤐 TAIRE")

        output_lines.append(
                f"* **Neurone {idx3}** : Mémoire({mem_val:.2f}) filtrée par Porte({gate_val3:.2f}) $\\rightarrow$ **Sortie h_t : {h_val:.2f}** ({action3})"
                )

    mo.vstack(
            [
                mo.md("### 🗣️ Output Gate : Le Porte-Parole"),
                mo.md(r"$$h_t = o_t \odot \tanh(C_t)$$"),

                mo.md(
                    """
                            La **Cell State ($C_t$)** contient *tout* l'historique (compteurs, parenthèses ouvertes, contexte lointain).
                            L'**Output Gate ($o_t$)** décide quelle partie est utile *immédiatement*.
                            """
                    ),

                mo.callout(mo.md("\n".join(output_lines)), kind="info"),

                mo.md("### 📏 Bilan des Dimensions Finales"),
                mo.md(
                    f"""
        * **Porte ($o_t$)** : `{tuple(lstm_o_t.shape)}`
        * **Mémoire normalisée ($\tanh(C_t)$)** : `{tuple(lstm_memory_tanh.shape)}`
        * **État Caché Final ($h_t$)** : `{tuple(lstm_h_t.shape)}`
    
        Ce vecteur **$h_t$** sera :
        1. La sortie de cette couche pour ce pas de temps.
        2. L'entrée $h_{{t-1}}$ pour le prochain pas de temps.
        """
                    )
                ]
            )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Verification avec `nn.LSTMCell`
        
            Comparons notre implementation manuelle avec PyTorch :
            """
        )
    return


@app.cell
def _(nn, torch):
    def _():
        # === LSTM COMPLET : MANUEL vs PYTORCH ===

        inp_size = 4
        hid_size = 3

        torch.manual_seed(100)

        # Creer une LSTMCell PyTorch
        lstm_cell = nn.LSTMCell(inp_size, hid_size)

        # Entrees
        x_test = torch.randn(1, inp_size)
        h_test = torch.randn(1, hid_size)
        c_test = torch.randn(1, hid_size)

        # Forward PyTorch
        h_out_pt, c_out_pt = lstm_cell(x_test, (h_test, c_test))

        # === IMPLEMENTATION MANUELLE ===
        # PyTorch stocke les poids dans l'ordre: [i, f, g, o]
        # weight_ih : (4*hidden, input)
        # weight_hh : (4*hidden, hidden)

        with torch.no_grad():
            W_ih = lstm_cell.weight_ih  # (12, 4)
            W_hh = lstm_cell.weight_hh  # (12, 3)
            b_ih = lstm_cell.bias_ih  # (12,)
            b_hh = lstm_cell.bias_hh  # (12,)

            # Calcul des gates
            gates = x_test @ W_ih.T + h_test @ W_hh.T + b_ih + b_hh  # (1, 12)

            # Separer en 4 parties (i, f, g, o)
            i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)

            i_g = torch.sigmoid(i_g)  # input gate
            f_g = torch.sigmoid(f_g)  # forget gate
            g_g = torch.tanh(g_g)  # cell candidate
            o_g = torch.sigmoid(o_g)  # output gate

            # Mise a jour cellule et hidden
            c_out_manual = f_g * c_test + i_g * g_g
            h_out_manual = o_g * torch.tanh(c_out_manual)

        print("=== Comparaison Manuel vs PyTorch ===")
        print(f"h_t PyTorch: {h_out_pt.squeeze().tolist()}")
        print(f"h_t Manuel:  {h_out_manual.squeeze().tolist()}")
        print(f"C_t PyTorch: {c_out_pt.squeeze().tolist()}")
        print(f"C_t Manuel:  {c_out_manual.squeeze().tolist()}")
        return print(f"\nDifference max: {(h_out_pt - h_out_manual).abs().max().item():.2e}")

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## 🛣️ Pourquoi le LSTM résout-il le Vanishing Gradient ?
        
            Le secret ne réside pas dans la complexité, mais dans une opération arithmétique simple : **l'Addition**.
        
            ### 1. Le Problème du RNN (Le "Téléphone Arabe")
            Dans un RNN classique, l'état caché est mis à jour par multiplication matricielle et fonction d'activation :
        
            $$h_t = \tanh(W_{hh} h_{t-1} + \dots)$$
        
            Lors de la rétropropagation, le gradient est multiplié répétitivement par $W$ et par la dérivée de la tangente hyperbolique (qui est toujours $< 1$).
            > **Résultat :** $0.9 \times 0.9 \times \dots \times 0.9 \approx 0$. Le signal meurt.
        
            ### 2. La Solution du LSTM (L'Autoroute)
            Regardons l'équation de mise à jour de la Cell State ($C_t$) :
        
            $$C_t = \underbrace{f_t \odot C_{t-1}}_{\text{Flux Direct}} + \underbrace{i_t \odot \tilde{C}_t}_{\text{Nouvelle Info}}$$
        
            Observez le premier terme : $C_t$ est relié à $C_{t-1}$ par une addition linéaire. Si nous calculons le gradient (la dérivée partielle) qui remonte le temps :
        
            $$ \frac{\partial C_t}{\partial C_{t-1}} = f_t $$
        
            * Si la porte d'oubli est ouverte (**$f_t \approx 1$**), le gradient est multiplié par 1.
            * Il peut donc traverser 100 pas de temps sans être modifié ni écrasé.
            * C'est ce qu'on appelle la **Constant Error Carousel** ou l'**Autoroute du Gradient**.
            """
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Visualisation interactive des portes LSTM
            """
        )
    return


@app.cell
def _(mo):
    lstm_sequence_cb = mo.ui.text(
            value="le deep learning est fascinant",
            label="Séquence d'entrée (phrase)"
            )
    lstm_hidden_slider_cb = mo.ui.slider(
            start=4, stop=32, step=4, value=8,
            label="Taille de l'État Caché (Neurones)"
            )

    mo.vstack(
            [
                mo.md("### 🎛️ Paramètres"),
                mo.hstack([lstm_sequence_cb, lstm_hidden_slider_cb])
                ]
            )
    return lstm_hidden_slider_cb, lstm_sequence_cb


@app.cell
def _(
        go,
        lstm_hidden_slider_cb,
        lstm_sequence_cb,
        make_subplots,
        mo,
        nn,
        np,
        torch,
        ):
    def _():
        CB_COLORS = {
            "forget": "#D55E00",  # Vermillon (remplace le Rouge)
            "input":  "#009E73",  # Vert bleuté (remplace le Vert)
            "output": "#56B4E9",  # Bleu ciel (remplace le Bleu standard)
            "h_norm": "#0072B2",  # Bleu foncé (remplace le Violet)
            "c_norm": "#E69F00",  # Orange (remplace l'Orange précédent, plus distinct)
            }

        # --- PREPARATION (Idem précédent) ---
        text_val = lstm_sequence_cb.value
        words = text_val.split()
        vocab = {word: i for i, word in enumerate(sorted(list(set(words))))}
        vocab_size = len(vocab)
        embed_dim = 16
        hidden_dim = lstm_hidden_slider_cb.value

        # --- MODELE ET BOUCLE (Idem précédent) ---
        lstm_embedding_layer = nn.Embedding(vocab_size, embed_dim)
        lstm_cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_dim)
        h_t = torch.zeros(1, hidden_dim)
        c_t = torch.zeros(1, hidden_dim)
        indices = torch.tensor([vocab[w] for w in words])
        word_embeddings = lstm_embedding_layer(indices)

        history = {
            "h": [h_t.detach().numpy().flatten()], "c": [c_t.detach().numpy().flatten()],
            "gate_forget": [], "gate_input": [], "gate_output": [],
            "h_norm": [0], "c_norm": [0]
            }

        for t in range(len(words)):
            x_t = word_embeddings[t].unsqueeze(0)
            h_prev = h_t.clone()
            h_t, c_t = lstm_cell(x_t, (h_t, c_t))

            with torch.no_grad():
                gates_raw = (x_t @ lstm_cell.weight_ih.T + h_prev @ lstm_cell.weight_hh.T +
                             lstm_cell.bias_ih + lstm_cell.bias_hh)
                i_raw, f_raw, g_raw, o_raw = gates_raw.chunk(4, dim=1)
                history["gate_input"].append(torch.sigmoid(i_raw).mean().item())
                history["gate_forget"].append(torch.sigmoid(f_raw).mean().item())
                history["gate_output"].append(torch.sigmoid(o_raw).mean().item())

            history["h"].append(h_t.detach().numpy().flatten())
            history["c"].append(c_t.detach().numpy().flatten())
            history["h_norm"].append(np.linalg.norm(h_t.detach().numpy()))
            history["c_norm"].append(np.linalg.norm(c_t.detach().numpy()))

        arr_h = np.array(history["h"][1:]).T
        arr_c = np.array(history["c"][1:]).T

        # --- VISUALISATION ADAPTÉE ---
        fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "<b>1. Activité Moyenne des Portes</b>",
                    "<b>2. État Caché (Court Terme)</b> $h_t$",
                    "<b>3. Norme des Vecteurs</b> (Force du signal)",
                    "<b>4. État de Cellule (Long Terme)</b> $C_t$"
                    ),
                vertical_spacing=0.15, horizontal_spacing=0.1
                )

        x_axis = words

        # Plot 1: Les Portes (Utilisation des nouvelles couleurs CB)
        # Note : J'ai aussi changé les styles de ligne pour ajouter une distinction non-couleur
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=history["gate_forget"], name='Forget Gate (Oubli)',
                line=dict(color=CB_COLORS["forget"], width=3, dash='solid')
                ), row=1, col=1
            )
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=history["gate_input"], name='Input Gate (Ajout)',
                line=dict(color=CB_COLORS["input"], width=3, dash='dot')
                ), row=1, col=1
            )
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=history["gate_output"], name='Output Gate (Sortie)',
                line=dict(color=CB_COLORS["output"], width=2, dash='dash')
                ), row=1, col=1
            )

        # Plot 2: Heatmap h_t (Viridis est 'colorblind-safe')
        fig.add_trace(
            go.Heatmap(
                z=arr_h, x=x_axis, y=[f"N{i}" for i in range(hidden_dim)],
                colorscale='Viridis', showscale=False, name="Hidden State"
                ), row=1, col=2
            )

        # Plot 3: Normes (Nouvelles couleurs CB)
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=history["h_norm"][1:], name='||h_t|| (Court terme)',
                line=dict(color=CB_COLORS["h_norm"], width=2)
                ), row=2, col=1
            )
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=history["c_norm"][1:], name='||C_t|| (Long terme)',
                line=dict(color=CB_COLORS["c_norm"], width=3)
                ), row=2, col=1
            )

        # Plot 4: Heatmap C_t (Remplacement de Plasma par CIVIDIS)
        # Cividis est spécifiquement conçue pour les daltoniens.
        fig.add_trace(
            go.Heatmap(
                z=arr_c, x=x_axis, y=[f"N{i}" for i in range(hidden_dim)],
                colorscale='Cividis', showscale=False, name="Cell State"
                ), row=2, col=2
            )

        fig.update_layout(
            height=700, title_text=f"🧠 IRM d'un LSTM (Mode Accessible)", template="plotly_white",
            legend=dict(orientation="h", y=-0.1)
            )
        fig.update_xaxes(tickangle=-45)

        # Mise à jour du texte explicatif pour refléter les nouvelles couleurs
        return mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"""
            **Guide des Couleurs (Palette Accessible Okabe-Ito) :**
        
            * **Graphique 1 (Portes)** :
                * <span style="color:{CB_COLORS['forget']}">■</span> **Vermillon (Ligne pleine)** : Forget Gate (Oubli).
                * <span style="color:{CB_COLORS['input']}">■</span> **Vert Bleuté (Pointillés)** : Input Gate (Ajout).
                * <span style="color:{CB_COLORS['output']}">■</span> **Bleu Ciel (Tirets)** : Output Gate.
        
            * **Graphique 3 (Normes)** :
                * <span style="color:{CB_COLORS['c_norm']}">■</span> **Orange (Ligne épaisse)** : Mémoire Long Terme ($C_t$).
                * <span style="color:{CB_COLORS['h_norm']}">■</span> **Bleu Foncé (Ligne fine)** : Mémoire Court Terme ($h_t$).
            """
                            ), kind="info"
                        ),
                    mo.ui.plotly(fig)
                    ]
                )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ---
        
            # 4. GRU : Alternative simplifiee
        
            Le **GRU** (Cho et al., 2014) simplifie le LSTM :
            - **2 portes** au lieu de 3 : reset ($r_t$) et update ($z_t$)
            - **1 seul etat** $h_t$ (pas de $C_t$ separe)
        
            ## Equations du GRU
        
            update gate:
        
            $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
        
            reset gate:
        
            $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
        
            $$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
        
            $$\boxed{h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t}$$
            """
        )
    return


@app.cell
def _(nn, torch):
    # === GRU : IMPLEMENTATION ===

    torch.manual_seed(42)

    gru_cell = nn.GRUCell(input_size=4, hidden_size=3)

    x_gru = torch.randn(1, 4)
    h_gru = torch.randn(1, 3)

    # Forward
    h_gru_out = gru_cell(x_gru, h_gru)

    # Verification manuelle
    with torch.no_grad():
        W_ir, W_iz, W_in = gru_cell.weight_ih.chunk(3)  # (3, 4) each
        W_hr, W_hz, W_hn = gru_cell.weight_hh.chunk(3)  # (3, 3) each
        b_ir, b_iz, b_in = gru_cell.bias_ih.chunk(3)
        b_hr, b_hz, b_hn = gru_cell.bias_hh.chunk(3)

        r_t_gru = torch.sigmoid(x_gru @ W_ir.T + h_gru @ W_hr.T + b_ir + b_hr)
        z_t_gru = torch.sigmoid(x_gru @ W_iz.T + h_gru @ W_hz.T + b_iz + b_hz)
        n_t_gru = torch.tanh(x_gru @ W_in.T + b_in + r_t_gru * (h_gru @ W_hn.T + b_hn))
        h_manual_gru = (1 - z_t_gru) * h_gru + z_t_gru * n_t_gru

    print("=== GRU ===")
    print(f"r_t (reset):  {r_t_gru.squeeze().tolist()}")
    print(f"z_t (update): {z_t_gru.squeeze().tolist()}")
    print(f"h_t PyTorch:  {h_gru_out.squeeze().tolist()}")
    print(f"h_t Manuel:   {h_manual_gru.squeeze().tolist()}")
    print(f"Diff: {(h_gru_out - h_manual_gru).abs().max().item():.2e}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Comparatif RNN vs LSTM vs GRU
        
            | Aspect | RNN | LSTM | GRU |
            |--------|-----|------|-----|
            | **Portes** | 0 | 3 (f, i, o) | 2 (r, z) |
            | **Etats** | 1 ($h$) | 2 ($h$, $C$) | 1 ($h$) |
            | **Parametres** | $O(h^2)$ | $O(4h^2)$ | $O(3h^2)$ |
            | **Vanishing gradient** | Problematique | Resolu | Resolu |
            | **Cas d'usage** | Sequences courtes | NLP, longues deps | Compromis perf/vitesse |
            """
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ---
        
            # 5. Application : Classification de Sentiments
        
            Entrainons un LSTM sur le dataset **Allocine** (critiques de films en francais).
        
            ```
            Texte -> Tokenization -> Embedding -> LSTM -> Linear -> Prediction
            ```
            """
        )
    return


@app.cell
def _():
    from datasets import load_dataset
    import torch.utils.data as data_utils

    allocine = load_dataset("allocine", split="train[:3000]")
    allocine_test = load_dataset("allocine", split="test[:500]")

    print(f"Train: {len(allocine)} | Test: {len(allocine_test)}")
    print(f"Exemple: {allocine[0]['review'][:80]}...")
    print(f"Label: {allocine[0]['label']} (0=negatif, 1=positif)")
    return allocine, allocine_test, data_utils


@app.cell
def _(allocine, np):
    import re
    from collections import Counter

    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    all_tokens = []
    for review in allocine["review"]:
        all_tokens.extend(tokenize(review))

    vocab_size_sentiment = 5000
    token_counts = Counter(all_tokens)
    most_common = token_counts.most_common(vocab_size_sentiment - 2)

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)

    print(f"Vocabulaire: {len(word2idx)} mots")
    print(f"Top 5: {most_common[:5]}")

    review_lengths = [len(tokenize(r)) for r in allocine["review"]]
    print(f"Longueur: moyenne={np.mean(review_lengths):.0f}, mediane={np.median(review_lengths):.0f}")
    return tokenize, vocab_size_sentiment, word2idx


@app.cell
def _(allocine, allocine_test, data_utils, tokenize, torch, word2idx):
    max_len = 100

    def encode_review(review, max_length):
        tokens = tokenize(review)[:max_length]
        indices = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
        if len(indices) < max_length:
            indices += [word2idx["<PAD>"]] * (max_length - len(indices))
        return indices

    class SentimentDataset(data_utils.Dataset):
        def __init__(self, dataset, max_length):
            self.reviews = [encode_review(r, max_length) for r in dataset["review"]]
            self.labels = dataset["label"]

        def __len__(self):
            return len(self.reviews)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.reviews[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long)
                )

    train_dataset = SentimentDataset(allocine, max_len)
    test_dataset = SentimentDataset(allocine_test, max_len)

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    print(f"Shape exemple: {train_dataset[0][0].shape}")
    return encode_review, max_len, test_dataset, train_dataset


@app.cell
def _(mo):
    mo.md(
        r"""
            ### Modele LSTM pour la classification
            """
        )
    return


@app.cell
def _(nn, torch):
    class SentimentLSTM(nn.Module):
        def __init__(
                self, vocab_size, embedding_dim, hidden_dim, output_dim,
                n_layers=1, bidirectional=False, dropout=0.3
                ):
            super().__init__()

            # Embedding: mot -> vecteur dense
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

            # LSTM: sequence -> representation
            self.lstm = nn.LSTM(
                    embedding_dim, hidden_dim,
                    num_layers=n_layers,
                    bidirectional=bidirectional,
                    dropout=dropout if n_layers > 1 else 0,
                    batch_first=True
                    )

            # Classifieur
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.fc = nn.Linear(lstm_output_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, text):
            # text: (batch, seq_len)
            embedded = self.dropout(self.embedding(text))
            # embedded: (batch, seq_len, embed_dim)

            output, (hidden, cell) = self.lstm(embedded)
            # hidden: (n_layers * n_directions, batch, hidden_dim)

            if self.lstm.bidirectional:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]

            return self.fc(self.dropout(hidden))

    return (SentimentLSTM,)


@app.cell
def _(mo):
    mo.md(
        r"""
            ### Configuration de l'entrainement
            """
        )
    return


@app.cell
def _(mo):
    embed_dim_slider = mo.ui.slider(start=32, stop=256, step=32, value=64, label="Dim embedding")
    hidden_dim_slider = mo.ui.slider(start=32, stop=256, step=32, value=128, label="Dim LSTM")
    n_layers_slider = mo.ui.slider(start=1, stop=3, step=1, value=1, label="Nb couches")
    bidirectional_checkbox = mo.ui.checkbox(value=True, label="Bidirectionnel")
    batch_size_select = mo.ui.dropdown(options={"32": 32, "64": 64, "128": 128}, value="64", label="Batch")
    lr_slider = mo.ui.slider(start=0.0001, stop=0.01, step=0.0001, value=0.001, label="Learning rate")
    epochs_select = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Epoques")

    mo.vstack(
            [
                mo.hstack([embed_dim_slider, hidden_dim_slider]),
                mo.hstack([n_layers_slider, bidirectional_checkbox]),
                mo.hstack([batch_size_select, lr_slider, epochs_select])
                ]
            )
    return (
        batch_size_select,
        bidirectional_checkbox,
        embed_dim_slider,
        epochs_select,
        hidden_dim_slider,
        lr_slider,
        n_layers_slider,
        )


@app.cell
def _(
        SentimentLSTM,
        batch_size_select,
        bidirectional_checkbox,
        data_utils,
        embed_dim_slider,
        hidden_dim_slider,
        mo,
        n_layers_slider,
        test_dataset,
        train_dataset,
        vocab_size_sentiment,
        ):
    sentiment_model = SentimentLSTM(
            vocab_size=vocab_size_sentiment,
            embedding_dim=embed_dim_slider.value,
            hidden_dim=hidden_dim_slider.value,
            output_dim=2,
            n_layers=n_layers_slider.value,
            bidirectional=bidirectional_checkbox.value,
            dropout=0.3
            )

    total_params = sum(p.numel() for p in sentiment_model.parameters())

    train_loader = data_utils.DataLoader(train_dataset, batch_size=int(batch_size_select.value), shuffle=True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=int(batch_size_select.value), shuffle=False)

    mo.md(
        f"""
    **Modele**: {"Bi-" if bidirectional_checkbox.value else ""}LSTM | **Params**: {total_params:,}
    """
        )
    return sentiment_model, test_loader, train_loader


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Lancer l'entrainement")
    train_button
    return (train_button,)


@app.cell
def _(
        epochs_select,
        go,
        lr_slider,
        mo,
        nn,
        sentiment_model,
        test_loader,
        torch,
        train_button,
        train_loader,
        ):
    mo.stop(not train_button.value, mo.md("Cliquez sur le bouton pour lancer l'entrainement"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_train = sentiment_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=lr_slider.value)

    train_losses, train_accs, test_accs = [], [], []

    def calc_acc(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean()

    for epoch in range(epochs_select.value):
        model_to_train.train()
        epoch_loss, epoch_acc, n_batches = 0, 0, 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model_to_train(texts)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += calc_acc(preds, labels).item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)
        train_accs.append(epoch_acc / n_batches)

        model_to_train.eval()
        test_acc, test_batches = 0, 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                test_acc += calc_acc(model_to_train(texts), labels).item()
                test_batches += 1
        test_accs.append(test_acc / test_batches)

    fig_training = go.Figure()
    fig_training.add_trace(
        go.Scatter(
            x=list(range(1, len(train_losses) + 1)), y=train_losses,
            name='Loss', yaxis='y2'
            )
        )
    fig_training.add_trace(
        go.Scatter(
            x=list(range(1, len(train_accs) + 1)),
            y=[a * 100 for a in train_accs], name='Train Acc'
            )
        )
    fig_training.add_trace(
        go.Scatter(
            x=list(range(1, len(test_accs) + 1)),
            y=[a * 100 for a in test_accs], name='Test Acc'
            )
        )
    fig_training.update_layout(
            title="Courbes d'entrainement",
            xaxis_title="Epoque", yaxis_title="Accuracy (%)",
            yaxis2=dict(title="Loss", overlaying='y', side='right'),
            height=350
            )

    mo.vstack(
            [
                mo.md(f"**Termine!** Train: **{train_accs[-1] * 100:.1f}%** | Test: **{test_accs[-1] * 100:.1f}%**"),
                fig_training
                ]
            )
    return device, model_to_train


@app.cell
def _(mo):
    mo.md(
        r"""
            ### Testez le modele !
            """
        )
    return


@app.cell
def _(mo):
    user_review = mo.ui.text_area(
            value="Ce film est vraiment excellent, les acteurs jouent a merveille !",
            label="Votre critique",
            rows=2
            )
    user_review
    return (user_review,)


@app.cell
def _(
        F,
        device,
        encode_review,
        max_len,
        mo,
        model_to_train,
        torch,
        user_review,
        ):
    model_to_train.eval()
    with torch.no_grad():
        encoded = encode_review(user_review.value, max_len)
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
        output = model_to_train(input_tensor)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        conf = probs[0][pred].item() * 100

    sentiment = "Positif" if pred == 1 else "Negatif"
    mo.md(f"**Prediction**: {sentiment} ({conf:.0f}% confiance)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ---
        
            # Resume
        
            | | RNN | LSTM | GRU |
            |-|-----|------|-----|
            | **Formule cle** | $h_t = \tanh(Wx + Wh + b)$ | $C_t = f \odot C + i \odot \tilde{C}$ | $h_t = (1-z) \odot h + z \odot \tilde{h}$ |
            | **Portes** | 0 | 3 | 2 |
            | **Vanishing gradient** | Oui | Non | Non |
            | **Quand l'utiliser** | Sequences courtes | Longues dependances | Compromis |
        
            ## Pour aller plus loin
        
            - **Attention** : permet de "regarder" differentes parties de la sequence
            - **Transformer** : remplace les RNN dans beaucoup d'applications modernes
            - **BERT, GPT** : modeles pre-entraines bases sur Transformers
            """
        )
    return


if __name__ == "__main__":
    app.run()
