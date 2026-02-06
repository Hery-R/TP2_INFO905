import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    from pathlib import Path

    # Chemin relatif au fichier notebook
    _asset_dir = Path(__file__).parent / "asset"

    mo.vstack(
            [
                mo.md(
                    r"""
                        # Word Embedding (Représentation vectorielle des mots)
                    
                        **Word Embedding** est une représentation des mots qui permet à des mots ayant un sens similaire d'avoir une représentation semblable. Il s'agit d'une méthode d'apprentissage non supervisé sur un vaste corpus textuel, où le modèle apprend à prédire un mot à partir de son contexte ou inversement. Une fois entraîné, cette méthode produit des représentations vectorielles où des mots proches dans cet espace à haute dimension sont censés être sémantiquement similaires.
                    
                        Contrairement à une simple assignation de vecteurs uniques par mot, les embeddings capturent des similarités **sémantiques** ou **syntaxiques** basées sur le corpus d'entraînement. Les vecteurs d'embedding contiennent souvent des centaines de dimensions et identifient des relations nuancées entre les mots.
                            """
                    ),
                mo.image(src=_asset_dir / "word_embed.png"),
                mo.md(
                    r"""
                        ---
                    
                        ## Couche d'Embedding (Embedding Layer)
                    
                        Une **couche d'embedding** en apprentissage machine permet de créer des représentations vectorielles (embeddings) à partir de séquences d'entrée. Elle associe des mots ou des indices entiers à des vecteurs denses de nombres réels.
                    
                        ### Processus :
                        - En entrée : une séquence d'indices de mots (par exemple, des entiers correspondant à des mots dans un vocabulaire).
                        - En sortie : un tenseur où chaque séquence garde sa longueur originale, mais chaque mot/entier est représenté par un vecteur dense.
                    
                        Ces vecteurs capturent les relations **sémantiques** entre les mots. La dimensionnalité de ces vecteurs est un hyperparamètre que l'on peut ajuster selon la tâche.
                    
                        ---
                    
                        ### Utilisation basique
                    
                        La couche d'embedding agit uniquement comme une **table de correspondance**. Chaque index est associé à un vecteur dense qui peut être mis à jour lors de l'entraînement.
                            """
                    )
                ]
            )
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.data as data
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    return F, data, go, nn, np, pd, torch


@app.cell
def _(mo):
    # UI pour configurer la couche d'embedding de démonstration
    embedding_dim_slider = mo.ui.slider(
            start=2, stop=16, step=1, value=5,
            label="Dimension de l'embedding"
            )
    word_selector = mo.ui.dropdown(
            options={"hello": "hello", "world": "world"},
            value="hello",
            label="Mot à visualiser"
            )
    mo.hstack([embedding_dim_slider, word_selector])
    return embedding_dim_slider, word_selector


@app.cell
def _(embedding_dim_slider, mo, nn, torch, word_selector):
    # Création d'un dictionnaire qui associe chaque mot à un index unique
    word_to_ix = {"hello": 0, "world": 1}

    # Initialisation de la couche d'embedding avec dimension configurable
    embeds = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim_slider.value)

    # Transformation du mot sélectionné en tenseur
    lookup_tensor = torch.tensor([word_to_ix[word_selector.value]], dtype=torch.long)

    # Récupération de l'embedding correspondant au mot sélectionné
    selected_embed = embeds(lookup_tensor)

    mo.md(
        f"""
    **Configuration actuelle:**
    - Mot sélectionné: `{word_selector.value}` (index: {word_to_ix[word_selector.value]})
    - Dimension de l'embedding: {embedding_dim_slider.value}
    - Vecteur d'embedding: `{selected_embed.detach().numpy()}`
    """
        )
    return (embeds,)


@app.cell
def _(mo):
    mo.md(
        r"""
            Paramètres d'entrainements
            """
        )
    return


@app.cell
def _(embeds, mo):
    params_list = [str(param.data) for param in embeds.parameters()]
    mo.md(
        f"""
    **Paramètres d'entraînement (matrice d'embedding):**
    ```
    {params_list[0]}
    ```
    """
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            Juste pour le fun on définie nos propres paramètres
            """
        )
    return


@app.cell
def _(nn, torch):
    # Création d'une nouvelle couche d'embedding pour démonstration
    embeds_custom = nn.Embedding(num_embeddings=2, embedding_dim=5)

    embedding_lookup = torch.tensor(
            [
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
                ], dtype=torch.float32
            )
    embeds_custom.weight = nn.Parameter(embedding_lookup)
    for param_custom in embeds_custom.parameters():
        print(param_custom)
    return (embeds_custom,)


@app.cell
def _(mo):
    mo.md(
        r"""
            Comme vous pouvez le constater, si je sélectionne l'index 0 ou 1, j'obtiens ma ligne embedding_lookup
            """
        )
    return


@app.cell
def _(embeds_custom, torch):
    print(embeds_custom(torch.tensor([0])))
    print(embeds_custom(torch.tensor([1])))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## GPT embedding
        
            Regardons le tout premier modèle GPT et voyons la taille de la couche d'embedding.
            """
        )
    return


@app.cell
def _():
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_model = GPT2Model.from_pretrained("gpt2")

    inputs_gpt = tokenizer("Hello, my dog is cute", return_tensors="pt")

    print("vocab size", tokenizer.vocab_size)

    # expected Embedding(50257, 768)
    # 50257 = vocabulary size
    # 768 = number of features
    print("Embedding size", gpt_model.wte)
    print(inputs_gpt)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            [Regardont le code de ce GPT](https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L667)
            """
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Entraîner la première couche d'embedding
        
            Dans cette section, nous allons entraîner notre première couche d'embedding sur des critiques de films en français (dataset Allociné) !
            Pour commencer, nous entraînerons uniquement cette couche sur les lettres composant les mots.
            """
        )
    return


@app.cell
def _():
    from datasets import load_dataset

    # Chargement du dataset Allociné (critiques de films en français)
    # Le dataset contient des critiques avec des labels de sentiment (0: négatif, 1: positif)
    allocine_dataset = load_dataset("allocine", split="train[:5000]")
    allocine_dataset
    return (allocine_dataset,)


@app.cell
def _(allocine_dataset, pd):
    # Convertir en DataFrame pour visualisation
    df = pd.DataFrame(allocine_dataset)
    # Compter les différentes valeurs de label (0: négatif, 1: positif)
    df
    return


@app.cell
def _(allocine_dataset):
    # Récupérer les critiques (texte en français)
    reviews = allocine_dataset["review"]
    return (reviews,)


@app.cell
def _(mo):
    mo.md(
        r"""
            Maintenant, nous allons créer une séquence de lettres basée sur des phrases.
            Par exemple :
        
            ```
            aba decides a
            ```
        
            produira:
        
            ```
            [
              ('a', 'b'),
              ('b', 'a'),
              ('a', ' '),
              (' ', 'd'),
              ('d', 'e'),
              ('e', 'c'),
              ('c', 'i'),
              ('i', 'd'),
              ('d', 'e'),
              ('e', 's'),
              ('s', ' '),
              (' ', 'a'),
            ]
            ```
            """
        )
    return


@app.cell
def _(reviews):
    import itertools as it
    import re

    def sliding_window(txt):
        # Génère des paires de caractères (bigrammes) à partir du texte donné
        # Exemple : 'chat' -> ('c', 'h'), ('h', 'a'), ('a', 't')
        for i in range(len(txt) - 1):
            yield txt[i], txt[i + 1]

    window = []

    for title in reviews:
        # Nettoie chaque titre pour ne conserver que les lettres (a-z) et les chiffres (0-9)
        title = re.sub('[^a-zA-Z0-9]+', '', title.lower())

        # Applique la fonction sliding_window au titre nettoyé et l'ajoute à la liste `window`
        window.append(sliding_window(title))

    window = list(it.chain(*window))

    # Affiche le nombre total de paires (bigrammes) générées
    print(len(window))

    # Affiche les 5 premières paires générées en guise d'exemple
    window[:5]
    return (window,)


@app.cell
def _(mo):
    mo.md(
        r"""
            Maintenant, effectuons un encodage one-hot de manière à ce qu'une lettre corresponde à un identifiant (comme un identifiant dans une table SQL).
        
            ```
            {' ': 2,
             'a': 0,
             'b': 1,
             'c': 5,
             'd': 3,
             'e': 4,
             'g': 8,
             'i': 6,
             'l': 16,
             'm': 12,
             'n': 9,
             'o': 11,
             'r': 15,
             's': 7,
             't': 10,
             'u': 13,
             'y': 14}
            ```
            """
        )
    return


@app.cell
def _(np, pd, window):
    # Mapping lettre avec un ID
    mapping = {c: i for i, c in enumerate(pd.DataFrame(window)[0].unique())}
    # Id en entrée du modèle
    integers_in = np.array([mapping[w[0]] for w in window])
    # Id en sortie du modèle
    integers_out = np.array([mapping[w[1]] for w in window])

    print("Shape of input", integers_in.shape)
    print("Input example", integers_in[0], integers_out[0])
    print("Show generate mapping\n", mapping)
    return integers_in, integers_out, mapping


@app.cell
def _(mo):
    mo.md(
        r"""
            ### La classe Dataset
        
            La classe `Dataset` résume les fonctionnalités de base d'un jeu de données de manière naturelle.
            Pour définir un jeu de données dans PyTorch, il suffit d'implémenter deux fonctions principales : `__getitem__` et `__len__`.
        
            1. **`__getitem__`** : Cette fonction doit retourner le i-ème échantillon du jeu de données.
            2. **`__len__`** : Cette fonction retourne la taille totale du jeu de données.
        
            Ces deux fonctions garantissent une structure cohérente et standardisée pour interagir avec vos données.
            """
        )
    return


@app.cell
def _(data, torch):
    from typing import List, Tuple

    class NextLetterDataset(data.Dataset):
        def __init__(self, _integers_in: List[int], _integers_out: List[int]):
            self.integers_in = _integers_in  # Stocke les données d'entrée
            self.integers_out = _integers_out  # Stocke les étiquettes de sortie

        def __len__(self):
            return len(self.integers_in)

        def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
            """
            Retourne le i-ème échantillon et son étiquette à partir du dataset.
            Les données et étiquettes sont converties en tenseurs PyTorch avant d'être renvoyées.

            Args:
            - idx (int): L'index de l'échantillon à récupérer.

            Returns:
            - Tuple[torch.tensor, torch.tensor]: Une paire contenant :
                - Le tenseur représentant la lettre en entrée
                - Le tenseur représentant la lettre en sortie
            """
            data_point = self.integers_in[idx]
            data_label = self.integers_out[idx]
            return torch.tensor(data_point), torch.tensor(data_label, dtype=torch.int64)

    return (NextLetterDataset,)


@app.cell
def _(mo):
    from pathlib import Path as _Path

    _asset_path = _Path(__file__).parent / "asset"

    mo.vstack(
            [
                mo.md(
                    r"""
                        ## Construire le premier modèle d'embedding
                        Nous allons construire un réseau simple pour prédire la lettre suivante.
                            """
                    ),
                mo.image(src=_asset_path / "next_letter_prediction.png")
                ]
            )
    return


@app.cell
def _(F, torch):
    class NextLetterPrediction(torch.nn.Module):
        def __init__(self, vocab_size, embedding_size):
            super(NextLetterPrediction, self).__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
            self.fc = torch.nn.Linear(embedding_size, vocab_size)

        def forward(self, x):
            x = F.relu(self.embedding(x))
            x = self.fc(x)
            return x

    return (NextLetterPrediction,)


@app.cell
def _(mo):
    mo.md(
        r"""
            ### Visualisation des lettres avant l'entraînement
            Visualisons les embeddings des lettres avant l'entraînement du modèle.
            """
        )
    return


@app.cell
def _(NextLetterPrediction, embedding_size_model, mapping, mo):
    model = NextLetterPrediction(
            vocab_size=len(mapping),
            embedding_size=embedding_size_model.value
            )
    mo.md(
        f"""
    **Modèle créé avec:**
    - Taille du vocabulaire: {len(mapping)}
    - Dimension de l'embedding: {embedding_size_model.value}
    """
        )
    return (model,)


@app.cell
def _(go, mapping, model, np, torch):
    idx_to_calc = list(mapping.values())
    idx_to_calc = np.array([idx_to_calc]).T

    translator = {v: k for k, v in mapping.items()}
    preds = model.embedding(torch.tensor(idx_to_calc)).detach().numpy()

    # Créer un graphique Plotly interactif
    fig_before = None
    fig_before = go.Figure()
    fig_before.add_trace(
            go.Scatter(
                    x=preds[:, 0, 0],
                    y=preds[:, 0, 1],
                    mode='text',
                    text=[translator[idx[0]] for idx in idx_to_calc],
                    textfont=dict(size=14),
                    hoverinfo='text',
                    hovertext=[f"Lettre: {translator[idx[0]]}<br>x: {preds[i, 0, 0]:.3f}<br>y: {preds[i, 0, 1]:.3f}"
                               for i, idx in enumerate(idx_to_calc)]
                    )
            )
    fig_before.update_layout(
            title="Embeddings avant entraînement",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            height=500
            )
    fig_before
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ### Train loop
        
            Configurez les hyperparamètres d'entraînement ci-dessous:
            """
        )
    return


@app.cell
def _(mo):
    # Sliders pour les hyperparamètres d'entraînement
    epochs_slider = mo.ui.slider(
            start=1, stop=10, step=1, value=1,
            label="Nombre d'époques"
            )
    batch_size_slider = mo.ui.slider(
            start=32, stop=512, step=32, value=128,
            label="Taille du batch"
            )
    learning_rate_slider = mo.ui.slider(
            start=0.001, stop=0.1, step=0.001, value=0.01,
            label="Taux d'apprentissage"
            )
    embedding_size_model = mo.ui.slider(
            start=2, stop=32, step=1, value=2,
            label="Dimension embedding (modèle)"
            )

    mo.vstack(
            [
                mo.hstack([epochs_slider, batch_size_slider]),
                mo.hstack([learning_rate_slider, embedding_size_model])
                ]
            )
    return (
        batch_size_slider,
        embedding_size_model,
        epochs_slider,
        learning_rate_slider,
        )


@app.cell
def _(
        NextLetterDataset,
        batch_size_slider,
        data,
        integers_in,
        integers_out,
        learning_rate_slider,
        model,
        nn,
        ):
    # Initialisation du dataset dans le DataLoader avec taille de batch configurable
    dataset = NextLetterDataset(integers_in, integers_out)
    trainloader = data.DataLoader(dataset, batch_size=batch_size_slider.value, shuffle=True)

    # Fonction de perte CrossEntropyLoss pour classification multi-classes
    criterion = nn.CrossEntropyLoss()

    # Optimiseur AdamW avec taux d'apprentissage configurable
    import torch as torch_optim
    optimizer = torch_optim.optim.AdamW(model.parameters(), lr=learning_rate_slider.value)
    return criterion, optimizer, trainloader


@app.cell
def _(mo):
    run_button = mo.ui.run_button()
    run_button
    return (run_button,)


@app.cell
def _(
        criterion,
        epochs_slider,
        mo,
        model,
        optimizer,
        run_button,
        torch,
        trainloader,
        ):
    mo.stop(not run_button.value, mo.md("Click 👆 to run this cell"))
    # Vérifie si un GPU CUDA est disponible ; sinon, utilise le CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle sur le dispositif choisi (GPU ou CPU) en mode entraînement.
    model_train = model.to(device)
    model_train.train()

    # Historique des pertes pour visualisation
    loss_history = []

    # Boucle principale d'entraînement
    for epoch in range(epochs_slider.value):
        running_loss = 0.0
        epoch_loss = 0.0
        n_batches = 0

        for i_batch, batch in enumerate(trainloader, 0):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model_train(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            n_batches += 1

        # Enregistrer la perte moyenne par époque
        loss_history.append(epoch_loss / n_batches)

    mo.md(
        f"""
    **Entraînement terminé!**
    - Device: `{device}`
    - Époques: {epochs_slider.value}
    - Perte finale: {loss_history[-1]:.4f}
    """
        )
    return device, loss_history, model_train


@app.cell
def _(go, loss_history, mo):
    # Graphique de l'historique des pertes
    fig_loss = go.Figure()
    fig_loss.add_trace(
            go.Scatter(
                    x=list(range(1, len(loss_history) + 1)),
                    y=loss_history,
                    mode='lines+markers',
                    name='Perte',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                    )
            )
    fig_loss.update_layout(
            title="Évolution de la perte pendant l'entraînement",
            xaxis_title="Époque",
            yaxis_title="Perte moyenne",
            height=400
            )
    mo.vstack([mo.md("### Courbe d'apprentissage"), fig_loss])
    return


@app.cell
def _(device, go, mapping, mo, model_train, np, torch):
    idx_to_calc_after = list(mapping.values())
    idx_to_calc_after = np.array([idx_to_calc_after]).T

    translator_after = {v: k for k, v in mapping.items()}
    preds_after = model_train.embedding(torch.tensor(idx_to_calc_after).to(device)).cpu().detach().numpy()

    # Graphique Plotly interactif pour les embeddings après entraînement
    fig_after = go.Figure()
    fig_after.add_trace(
            go.Scatter(
                    x=preds_after[:, 0, 0],
                    y=preds_after[:, 0, 1],
                    mode='text+markers',
                    text=[translator_after[idx[0]] for idx in idx_to_calc_after],
                    textfont=dict(size=14),
                    marker=dict(size=10, opacity=0.6),
                    hoverinfo='text',
                    hovertext=[
                        f"Lettre: {translator_after[idx[0]]}<br>x: {preds_after[i, 0, 0]:.3f}<br>y: {preds_after[i, 0, 1]:.3f}"
                        for i, idx in enumerate(idx_to_calc_after)]
                    )
            )
    fig_after.update_layout(
            title="Embeddings après entraînement",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            height=500
            )
    mo.vstack(
            [
                mo.md("### Visualisation des embeddings appris"),
                fig_after
                ]
            )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
            ## Comparaison sémantique avec CamemBERT
        
            Utilisez les champs ci-dessous pour comparer la similarité sémantique entre deux mots français.
            """
        )
    return


@app.cell
def _(mo):
    # UI pour la comparaison de mots
    word1_input = mo.ui.text(value="reine", label="Premier mot")
    word2_input = mo.ui.text(value="roi", label="Deuxième mot")
    mo.hstack([word1_input, word2_input], justify="start")
    return word1_input, word2_input


@app.cell
def _(F, mo, torch, word1_input, word2_input):
    from transformers import CamembertModel, CamembertTokenizer

    # Chargement du modèle CamemBERT
    camembert_model = CamembertModel.from_pretrained("camembert-base")
    camembert_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # Comparaison des mots saisis par l'utilisateur
    word1 = word1_input.value.strip() or "reine"
    word2 = word2_input.value.strip() or "roi"

    def get_word_embedding(word, model, tokenizer):
        """Extrait l'embedding d'un mot en utilisant la moyenne des tokens (excluant [CLS] et [SEP])"""
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # last_hidden_state: [batch, seq_len, hidden_dim]
        # On exclut le premier token ([CLS]) et le dernier ([SEP])
        # et on fait la moyenne des embeddings des tokens du mot
        hidden_states = outputs.last_hidden_state[0, 1:-1, :]  # Exclure [CLS] et [SEP]
        return hidden_states.mean(dim=0)  # Moyenne sur les tokens

    embed1 = get_word_embedding(word1, camembert_model, camembert_tokenizer)
    embed2 = get_word_embedding(word2, camembert_model, camembert_tokenizer)

    # Calcul de la similarité cosinus sur les embeddings moyennés
    cosine_sim = F.cosine_similarity(embed1.unsqueeze(0), embed2.unsqueeze(0), dim=1)
    sim_value = cosine_sim.item()

    # Interprétation du score (seuils ajustés pour les embeddings de tokens)
    if sim_value > 0.85:
        interpretation = "Très similaires"
    elif sim_value > 0.7:
        interpretation = "Assez similaires"
    elif sim_value > 0.5:
        interpretation = "Modérément similaires"
    else:
        interpretation = "Peu similaires"

    mo.md(
        f"""
    ### Résultat de la comparaison

    | Mot 1 | Mot 2 | Similarité cosinus | Interprétation |
    |-------|-------|-------------------|----------------|
    | **{word1}** | **{word2}** | **{sim_value:.4f}** | {interpretation} |

    > **Conseil:** Essayez des paires comme "chat/chien", "paris/france", "heureux/triste", "manger/boire"
    """
        )
    return


if __name__ == "__main__":
    app.run()
