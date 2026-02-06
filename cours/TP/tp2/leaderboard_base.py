"""
Module de base pour les leaderboards Gradio.

Ce module contient le code générique partagé entre les différents leaderboards:
- Oracle (dataset tabulaire avec features numériques)
- Dungeon (dataset séquentiel avec tokens)

Usage:
    from leaderboard_base import LeaderboardApp, LeaderboardConfig
"""

import hashlib
import json
import sqlite3
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LeaderboardConfig:
    """Configuration pour un leaderboard."""
    name: str  # Nom du tournoi (ex: "Oracle", "Dungeon")
    title: str  # Titre affiché
    description: str  # Description courte
    db_path: Path  # Chemin vers la base SQLite
    test_secret_path: Path  # Chemin vers le dataset test secret
    val_path: Path  # Chemin vers le dataset de validation
    table_name: str  # Nom de la table SQL
    port: int = 7860  # Port du serveur
    rules_markdown: str = ""  # Règles en markdown


# =============================================================================
# Utilitaires modèle
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_architecture_summary(model: nn.Module) -> dict:
    """Extrait un résumé de l'architecture du modèle."""
    summary = {
        'class_name': model.__class__.__name__,
        'n_params':   count_parameters(model),
        'layers':     []
        }
    for name, module in model.named_modules():
        if name:
            layer_info = {'name': name, 'type': module.__class__.__name__}
            if isinstance(module, nn.Linear):
                layer_info['shape'] = f"{module.in_features}->{module.out_features}"
            elif isinstance(module, nn.Embedding):
                layer_info['shape'] = f"{module.num_embeddings}x{module.embedding_dim}"
            elif isinstance(module, nn.LSTM):
                layer_info['shape'] = f"in={module.input_size}, hidden={module.hidden_size}"
            elif isinstance(module, nn.GRU):
                layer_info['shape'] = f"in={module.input_size}, hidden={module.hidden_size}"
            elif isinstance(module, nn.Dropout):
                layer_info['p'] = module.p
            summary['layers'].append(layer_info)
    return summary


def compute_model_hash(model_path: str) -> str:
    """Calcule le hash MD5 du modèle."""
    with open(model_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def load_model_from_file(model_path: str) -> tuple:
    """
    Charge un modèle et retourne (model, architecture_info) ou (None, error).

    Args:
        model_path: Chemin vers le fichier .pt

    Returns:
        (model, architecture) si succès, (None, error_message) sinon
    """
    try:
        loaded = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(loaded, nn.Module):
            model = loaded
            model.eval()
            architecture = get_architecture_summary(model)
            return model, architecture
        else:
            return None, f"Type de fichier non reconnu: {type(loaded)}"

    except Exception as e:
        return None, f"Erreur de chargement: {str(e)}"


# =============================================================================
# Base de données SQLite
# =============================================================================

class LeaderboardDB:
    """Gestionnaire de base de données pour le leaderboard."""

    def __init__(self, db_path: Path, table_name: str):
        self.db_path = db_path
        self.table_name = table_name

    def init_database(self):
        """Initialise la base de données SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name}
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                model_hash TEXT NOT NULL,
                val_accuracy REAL,
                test_accuracy REAL,
                test_f1 REAL,
                n_params INTEGER,
                architecture TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_best INTEGER DEFAULT 0
            )
        """
            )

        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_team_name
            ON {self.table_name}(team_name)
        """
            )

        conn.commit()
        conn.close()

    def save_submission(
            self,
            team_name: str,
            model_hash: str,
            val_acc: float,
            test_acc: float,
            test_f1: float,
            n_params: int,
            architecture: dict
            ) -> int:
        """Sauvegarde une soumission dans la base."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Vérifier si c'est le meilleur score de l'équipe
        cursor.execute(
            f"""
            SELECT MAX(test_accuracy)
            FROM {self.table_name}
            WHERE team_name = ?
        """, (team_name,)
            )
        best_score = cursor.fetchone()[0]

        is_best = 1 if best_score is None or test_acc > best_score else 0

        # Si c'est le meilleur, reset les autres
        if is_best:
            cursor.execute(
                f"""
                UPDATE {self.table_name}
                SET is_best = 0
                WHERE team_name = ?
            """, (team_name,)
                )

        cursor.execute(
            f"""
            INSERT INTO {self.table_name}
            (team_name, model_hash, val_accuracy, test_accuracy, test_f1,
             n_params, architecture, is_best)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
                    team_name, model_hash, val_acc, test_acc, test_f1,
                    n_params, json.dumps(architecture), is_best
                    )
            )

        submission_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return submission_id

    def get_leaderboard(self) -> pd.DataFrame:
        """Récupère le leaderboard (meilleur score par équipe)."""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query(
            f"""
            SELECT team_name as "Equipe",
                   ROUND(val_accuracy * 100, 2) as "Val Acc (%)",
                   ROUND(test_accuracy * 100, 2) as "Test Secret (%)",
                   ROUND((val_accuracy - test_accuracy) * 100, 2) as "Gap (%)",
                   n_params as "Params",
                   submitted_at as "Soumis le"
            FROM {self.table_name}
            WHERE is_best = 1
            ORDER BY test_accuracy DESC
        """, conn
            )

        conn.close()

        if len(df) > 0:
            df.insert(0, 'Rang', range(1, len(df) + 1))

        return df

    def get_leaderboard_efficient(self) -> pd.DataFrame:
        """
        Récupère le leaderboard efficace.
        Utilise une métrique logarithmique pour pénaliser la taille du modèle
        de manière moins agressive que la division linéaire.
        """
        conn = sqlite3.connect(self.db_path)

        # 1. On récupère les données brutes (plus simple et plus rapide pour la BDD)
        # On filtre seulement les paramètres > 0 pour éviter la division par zéro
        query = f"""
            SELECT team_name, val_accuracy, test_accuracy, n_params, submitted_at
            FROM {self.table_name}
            WHERE n_params > 0
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return df

        # 2. Calcul de l'efficacité avec Numpy (Plus robuste que le SQL)
        # Formule : Accuracy / Log10(Params)
        # Cela évite qu'un modèle à 10 paramètres avec 5% d'acc batte un modèle à 1M params et 90% d'acc.
        df['efficiency_score'] = df['test_accuracy'] / np.log10(df['n_params'])

        # 3. Filtrage : On ne garde que la meilleure soumission par équipe (selon l'efficacité)
        df = df.sort_values('efficiency_score', ascending=False)
        df = df.drop_duplicates(subset=['team_name'], keep='first')

        # 4. Mise en forme pour l'affichage (Cosmétique uniquement)
        # On crée un nouveau DataFrame propre pour l'affichage final
        display_df = pd.DataFrame()
        display_df['Rang'] = range(1, len(df) + 1)
        display_df['Equipe'] = df['team_name']
        display_df['Val Acc (%)'] = (df['val_accuracy'] * 100).round(2)
        display_df['Test Secret (%)'] = (df['test_accuracy'] * 100).round(2)
        display_df['Gap (%)'] = ((df['val_accuracy'] - df['test_accuracy']) * 100).round(2)

        # Formatage des paramètres (ex: 1.2M ou entier)
        display_df['Params'] = df['n_params']

        # Score d'efficacité (multiplié par 100 pour lisibilité, mais basé sur le log)
        display_df['Efficacité'] = (df['efficiency_score'] * 100).round(4)

        display_df['Soumis le'] = df['submitted_at']

        return display_df

    def get_stats(self) -> dict:
        """Récupère les statistiques globales."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f'SELECT COUNT(DISTINCT team_name) FROM {self.table_name}')
        n_teams = cursor.fetchone()[0]

        cursor.execute(f'SELECT COUNT(*) FROM {self.table_name}')
        n_submissions = cursor.fetchone()[0]

        cursor.execute(f'SELECT MAX(test_accuracy) FROM {self.table_name}')
        best_score = cursor.fetchone()[0]

        cursor.execute(f'SELECT AVG(test_accuracy) FROM {self.table_name} WHERE is_best = 1')
        avg_score = cursor.fetchone()[0]

        conn.close()

        return {
            'n_teams':       n_teams or 0,
            'n_submissions': n_submissions or 0,
            'best_score':    best_score or 0,
            'avg_score':     avg_score or 0
            }


# =============================================================================
# Évaluation de modèle (abstrait)
# =============================================================================

class ModelEvaluator(ABC):
    """Classe abstraite pour l'évaluation de modèles."""

    @abstractmethod
    def evaluate(self, model: nn.Module, data_path: str) -> dict:
        """
        Évalue un modèle sur un dataset.

        Returns:
            dict avec 'accuracy', 'f1', 'precision', 'recall'
        """
        pass


def compute_metrics(
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray]
        ) -> dict:
    """Calcule les métriques de classification binaire."""
    predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    accuracy = (predictions == labels).mean()

    # F1 score
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy':  float(accuracy),
        'f1':        float(f1),
        'precision': float(precision),
        'recall':    float(recall)
        }


# =============================================================================
# Application Gradio
# =============================================================================

class LeaderboardApp:
    """Application Gradio pour un leaderboard."""

    def __init__(self, config: LeaderboardConfig, evaluator: ModelEvaluator):
        self.config = config
        self.evaluator = evaluator
        self.db = LeaderboardDB(config.db_path, config.table_name)

    def _get_both_leaderboards(self) -> tuple:
        """Retourne les deux leaderboards."""
        return self.db.get_leaderboard(), self.db.get_leaderboard_efficient()

    def process_submission(self, team_name: str, model_file) -> tuple:
        """Traite une soumission : charge, évalue, sauvegarde."""
        if not team_name or not team_name.strip():
            return "Erreur: Veuillez entrer un nom d'équipe.", *self._get_both_leaderboards()

        team_name = team_name.strip()

        if model_file is None:
            return "Erreur: Veuillez uploader un fichier modèle (.pt)", *self._get_both_leaderboards()

        if not self.config.test_secret_path.exists():
            return "Erreur: Dataset test secret non trouvé. Contactez l'enseignant.", *self._get_both_leaderboards()

        if not self.config.val_path.exists():
            return "Erreur: Dataset validation non trouvé.", *self._get_both_leaderboards()

        model_path = model_file.name if hasattr(model_file, 'name') else model_file

        if model_path.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)

                pt_files = list(Path(tmp_dir).rglob('*.pt'))
                if not pt_files:
                    return "Erreur: Aucun fichier .pt trouvé dans le ZIP.", *self._get_both_leaderboards()

                model_path = str(pt_files[0])
                return self._evaluate_and_save(team_name, model_path)
        else:
            return self._evaluate_and_save(team_name, model_path)

    def _evaluate_and_save(self, team_name: str, model_path: str) -> tuple:
        """Évalue et sauvegarde le modèle."""
        model, config = load_model_from_file(model_path)

        if model is None:
            return f"Erreur: {config}", *self._get_both_leaderboards()

        n_params = count_parameters(model)
        model_hash = compute_model_hash(model_path)

        # Évaluer sur validation
        val_results = self.evaluator.evaluate(model, str(self.config.val_path))

        # Évaluer sur test secret
        test_results = self.evaluator.evaluate(model, str(self.config.test_secret_path))

        # Sauvegarder
        submission_id = self.db.save_submission(
                team_name=team_name,
                model_hash=model_hash,
                val_acc=val_results['accuracy'],
                test_acc=test_results['accuracy'],
                test_f1=test_results['f1'],
                n_params=n_params,
                architecture=config
                )

        # Construire le message de résultat
        gap = val_results['accuracy'] - test_results['accuracy']

        val_acc_str = f"{val_results['accuracy']:.2%}"
        test_acc_str = f"{test_results['accuracy']:.2%}"
        gap_str = f"{gap:+.2%}"
        f1_str = f"{test_results['f1']:.2%}"

        if gap > 0.10:
            badge = "**ATTENTION** : Gros écart ! Votre modèle sur-apprend."
            color = "🔴"
        elif gap > 0.05:
            badge = "**Modéré** : Pensez à plus de régularisation."
            color = "🟡"
        else:
            badge = "**Excellent** : Votre modèle généralise bien."
            color = "🟢"

        message = f"""
**Résultats de votre soumission - ID: {submission_id}**

<div align="center">
<strong>{color} {badge}</strong>
</div>

### Détails de l'équipe
- **Nom:** `{team_name}`
- **Hash du modèle:** `{model_hash}`
- **Paramètres:** `{n_params:,}`

### Scores
| Métrique | Valeur |
| :--- | :--- |
| **Accuracy Validation** | `{val_acc_str}` |
| **Accuracy Test Secret**| `{test_acc_str}` |
| **Gap (Val - Test)** | `{gap_str}` |
| **F1-Score Test** | `{f1_str}` |
        """

        del model
        return message, *self._get_both_leaderboards()

    def get_stats_text(self) -> str:
        """Retourne les stats formatées."""
        stats = self.db.get_stats()
        return f"""
**Statistiques du Tournoi {self.config.name}**
- Équipes participantes: {stats['n_teams']}
- Total soumissions: {stats['n_submissions']}
- Meilleur score: {stats['best_score']:.2%}
- Score moyen: {stats['avg_score']:.2%}
"""

    def create_app(self) -> gr.Blocks:
        """Crée l'application Gradio."""
        self.db.init_database()

        with gr.Blocks(
                title=self.config.title,
                theme=gr.themes.Soft()
                ) as app:
            gr.Markdown(
                f"""
# {self.config.title}
### {self.config.description}
            """
                )

            with gr.Tabs():
                # Tab 1: Soumission
                with gr.TabItem("Soumettre"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            team_input = gr.Textbox(
                                    label="Nom de l'équipe",
                                    placeholder="Ex: Les Dragons de PyTorch",
                                    max_lines=1
                                    )
                            model_input = gr.File(
                                    label="Modèle (.pt ou .zip)",
                                    file_types=[".pt", ".zip"],
                                    type="filepath"
                                    )
                            submit_btn = gr.Button(
                                    "Soumettre",
                                    variant="primary",
                                    size="lg"
                                    )

                        with gr.Column(scale=2):
                            result_output = gr.Markdown(label="Résultat")

                # Tab 2: Leaderboard
                with gr.TabItem("Leaderboard"):
                    gr.Markdown("### Classement en temps réel")

                    stats_display = gr.Markdown(self.get_stats_text())

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Meilleure Accuracy")
                            leaderboard_table = gr.Dataframe(
                                    value=self.db.get_leaderboard(),
                                    label="Classement par accuracy (meilleur score par équipe)",
                                    interactive=False,
                                    wrap=True
                                    )

                        with gr.Column():
                            gr.Markdown("#### Meilleur Ratio Accuracy/Taille")
                            leaderboard_efficient_table = gr.Dataframe(
                                    value=self.db.get_leaderboard_efficient(),
                                    label="Classement par efficacite (accuracy / params)",
                                    interactive=False,
                                    wrap=True
                                    )

                    refresh_btn = gr.Button("Rafraîchir", size="sm")

                # Tab 3: Règles
                with gr.TabItem("Règles"):
                    gr.Markdown(self.config.rules_markdown or self._default_rules())

            # Events
            submit_btn.click(
                    fn=self.process_submission,
                    inputs=[team_input, model_input],
                    outputs=[result_output, leaderboard_table, leaderboard_efficient_table]
                    )

            refresh_btn.click(
                    fn=lambda: (self.db.get_leaderboard(), self.db.get_leaderboard_efficient(), self.get_stats_text()),
                    outputs=[leaderboard_table, leaderboard_efficient_table, stats_display]
                    )

        return app

    def _default_rules(self) -> str:
        return f"""
## Règles du Tournoi {self.config.name}

### Comment participer

1. **Entraînez** votre modèle
2. **Soumettez** le fichier .pt sur cette interface

### Scoring

- Votre modèle est évalué sur un **dataset secret**
- Le classement est basé sur l'**accuracy** du test secret
- Seul votre **meilleur score** compte

### Conseils

- Le dataset de test peut avoir une distribution **différente** !
- Les modèles sur-appris seront pénalisés
- Pensez à: Dropout, Weight Decay, Early Stopping

### Anti-triche

- Chaque soumission est hashée
- L'historique complet est conservé
- Le dataset secret n'est jamais révélé

---
*Bonne chance !*
        """

    def launch(self, share: bool = False):
        """Lance l'application."""
        print("=" * 60)
        print(f"Tournoi {self.config.name} - Serveur Leaderboard")
        print("=" * 60)

        if not self.config.test_secret_path.exists():
            print(f"[!] ATTENTION: {self.config.test_secret_path} non trouvé!")
        else:
            print(f"[OK] Dataset test secret: {self.config.test_secret_path}")

        if not self.config.val_path.exists():
            print(f"[!] ATTENTION: {self.config.val_path} non trouvé!")
        else:
            print(f"[OK] Dataset validation: {self.config.val_path}")

        print(f"[OK] Base de données: {self.config.db_path}")
        print("=" * 60)

        app = self.create_app()
        app.launch(
                server_port=self.config.port,
                share=True,
                show_error=True
                )
