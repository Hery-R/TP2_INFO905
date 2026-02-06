"""
Application Gradio : Tournoi Oracle - Leaderboard

Interface web pour le dataset Oracle (aventuriers avec features tabulaires).

Usage:
    python app_leaderboard_oracle.py
    # Ouvre http://localhost:7860
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from leaderboard_base import (
    LeaderboardApp,
    LeaderboardConfig,
    ModelEvaluator,
    compute_metrics
)
from train_oracle import AdventurerDataset


# =============================================================================
# Évaluateur spécifique Oracle
# =============================================================================

class OracleEvaluator(ModelEvaluator):
    """Évaluateur pour le dataset Oracle (features tabulaires)."""

    def __init__(self, input_dim: int = 8, batch_size: int = 64):
        self.input_dim = input_dim
        self.batch_size = batch_size

    def evaluate(self, model: nn.Module, data_path: str) -> dict:
        """Évalue un modèle sur le dataset Oracle."""
        dataset = AdventurerDataset(data_path, normalize=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_predictions = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for features, labels in dataloader:
                logits = model(features).squeeze()
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).int()

                all_predictions.append(predictions)
                all_labels.append(labels)

        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()

        return compute_metrics(all_predictions, all_labels)


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent

CONFIG = LeaderboardConfig(
    name="Oracle",
    title="Tournoi de la Guilde des Aventuriers - Oracle",
    description="Soumettez votre modèle Oracle et grimpez dans le classement !",
    db_path=BASE_DIR / "leaderboard_oracle.db",
    test_secret_path=BASE_DIR / "solution" / "test_secret.csv",
    val_path=BASE_DIR / "data" / "val.csv",
    table_name="submissions_oracle",
    port=7860,
    rules_markdown="""
## Règles du Tournoi Oracle

### Dataset

Le dataset contient des aventuriers avec 8 caractéristiques :
- `force`, `intelligence`, `agilite`, `chance`
- `experience`, `niveau_quete`, `equipement`, `fatigue`

La cible est `survie` (0 = échec, 1 = survie).

### Comment participer

1. **Entraînez** votre modèle avec `train_oracle.py`
2. **Sauvegardez** avec `torch.save(model, 'model.pt')`
3. **Upload** le fichier .pt sur cette interface

### Scoring

- Votre modèle est évalué sur un **dataset secret** (Terres Maudites)
- Le classement est basé sur l'**accuracy** du test secret
- Seul votre **meilleur score** compte

### Conseils

- Le dataset de test a une distribution **différente** !
- Les modèles sur-appris seront pénalisés
- Pensez à: Dropout, Weight Decay, Early Stopping
- La normalisation des données est activée automatiquement

### Anti-triche

- Chaque soumission est hashée
- L'historique complet est conservé
- Le dataset secret n'est jamais révélé

---
*Que la chance soit avec vous, jeune Oracle !*
    """
)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    evaluator = OracleEvaluator(input_dim=8)
    app = LeaderboardApp(CONFIG, evaluator)
    app.launch(share=False)
