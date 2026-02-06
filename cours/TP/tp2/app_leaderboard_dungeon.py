"""
Application Gradio : Tournoi Dungeon - Leaderboard

Interface web pour le dataset Dungeon (séquences d'événements).

Usage:
    python app_leaderboard_dungeon.py
    # Ouvre http://localhost:7861
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_dungeon_logs import DungeonLogDataset

from leaderboard_base import (
    LeaderboardApp,
    LeaderboardConfig,
    ModelEvaluator,
    compute_metrics
)

# =============================================================================
# Évaluateur spécifique Dungeon
# =============================================================================

class DungeonEvaluator(ModelEvaluator):
    """Évaluateur pour le dataset Dungeon (séquences)."""

    def __init__(self, vocab_path: str, batch_size: int = 64):
        self.vocab_path = vocab_path
        self.batch_size = batch_size

        # Charger le vocabulaire pour connaître sa taille
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)

    def evaluate(self, model: nn.Module, data_path: str) -> dict:
        """Évalue un modèle sur le dataset Dungeon."""
        dataset = DungeonLogDataset(
            data_path,
            vocab_path=self.vocab_path
        )
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False)

        all_predictions = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                sequences, labels, lengths = batch

                # Le modèle peut accepter (sequences) ou (sequences, lengths)
                try:
                    logits = model(sequences, lengths).squeeze()
                except TypeError:
                    # Si le modèle n'accepte pas lengths
                    logits = model(sequences).squeeze()

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
    name="Dungeon",
    title="Les Archives Interdites - Tournoi Dungeon",
    description="Prédisez la survie des aventuriers dans les donjons !",
    db_path=BASE_DIR / "leaderboard_dungeon.db",
    test_secret_path=BASE_DIR / "solution" / "test_dungeon.csv",
    val_path=BASE_DIR / "data" / "val_dungeon.csv",
    table_name="submissions_dungeon",
    port=7861,
    rules_markdown="""
## Règles du Tournoi Dungeon

### Dataset

Le dataset contient des séquences d'événements de donjon :
- **Entrée** : Séquence de salles/événements (ex: "Entree -> Rat -> Potion -> Dragon -> Sortie")
- **Sortie** : Survie (0 = mort, 1 = survie)

### Vocabulaire

44 tokens différents répartis en catégories :
- **Monstres** : Rat, Gobelin, Orc, Troll, Dragon...
- **Soins** : Potion, Feu_de_Camp, Fontaine_Sacree...
- **Pièges** : Piege_a_Pics, Fleches_Empoisonnees, Fosse...
- **Trésors** : Coffre, Gemmes, Or, Relique
- **Spéciaux** : Amulette_Protection, Armure_Ancienne, Epee_Legendaire

### Pièges pédagogiques

1. **ORDRE** : "Potion -> Dragon" survit, "Dragon -> Potion" meurt
2. **LONG-TERME** : L'Amulette au début protège contre le Boss final
3. **SÉMANTIQUE** : Les monstres/soins ont des effets similaires intra-groupe

### Comment participer

1. **Entraînez** votre modèle (RNN, LSTM, Transformer...)
2. **Sauvegardez** avec `torch.save(model, 'model.pt')`
3. **Upload** le fichier .pt sur cette interface

### Scoring

- Votre modèle est évalué sur un **dataset secret**
- Le classement est basé sur l'**accuracy** du test secret
- Seul votre **meilleur score** compte

---
*Survivrez-vous aux Archives Interdites ?*
    """
)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    vocab_path = BASE_DIR / "data" / "vocabulary_dungeon.json"

    if not vocab_path.exists():
        print(f"[!] Vocabulaire non trouvé: {vocab_path}")
        print("    Exécutez d'abord: python solution/dungeon_logs.py")
        exit(1)

    evaluator = DungeonEvaluator(
        vocab_path=str(vocab_path)
    )
    app = LeaderboardApp(CONFIG, evaluator)
    app.launch(share=False)
