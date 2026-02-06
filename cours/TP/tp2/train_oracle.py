"""
Script d'entraînement : Oracle de la Guilde

Ce script entraîne le modèle sur le dataset des aventuriers.
Il contient volontairement des pratiques non optimales à améliorer.

Problèmes à corriger :
1. Pas de normalisation des données
2. Pas de shuffling (mélange) des données
3. Pas d'early stopping
4. Pas de weight decay dans l'optimiseur
5. Learning rate fixe (pas de scheduler)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from baseline_model import GuildOracle, count_parameters


# ============================================================================
# Dataset PyTorch
# ============================================================================

class AdventurerDataset(Dataset):
    """Dataset des aventuriers de la Guilde."""

    def __init__(self, csv_path: str, normalize: bool = False):
        """
        Args:
            csv_path: Chemin vers le fichier CSV
            normalize: Si True, normalise les features (recommandé mais désactivé par défaut)
        """
        self.df = pd.read_csv(csv_path)

        # Séparer features et labels
        self.labels = torch.tensor(self.df['survie'].values, dtype=torch.float32)
        self.features = self.df.drop('survie', axis=1).values

        # Normalisation des data
        if normalize:
            self.mean = self.features.mean(axis=0)
            self.std = self.features.std(axis=0) + 1e-8
            self.features = (self.features - self.mean) / self.std

        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# Boucle d'entraînement
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features).squeeze()

        # Loss et backward
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistiques
        total_loss += loss.item() * len(labels)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Évalue le modèle."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


# ============================================================================
# Fonction principale
# ============================================================================

def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Chemins
    data_dir = Path(__file__).parent / "data"
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Charger les données
    print("\nChargement des données...")
    train_dataset = AdventurerDataset(
            args.train_file if args.train_file else str(data_dir / "train.csv"),
            normalize=args.normalize
            )
    val_dataset = AdventurerDataset(
            str(data_dir / "val.csv"),
            normalize=args.normalize
            )

    # DataLoaders
    train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle
            )
    val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
            )

    print(f"Train: {len(train_dataset)} échantillons")
    print(f"Val: {len(val_dataset)} échantillons")

    # Modèle
    print("\nCréation du modèle...")
    model = GuildOracle(
            input_dim=train_dataset.features.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
            )
    model = model.to(device)
    print(f"Paramètres: {count_parameters(model):,}")

    # Loss et optimiseur
    criterion = nn.BCEWithLogitsLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
                )
    else:
        optimizer = optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=args.weight_decay
                )

    print(f"Optimiseur: {args.optimizer.upper()}, LR: {args.learning_rate}")

    # Historique
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc': []
        }

    # Entraînement
    print("\n" + "=" * 50)
    print("Début de l'entraînement")
    print("=" * 50)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
                )

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Affichage
        print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
                )

        # Sauvegarder le meilleur modèle (modele complet pour supporter architectures custom)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, checkpoint_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping (désactivé par défaut)
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\nEarly stopping après {epoch + 1} epochs")
            break

    print("\n" + "=" * 50)
    print(f"Meilleure accuracy validation: {best_val_acc:.2%}")
    print(f"Modèle sauvegardé: {checkpoint_dir / 'best_model.pt'}")
    print("=" * 50)

    with open(checkpoint_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=4)

    # Analyse de l'overfitting
    gap = history['train_acc'][-1] - history['val_acc'][-1]
    print(f"\nGap Train-Val (dernière epoch): {gap:.2%}")
    if gap > 0.10:
        print("ATTENTION: Gap important ! Risque d'overfitting.")
        print("Suggestions:")
        print("  - Ajouter Dropout")
        print("  - Augmenter weight_decay")
        print("  - Réduire la complexité du modèle")
        print("  - Utiliser early stopping")

    # Plot
    if args.plot:
        plot_history(history, checkpoint_dir / "training_curves.png")


def plot_history(history, save_path):
    """Affiche les courbes d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss au cours de l\'entraînement')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy au cours de l\'entraînement')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nCourbes sauvegardées: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement de l'Oracle de la Guilde")

    # Données
    parser.add_argument(
            '--normalize', action='store_true', default=False,
            help='Normaliser les features'
            )
    parser.add_argument(
            '--shuffle', action='store_true', default=False,
            help='Mélanger les données'
            )
    parser.add_argument(
            '--train_file', type=str, default=None,
            help='Fichier CSV de training (defaut: data/train.csv)'
            )

    # Modèle
    parser.add_argument(
            '--hidden_dim', type=int, default=256,
            help='Dimension des couches cachées'
            )
    parser.add_argument(
            '--num_layers', type=int, default=5,
            help="Nombre de couches cachées (0 = linéaire)"
            )
    parser.add_argument(
            '--dropout', type=float, default=0.3,
            help="Taux de dropout"
            )

    # Entraînement
    parser.add_argument(
            '--epochs', type=int, default=5,
            help='Nombre d\'epochs'
            )
    parser.add_argument(
            '--batch_size', type=int, default=32,
            help='Taille du batch'
            )
    parser.add_argument(
            '--learning_rate', type=float, default=0.1,
            help='Learning rate'
            )
    parser.add_argument(
            '--optimizer', type=str, default='adam',
            choices=['adam', 'sgd'],
            help='Optimiseur'
            )
    parser.add_argument(
            '--weight_decay', type=float, default=0.0,
            help='Weight decay (L2 regularization)'
            )

    # Early stopping
    parser.add_argument(
            '--early_stopping', action='store_true', default=False,
            help='Activer early stopping'
            )
    parser.add_argument(
            '--patience', type=int, default=10,
            help='Patience pour early stopping'
            )

    # Autres
    parser.add_argument(
            '--plot', action='store_true', default=True,
            help='Afficher les courbes'
            )

    args = parser.parse_args()

    # Afficher la configuration
    print("Configuration:")
    print("-" * 40)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 40)

    main(args)
