"""
Script d'entraînement avancé : Oracle du Donjon
Inclut : Data Augmentation + K-Fold CV + Grid Search
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
from baseline_model import DungeonOracle, count_parameters

# ============================================================================
# Data Augmentation
# ============================================================================

def augment_sequences_df(df: pd.DataFrame, n_variants: int = 1) -> pd.DataFrame:
    """
    Crée des versions augmentées des séquences.
    - Insertion de tokens neutres (SalleVide, Couloir, Or, Coffre)
    - Petites permutations de tokens neutres
    """
    import random
    neutral_tokens = ["SalleVide", "Couloir", "Or", "Coffre", "Inscription", "Escalier"]
    
    augmented_rows = []
    
    for _, row in df.iterrows():
        base_seq = row["sequence"]
        events = [e.strip() for e in base_seq.split(" -> ")]
        
        for _ in range(n_variants):
            toks = events.copy()
            
            # 1) Insertion de 1 à 3 tokens neutres aléatoires
            k_insert = random.randint(1, min(3, max(1, len(toks) // 15)))
            for _ in range(k_insert):
                if len(toks) > 2:
                    pos = random.randint(1, len(toks) - 1)
                    toks.insert(pos, random.choice(neutral_tokens))
            
            # 2) Permutation de 2 tokens neutres si possible
            neutral_idxs = [i for i, t in enumerate(toks) if t in neutral_tokens]
            if len(neutral_idxs) >= 2:
                i, j = random.sample(neutral_idxs, 2)
                toks[i], toks[j] = toks[j], toks[i]
            
            new_seq = " -> ".join(toks)
            
            augmented_rows.append({
                "id": -1,
                "sequence": new_seq,
                "length": len(toks),
                "survived": row["survived"],
                "category": row["category"],
            })
    
    if not augmented_rows:
        return df
    
    df_aug = pd.DataFrame(augmented_rows)
    df_big = pd.concat([df, df_aug], ignore_index=True)
    return df_big

# ============================================================================
# Dataset PyTorch
# ============================================================================

class DungeonLogDataset(Dataset):
    """Dataset des journaux de donjon (séquences d'événements)."""
    
    def __init__(self, csv_path: str, vocab_path: str, max_length: int = 140):
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.pad_idx = self.vocab.get("<PAD>", 0)
        self.unk_idx = self.vocab.get("<UNK>", 1)
        
        self.sequences = []
        self.labels = []
        self.lengths = []
        
        for _, row in self.df.iterrows():
            events = [e.strip() for e in row['sequence'].split(' -> ')]
            token_ids = [self.vocab.get(e, self.unk_idx) for e in events]
            
            if max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            self.sequences.append(torch.tensor(token_ids, dtype=torch.long))
            self.labels.append(row['survived'])
            self.lengths.append(len(token_ids))
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.lengths = torch.tensor(self.lengths, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        length = self.lengths[idx]
        
        if self.max_length is not None:
            padded_sequence = torch.full((self.max_length,), self.pad_idx, dtype=torch.long)
            seq_len = min(len(sequence), self.max_length)
            padded_sequence[:seq_len] = sequence[:seq_len]
            sequence = padded_sequence
        
        return sequence, self.labels[idx], length
    
    @property
    def vocab_size(self):
        return len(self.vocab)

# ============================================================================
# Training functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels, lengths in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences, lengths).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)
    
    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            outputs = model(sequences, lengths).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    return total_loss / total, correct / total

# ============================================================================
# K-Fold Cross-Validation avec Grid Search
# ============================================================================

def train_kfold_grid_search(dataset, vocab_size, pad_idx, device, args, hyperparams_grid):
    """
    Entraîne avec K-Fold CV et Grid Search.
    Retourne le meilleur modèle et ses hyperparamètres.
    """
    kfold = KFold(n_splits=args.kfolds, shuffle=True, random_state=42)
    criterion = nn.BCEWithLogitsLoss()
    
    best_global_acc = 0
    best_global_config = None
    best_global_model = None
    
    print("\n" + "=" * 80)
    print(f"GRID SEARCH avec {args.kfolds}-Fold Cross-Validation")
    print("=" * 80)
    print(f"Total configurations: {len(hyperparams_grid)}")
    print(f"Total entraînements: {len(hyperparams_grid) * args.kfolds}")
    print("=" * 80 + "\n")
    
    for config_idx, config in enumerate(hyperparams_grid, 1):
        print(f"\n[Config {config_idx}/{len(hyperparams_grid)}] Testing: {config}")
        
        fold_accs = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset))), 1):
            # Créer les samplers pour ce fold
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
            
            # Créer le modèle pour ce fold
            model = DungeonOracle(
                vocab_size=vocab_size,
                embed_dim=config['embed_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                mode=config['mode'],
                max_length=140,
                bidirectional=config['bidirectional'],
                padding_idx=pad_idx,
            ).to(device)
            
            # Optimizer
            if args.optimizer == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            else:
                optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
            
            # Entraînement pour ce fold
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(args.epochs_per_fold):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping par fold
                if patience_counter >= args.patience:
                    break
            
            fold_accs.append(best_val_acc)
            print(f"  Fold {fold}/{args.kfolds}: Val Acc = {best_val_acc:.4f}")
        
        # Moyenne sur les folds
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        print(f"  → Mean CV Acc: {mean_acc:.4f} (±{std_acc:.4f})")
        
        # Mettre à jour le meilleur modèle global
        if mean_acc > best_global_acc:
            best_global_acc = mean_acc
            best_global_config = config
            
            # Réentraîner sur tout le dataset avec la meilleure config
            print(f"  ★ New best config! Retraining on full dataset...")
            full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            
            best_global_model = DungeonOracle(
                vocab_size=vocab_size,
                embed_dim=config['embed_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                mode=config['mode'],
                max_length=140,
                bidirectional=config['bidirectional'],
                padding_idx=pad_idx,
            ).to(device)
            
            if args.optimizer == 'adam':
                optimizer = optim.Adam(best_global_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            else:
                optimizer = optim.SGD(best_global_model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
            
            for epoch in range(args.epochs_per_fold):
                train_loss, train_acc = train_epoch(best_global_model, full_loader, criterion, optimizer, device)
    
    print("\n" + "=" * 80)
    print("MEILLEURE CONFIGURATION TROUVÉE:")
    print("=" * 80)
    print(f"Config: {best_global_config}")
    print(f"Mean CV Accuracy: {best_global_acc:.4f}")
    print(f"Paramètres: {count_parameters(best_global_model):,}")
    print("=" * 80 + "\n")
    
    return best_global_model, best_global_config, best_global_acc

# ============================================================================
# Main function
# ============================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Chemins
    data_dir = Path(__file__).parent / "data"
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    vocab_path = data_dir / "vocabulary_dungeon.json"
    train_path = data_dir / "train_dungeon.csv"
    val_path = data_dir / "val_dungeon.csv"
    
    # Charger et concaténer train + val
    print("\n" + "=" * 80)
    print("CHARGEMENT ET PRÉPARATION DES DONNÉES")
    print("=" * 80)
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    print(f"Train original: {len(df_train)} séquences")
    print(f"Val original:   {len(df_val)} séquences")
    
    # Concat train + val
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    print(f"Total concat:   {len(df_all)} séquences")
    
    # Augmentation si demandée
    if args.augment:
        print(f"\nApplication de l'augmentation (factor={args.augment_factor})...")
        df_all = augment_sequences_df(df_all, n_variants=args.augment_factor)
        print(f"Après augmentation: {len(df_all)} séquences")
    
    # Sauvegarder temporairement
    tmp_path = data_dir / "train_dungeon_full_augmented.csv"
    df_all.to_csv(tmp_path, index=False)
    
    # Créer le dataset
    print("\nCréation du dataset PyTorch...")
    dataset = DungeonLogDataset(str(tmp_path), str(vocab_path))
    
    print(f"Dataset final: {len(dataset)} séquences")
    print(f"Vocabulaire: {dataset.vocab_size} tokens")
    print("=" * 80 + "\n")
    
    # Grille d'hyperparamètres
    hyperparams_grid = []
    
    for embed_dim in args.embed_dims:
        for hidden_dim in args.hidden_dims:
            for num_layers in args.num_layers_list:
                for dropout in args.dropouts:
                    for mode in args.modes:
                        for lr in args.learning_rates:
                            hyperparams_grid.append({
                                'embed_dim': embed_dim,
                                'hidden_dim': hidden_dim,
                                'num_layers': num_layers,
                                'dropout': dropout,
                                'mode': mode,
                                'bidirectional': args.bidirectional,
                                'lr': lr,
                                'weight_decay': args.weight_decay,
                            })
    
    # K-Fold + Grid Search
    best_model, best_config, best_acc = train_kfold_grid_search(
        dataset, 
        dataset.vocab_size, 
        dataset.pad_idx, 
        device, 
        args, 
        hyperparams_grid
    )
    
    # Sauvegarder le meilleur modèle
    model_path = checkpoint_dir / "best_dungeon_model_kfold.pt"
    config_path = checkpoint_dir / "best_config_kfold.json"
    
    torch.save(best_model, model_path)
    
    with open(config_path, 'w') as f:
        json.dump({
            'config': best_config,
            'mean_cv_acc': float(best_acc),
            'num_params': count_parameters(best_model),
        }, f, indent=4)
    
    print(f"✓ Meilleur modèle sauvegardé: {model_path}")
    print(f"✓ Configuration sauvegardée: {config_path}")

# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement avancé avec K-Fold CV + Grid Search")
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true', default=True, help='Activer data augmentation')
    parser.add_argument('--augment_factor', type=int, default=1, help='Nombre de variantes par séquence')
    
    # K-Fold
    parser.add_argument('--kfolds', type=int, default=3, help='Nombre de folds')
    parser.add_argument('--epochs_per_fold', type=int, default=20, help='Epochs par fold')
    parser.add_argument('--patience', type=int, default=5, help='Patience pour early stopping')
    
    # Grid search - Intervalles d'hyperparamètres
    parser.add_argument('--embed_dims', nargs='+', type=int, default=[8, 16], help='Embed dimensions à tester')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[16, 32], help='Hidden dimensions à tester')
    parser.add_argument('--num_layers_list', nargs='+', type=int, default=[1, 2], help='Num layers à tester')
    parser.add_argument('--dropouts', nargs='+', type=float, default=[0.2, 0.3, 0.4], help='Dropouts à tester')
    parser.add_argument('--modes', nargs='+', type=str, default=['gru', 'lstm'], help='Modes à tester')
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[0.001, 0.005], help='Learning rates à tester')
    
    # Autres hyperparamètres fixes
    parser.add_argument('--bidirectional', action='store_true', default=True, help='Bidirectionnel')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CONFIGURATION D'ENTRAÎNEMENT AVANCÉ")
    print("=" * 80)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80 + "\n")
    
    main(args)
