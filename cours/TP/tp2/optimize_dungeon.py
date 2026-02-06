import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from baseline_model import DungeonOracle, count_parameters
from train_dungeon_logs import DungeonLogDataset, train_epoch, evaluate
from pathlib import Path
from itertools import product
import pandas as pd

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(__file__).parent / "data"
    vocab_path = data_dir / "vocabulary_dungeon.json"
    train_path = data_dir / "train_dungeon.csv"
    val_path = data_dir / "val_dungeon.csv"

    # Load Data
    train_dataset = DungeonLogDataset(str(train_path), str(vocab_path))
    val_dataset = DungeonLogDataset(str(val_path), str(vocab_path))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Search Space
    configs = [
        # Efficient small models
        {'embed': 8, 'hidden': 16, 'layers': 1, 'mode': 'gru', 'bidir': True},
        {'embed': 16, 'hidden': 32, 'layers': 1, 'mode': 'gru', 'bidir': True},
        {'embed': 32, 'hidden': 32, 'layers': 1, 'mode': 'gru', 'bidir': True},
        
        # Slightly larger but potentially more accurate
        {'embed': 16, 'hidden': 64, 'layers': 1, 'mode': 'lstm', 'bidir': True},
        {'embed': 32, 'hidden': 64, 'layers': 2, 'mode': 'lstm', 'bidir': True},
    ]

    results = []

    print(f"{'Config':<50} | {'Params':<10} | {'Val Acc':<10}")
    print("-" * 80)

    for cfg in configs:
        model = DungeonOracle(
            vocab_size=train_dataset.vocab_size,
            embed_dim=cfg['embed'],
            hidden_dim=cfg['hidden'],
            num_layers=cfg['layers'],
            mode=cfg['mode'],
            bidirectional=cfg['bidir'],
            dropout=0.2 if cfg['layers'] > 1 else 0.0,
            padding_idx=train_dataset.pad_idx
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.BCEWithLogitsLoss()

        best_acc = 0.0
        
        # Quick train (5 epochs)
        for epoch in range(5):
            train_epoch(model, train_loader, criterion, optimizer, device)
            _, val_acc = evaluate(model, val_loader, criterion, device)
            best_acc = max(best_acc, val_acc)

        params = count_parameters(model)
        print(f"{str(cfg):<50} | {params:<10} | {best_acc:.2%}")
        
        results.append({
            'config': cfg,
            'params': params,
            'accuracy': best_acc,
            'score': best_acc / (params ** 0.1) # Custom metric: accuracy penalized slightly by size
        })

    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\nTop 3 Configuration (Accuracy):")
    for r in results[:3]:
        print(f"Acc: {r['accuracy']:.2%} | Params: {r['params']} | {r['config']}")

    # Sort by 'efficiency'
    results.sort(key=lambda x: x['score'], reverse=True)
    print("\nTop 3 Configuration (Efficiency):")
    for r in results[:3]:
        print(f"Acc: {r['accuracy']:.2%} | Params: {r['params']} | {r['config']}")

if __name__ == "__main__":
    main()
