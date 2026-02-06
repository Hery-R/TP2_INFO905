# -*- coding: utf-8 -*-
"""
Optimiseur de modele pour les Terres Maudites.
Teste systematiquement differentes configs et trouve la meilleure.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from baseline_model import GuildOracle, count_parameters

def prepare_data(df, normalize=True, ref_mean=None, ref_std=None):
    labels = df['survie'].values
    features = df.drop('survie', axis=1).values.astype(np.float32)
    
    if normalize:
        if ref_mean is not None:
            mean, std = ref_mean, ref_std
        else:
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
        features = (features - mean) / std
    else:
        mean, std = None, None
    
    return torch.tensor(features), torch.tensor(labels, dtype=torch.float32), mean, std


def train_and_evaluate(config, X_train, y_train, X_val, y_val, X_tm, y_tm, epochs=50):
    model_config = {
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout']
    }
    model = GuildOracle(input_dim=8, **model_config)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('lr', 0.01), 
        weight_decay=config.get('weight_decay', 0.01)
    )
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            val_preds = (torch.sigmoid(model(X_val).squeeze()) > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Restaurer le meilleur modele
    model.load_state_dict(best_model_state)
    
    # Evaluer sur TM
    model.eval()
    with torch.no_grad():
        tm_preds = (torch.sigmoid(model(X_tm).squeeze()) > 0.5).float()
        tm_acc = (tm_preds == y_tm).float().mean().item()
    
    return best_val_acc, tm_acc, model


def main():
    data_dir = Path(__file__).parent / "data"
    
    # Charger les donnees
    train_df = pd.read_csv(data_dir / "train_augmented_100.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    tm_df = pd.read_csv(data_dir / "synthetic_terres_maudites.csv")
    
    # Preparer (normaliser avec stats du train)
    X_train, y_train, train_mean, train_std = prepare_data(train_df, normalize=True)
    X_val, y_val, _, _ = prepare_data(val_df, normalize=True, ref_mean=train_mean, ref_std=train_std)
    
    # Pour TM, on normalise avec SES PROPRES stats (comme le test reel!)
    X_tm, y_tm, _, _ = prepare_data(tm_df, normalize=True)
    
    # Grille de recherche
    hidden_dims = [4, 6, 8, 10, 12, 16]
    num_layers_list = [1, 2]
    dropouts = [0.0, 0.2, 0.4]
    weight_decays = [0.0, 0.01, 0.1]
    lrs = [0.01, 0.05]
    
    results = []
    
    print("=" * 70)
    print("RECHERCHE DE LA MEILLEURE CONFIGURATION")
    print("=" * 70)
    
    total = len(hidden_dims) * len(num_layers_list) * len(dropouts) * len(weight_decays) * len(lrs)
    i = 0
    
    for hd, nl, do, wd, lr in product(hidden_dims, num_layers_list, dropouts, weight_decays, lrs):
        i += 1
        config = {
            'hidden_dim': hd,
            'num_layers': nl,
            'dropout': do,
            'weight_decay': wd,
            'lr': lr
        }
        
        val_acc, tm_acc, model = train_and_evaluate(
            config, X_train, y_train, X_val, y_val, X_tm, y_tm, epochs=50
        )
        
        n_params = count_parameters(model)
        gap = val_acc - tm_acc
        
        results.append({
            **config,
            'params': n_params,
            'val_acc': val_acc,
            'tm_acc': tm_acc,
            'gap': gap
        })
        
        if i % 20 == 0:
            print(f"Progress: {i}/{total}")
    
    # Trier par accuracy TM (notre cible!)
    results.sort(key=lambda x: x['tm_acc'], reverse=True)
    
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS (par accuracy Terres Maudites)")
    print("=" * 70)
    
    for i, r in enumerate(results[:10]):
        print(f"\n#{i+1}: TM Acc: {r['tm_acc']:.2%} | Val Acc: {r['val_acc']:.2%} | Gap: {r['gap']:+.2%}")
        print(f"    hidden={r['hidden_dim']}, layers={r['num_layers']}, dropout={r['dropout']}, wd={r['weight_decay']}, lr={r['lr']}")
        print(f"    Params: {r['params']}")
    
    # Sauvegarder la meilleure config
    best = results[0]
    print("\n" + "=" * 70)
    print("MEILLEURE CONFIG TROUVEE")
    print("=" * 70)
    print(f"hidden_dim={best['hidden_dim']}, num_layers={best['num_layers']}, dropout={best['dropout']}")
    print(f"weight_decay={best['weight_decay']}, lr={best['lr']}")
    print(f"Params: {best['params']}, TM Acc: {best['tm_acc']:.2%}")
    
    # Commande pour entrainer
    print("\n" + "=" * 70)
    print("COMMANDE D'ENTRAINEMENT")
    print("=" * 70)
    cmd = f"uv run train_oracle.py --normalize --shuffle --early_stopping --hidden_dim {best['hidden_dim']} --num_layers {best['num_layers']} --dropout {best['dropout']} --weight_decay {best['weight_decay']} --learning_rate {best['lr']} --epochs 100"
    print(cmd)
    
    return results


if __name__ == "__main__":
    main()
