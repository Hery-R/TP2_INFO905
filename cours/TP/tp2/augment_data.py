# -*- coding: utf-8 -*-
"""
Augmentation du dataset d'entrainement avec des exemples
qui suivent les regles des Terres Maudites.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def augment_training_data(train_df, tm_ratio=0.3, seed=42):
    """
    Augmente le dataset d'entrainement avec des exemples synthetiques
    qui suivent les regles des Terres Maudites.
    
    tm_ratio: proportion d'exemples TM a ajouter (0.3 = 30%)
    """
    np.random.seed(seed)
    
    n_tm = int(len(train_df) * tm_ratio)
    
    data = {
        'force': np.random.uniform(0, 100, n_tm),
        'intelligence': np.random.uniform(0, 100, n_tm),
        'agilite': np.random.uniform(0, 100, n_tm),
        'chance': np.random.uniform(0, 100, n_tm),
        'experience': np.random.uniform(0, 20, n_tm),
        'niveau_quete': np.random.uniform(1, 10, n_tm),
        'equipement': np.random.uniform(0, 100, n_tm),
        'fatigue': np.random.uniform(0, 100, n_tm),
    }
    tm_df = pd.DataFrame(data)

    df_n = tm_df.copy()
    df_n['force'] = tm_df['force'] / 100
    df_n['intelligence'] = tm_df['intelligence'] / 100
    df_n['agilite'] = tm_df['agilite'] / 100
    df_n['chance'] = tm_df['chance'] / 100
    df_n['experience'] = tm_df['experience'] / 20
    df_n['niveau_quete'] = (tm_df['niveau_quete'] - 1) / 9
    df_n['equipement'] = tm_df['equipement'] / 100
    df_n['fatigue'] = tm_df['fatigue'] / 100
    
    score = (
        0.40 * df_n['intelligence'] +   
        0.25 * df_n['agilite'] +          
        0.20 * df_n['chance'] +           
        0.10 * df_n['equipement'] +
        -0.05 * df_n['force'] +           
        0.05 * df_n['experience'] -
        0.15 * df_n['fatigue'] -          
        0.15 * df_n['niveau_quete']       
    )
    
    arrogance = tm_df['force'] > 65  
    score[arrogance] -= 0.25         
    
    score += np.random.normal(0, 0.03, n_tm)  
    tm_df['survie'] = (score > 0.25).astype(int)
    
    tm_df = tm_df[train_df.columns]
    
    augmented_df = pd.concat([train_df, tm_df], ignore_index=True)
    
    print(f"Dataset original: {len(train_df)} exemples")
    print(f"Exemples TM ajoutes: {n_tm}")
    print(f"Dataset augmente: {len(augmented_df)} exemples")
    print(f"\nNouvelles correlations:")
    print(augmented_df.corr()['survie'].sort_values())
    
    return augmented_df


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    
    train_df = pd.read_csv(data_dir / "train.csv")
    
    for ratio in [0.8, 1.0, 1.5]:
        print(f"\n{'='*50}")
        print(f"RATIO TM: {ratio}")
        print(f"{'='*50}")
        augmented = augment_training_data(train_df, tm_ratio=ratio)
        augmented.to_csv(data_dir / f"train_augmented_{int(ratio*100)}.csv", index=False)
