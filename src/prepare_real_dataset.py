"""
Script pour préparer le dataset d'emails réels avec du texte pour notre système.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_real_email_dataset(input_path: str, output_path: str):
    """
    Prépare le dataset d'emails réels pour notre système.
    
    Args:
        input_path: Chemin vers le fichier CSV original
        output_path: Chemin pour sauvegarder le dataset préparé
    """
    logger.info(f"Chargement du dataset depuis {input_path}")
    
    # Charger le dataset
    df = pd.read_csv(input_path)
    logger.info(f"Dataset original: {len(df)} emails")
    logger.info(f"Distribution: {df['Email Type'].value_counts().to_dict()}")
    
    # Nettoyer et préparer les données
    # Supprimer la colonne index inutile
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Renommer les colonnes pour correspondre à notre système
    df = df.rename(columns={
        'Email Text': 'text',
        'Email Type': 'label_text'
    })
    
    # Convertir les labels textuels en numériques
    # 0 = Safe Email (légitime), 1 = Phishing Email
    df['label'] = df['label_text'].map({
        'Safe Email': 0,
        'Phishing Email': 1
    })
    
    # Supprimer les emails vides ou avec du texte invalide
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 10]  # Au moins 10 caractères
    df = df[df['text'] != 'empty']  # Supprimer les emails marqués "empty"
    
    logger.info(f"Dataset après nettoyage: {len(df)} emails")
    logger.info(f"Distribution finale: {df['label'].value_counts().to_dict()}")
    
    # Créer les splits train/validation/test
    # 60% train, 20% validation, 20% test
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=42, stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    
    logger.info(f"Train set: {len(train_df)} emails")
    logger.info(f"Validation set: {len(val_df)} emails")
    logger.info(f"Test set: {len(test_df)} emails")
    
    # Sauvegarder les datasets
    os.makedirs(output_path, exist_ok=True)
    
    # Dataset complet
    df[['text', 'label']].to_csv(
        os.path.join(output_path, 'phishing_emails_full.csv'), 
        index=False
    )
    
    # Splits séparés
    train_df[['text', 'label']].to_csv(
        os.path.join(output_path, 'train.csv'), 
        index=False
    )
    
    val_df[['text', 'label']].to_csv(
        os.path.join(output_path, 'validation.csv'), 
        index=False
    )
    
    test_df[['text', 'label']].to_csv(
        os.path.join(output_path, 'test.csv'), 
        index=False
    )
    
    # Créer un échantillon pour tests rapides
    sample_df = df.sample(n=1000, random_state=42)
    sample_df[['text', 'label']].to_csv(
        os.path.join(output_path, 'sample_1000.csv'), 
        index=False
    )
    
    logger.info(f"Datasets sauvegardés dans {output_path}")
    
    # Afficher quelques statistiques
    print("\n=== STATISTIQUES DU DATASET ===")
    print(f"Total d'emails: {len(df):,}")
    print(f"Emails légitimes: {len(df[df['label']==0]):,}")
    print(f"Emails de phishing: {len(df[df['label']==1]):,}")
    print(f"Ratio phishing: {len(df[df['label']==1])/len(df)*100:.1f}%")
    
    print(f"\nLongueur moyenne du texte:")
    print(f"Légitimes: {df[df['label']==0]['text'].str.len().mean():.0f} caractères")
    print(f"Phishing: {df[df['label']==1]['text'].str.len().mean():.0f} caractères")
    
    print(f"\nExemples d'emails de phishing:")
    phishing_examples = df[df['label']==1]['text'].head(3)
    for i, email in enumerate(phishing_examples, 1):
        print(f"\n{i}. {email[:150]}...")
    
    return df

def main():
    """Fonction principale."""
    input_file = "data/Phishing_Email.csv"
    output_dir = "data/processed/"
    
    if not os.path.exists(input_file):
        print(f"Erreur: Le fichier {input_file} n'existe pas.")
        print("Assurez-vous d'avoir téléchargé le dataset depuis Kaggle.")
        return
    
    try:
        df = prepare_real_email_dataset(input_file, output_dir)
        print(f"\n✅ Dataset préparé avec succès!")
        print(f"📁 Fichiers créés dans: {output_dir}")
        print(f"📊 Prêt pour l'entraînement avec: python src/train.py --data_path data/processed/train.csv")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise

if __name__ == "__main__":
    main()
