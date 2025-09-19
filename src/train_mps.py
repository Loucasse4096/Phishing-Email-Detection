"""
Script d'entraînement simplifié sans attaques adversariales.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import EmailDataLoader
from features import FeatureExtractor
from model import PhishingDetectorModel, PhishingDataset, ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_train(data_path: str, 
                num_epochs: int = 5,
                batch_size: int = 8,
                learning_rate: float = 2e-5,
                save_path: str = "models/simple/"):
    """
    Entraînement simple du modèle.
    
    Args:
        data_path: Chemin vers le dataset
        num_epochs: Nombre d'époques
        batch_size: Taille des batches
        learning_rate: Taux d'apprentissage
        save_path: Dossier de sauvegarde
    """
    logger.info("=== Début de l'entraînement simple ===")
    
    # Device - priorité à MPS sur macOS, puis CUDA, puis CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Device utilisé: {device}")
    
    # Créer le dossier de sauvegarde
    os.makedirs(save_path, exist_ok=True)
    
    # Initialiser les composants
    logger.info("Initialisation des composants...")
    data_loader = EmailDataLoader()
    feature_extractor = FeatureExtractor()
    
    # Charger et préparer les données
    logger.info("Chargement des données...")
    train_encodings, val_encodings, train_labels, val_labels = data_loader.prepare_dataset(
        data_path, test_size=0.2
    )
    
    # Récupérer les textes pour extraire les features
    import pandas as pd
    df = pd.read_csv(data_path)
    all_texts = [data_loader.preprocess_text(str(text)) for text in df['text']]
    all_labels = df['label'].tolist()
    
    # Split identique
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels_list, val_labels_list = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # Extraire les features
    logger.info("Extraction des features...")
    train_features = feature_extractor.extract_batch_features(train_texts)
    val_features = feature_extractor.extract_batch_features(val_texts)
    
    # Créer les datasets
    train_dataset = PhishingDataset(train_encodings, train_features, train_labels)
    val_dataset = PhishingDataset(val_encodings, val_features, val_labels)
    
    # Créer les dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialiser le modèle
    logger.info("Initialisation du modèle...")
    num_lexical_features = len(feature_extractor.get_feature_names())
    model = PhishingDetectorModel(num_lexical_features=num_lexical_features)
    
    # Geler quelques couches BERT
    model.freeze_bert_layers(4)
    logger.info(f"Taille du modèle: {model.get_model_size()}")
    
    # Initialiser l'entraîneur
    trainer = ModelTrainer(model, device)
    
    # Optimiseur et critère
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # Historique d'entraînement
    train_history = []
    val_history = []
    best_val_acc = 0.0
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        logger.info(f"Époque {epoch + 1}/{num_epochs}")
        
        # Entraînement
        train_metrics = trainer.train_epoch(train_dataloader, optimizer, criterion)
        
        # Validation
        val_metrics = trainer.evaluate(val_dataloader, criterion)
        
        # Mise à jour du scheduler
        scheduler.step()
        
        # Logging
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Sauvegarder l'historique
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        # Sauvegarder le meilleur modèle
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_acc': best_val_acc
            }, os.path.join(save_path, 'best_model.pth'))
            logger.info(f"Nouveau meilleur modèle sauvegardé (Val Acc: {best_val_acc:.4f})")
    
    # Sauvegarder la configuration
    config = {
        'model_config': {
            'model_name': 'distilbert-base-uncased',
            'num_lexical_features': num_lexical_features,
            'model_size': model.get_model_size()
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_val_acc': best_val_acc
        },
        'final_metrics': {
            'final_train_accuracy': train_history[-1]['accuracy'],
            'final_val_accuracy': val_history[-1]['accuracy'],
            'best_val_accuracy': best_val_acc
        },
        'feature_names': feature_extractor.get_feature_names(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Sauvegarder l'historique
    history = {
        'train_history': train_history,
        'val_history': val_history,
        'config': config
    }
    
    with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Entraînement terminé. Meilleure validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Modèle sauvegardé dans: {save_path}")
    
    return model, history

def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraînement simple du modèle')
    parser.add_argument('--data_path', type=str, required=True, help='Chemin vers le dataset')
    parser.add_argument('--num_epochs', type=int, default=5, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=8, help='Taille des batches')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Taux d\'apprentissage')
    parser.add_argument('--save_path', type=str, default='models/simple/', help='Dossier de sauvegarde')
    
    args = parser.parse_args()
    
    try:
        model, history = simple_train(
            args.data_path,
            args.num_epochs,
            args.batch_size,
            args.learning_rate,
            args.save_path
        )
        
        print(f"\n✅ Entraînement terminé avec succès!")
        print(f"📊 Meilleure accuracy: {history['config']['training_config']['best_val_acc']:.4f}")
        print(f"💾 Modèle sauvegardé dans: {args.save_path}")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise

if __name__ == "__main__":
    main()
