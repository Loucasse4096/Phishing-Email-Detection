"""
Script d'entraînement pour le modèle de détection de phishing.
Inclut l'adversarial training pour améliorer la robustesse.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import argparse
from datetime import datetime
import logging
from typing import Dict, List, Tuple

# Imports locaux
from data_loader import EmailDataLoader
from features import FeatureExtractor
from adversarial import AdversarialAttacker
from model import PhishingDetectorModel, PhishingDataset, ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdversarialTrainer:
    """Classe pour l'entraînement adversarial."""
    
    def __init__(self, 
                 model: PhishingDetectorModel,
                 data_loader: EmailDataLoader,
                 feature_extractor: FeatureExtractor,
                 adversarial_attacker: AdversarialAttacker,
                 device: torch.device = None):
        """
        Initialise l'entraîneur adversarial.
        
        Args:
            model: Modèle à entraîner
            data_loader: Chargeur de données
            feature_extractor: Extracteur de features
            adversarial_attacker: Générateur d'attaques
            device: Device pour l'entraînement
        """
        self.model = model
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor
        self.adversarial_attacker = adversarial_attacker
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = ModelTrainer(model, device)
        
    def create_adversarial_batch(self, 
                               texts: List[str], 
                               labels: List[int],
                               adversarial_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """
        Crée un batch avec des exemples adversariaux.
        
        Args:
            texts: Textes originaux
            labels: Labels originaux
            adversarial_ratio: Ratio d'exemples adversariaux
            
        Returns:
            Tuple (textes_augmentés, labels_augmentés)
        """
        # Sélectionner des exemples pour les attaques adversariales
        num_adversarial = int(len(texts) * adversarial_ratio)
        indices = np.random.choice(len(texts), size=num_adversarial, replace=False)
        
        adversarial_texts = []
        adversarial_labels = []
        
        for idx in indices:
            original_text = texts[idx]
            original_label = labels[idx]
            
            # Générer une version adversariale
            adversarial_text = self.adversarial_attacker.generate_adversarial_examples([original_text])[0]
            
            adversarial_texts.append(adversarial_text)
            adversarial_labels.append(original_label)
        
        # Combiner avec les exemples originaux
        augmented_texts = texts + adversarial_texts
        augmented_labels = labels + adversarial_labels
        
        return augmented_texts, augmented_labels
    
    def train_with_adversarial_examples(self,
                                      train_texts: List[str],
                                      train_labels: List[int],
                                      val_texts: List[str],
                                      val_labels: List[int],
                                      num_epochs: int = 10,
                                      batch_size: int = 16,
                                      learning_rate: float = 2e-5,
                                      adversarial_ratio: float = 0.2,
                                      save_path: str = "models/"):
        """
        Entraîne le modèle avec des exemples adversariaux.
        
        Args:
            train_texts: Textes d'entraînement
            train_labels: Labels d'entraînement
            val_texts: Textes de validation
            val_labels: Labels de validation
            num_epochs: Nombre d'époques
            batch_size: Taille des batches
            learning_rate: Taux d'apprentissage
            adversarial_ratio: Ratio d'exemples adversariaux
            save_path: Chemin de sauvegarde
        """
        logger.info("Début de l'entraînement adversarial")
        logger.info(f"Train: {len(train_texts)} exemples, Val: {len(val_texts)} exemples")
        logger.info(f"Adversarial ratio: {adversarial_ratio}")
        
        # Créer le dossier de sauvegarde
        os.makedirs(save_path, exist_ok=True)
        
        # Préparer les données de validation (sans adversarial)
        val_encodings = self.data_loader.tokenize_texts(val_texts)
        val_features = self.feature_extractor.extract_batch_features(val_texts)
        val_dataset = PhishingDataset(val_encodings, val_features, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimiseur et critère
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        
        # Historique d'entraînement
        train_history = []
        val_history = []
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"Époque {epoch + 1}/{num_epochs}")
            
            # Créer des exemples adversariaux pour cette époque
            augmented_texts, augmented_labels = self.create_adversarial_batch(
                train_texts, train_labels, adversarial_ratio
            )
            
            logger.info(f"Dataset augmenté: {len(augmented_texts)} exemples "
                       f"({len(train_texts)} originaux + {len(augmented_texts) - len(train_texts)} adversariaux)")
            
            # Préparer les données d'entraînement
            train_encodings = self.data_loader.tokenize_texts(augmented_texts)
            train_features = self.feature_extractor.extract_batch_features(augmented_texts)
            train_dataset = PhishingDataset(train_encodings, train_features, augmented_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Entraînement
            train_metrics = self.trainer.train_epoch(train_dataloader, optimizer, criterion)
            
            # Validation
            val_metrics = self.trainer.evaluate(val_dataloader, criterion)
            
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
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_val_acc': best_val_acc
                }, os.path.join(save_path, 'best_model.pth'))
                logger.info(f"Nouveau meilleur modèle sauvegardé (Val Acc: {best_val_acc:.4f})")
        
        # Sauvegarder l'historique d'entraînement
        history = {
            'train_history': train_history,
            'val_history': val_history,
            'config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'adversarial_ratio': adversarial_ratio,
                'best_val_acc': best_val_acc
            }
        }
        
        with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Entraînement terminé. Meilleure validation accuracy: {best_val_acc:.4f}")
        return history


def main():
    """Fonction principale d'entraînement."""
    parser = argparse.ArgumentParser(description='Entraînement du modèle de détection de phishing')
    parser.add_argument('--data_path', type=str, required=True, help='Chemin vers le dataset CSV')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Modèle DistilBERT à utiliser')
    parser.add_argument('--num_epochs', type=int, default=10, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=16, help='Taille des batches')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Taux d\'apprentissage')
    parser.add_argument('--adversarial_ratio', type=float, default=0.2, help='Ratio d\'exemples adversariaux')
    parser.add_argument('--test_size', type=float, default=0.2, help='Taille du test set')
    parser.add_argument('--save_path', type=str, default='models/', help='Dossier de sauvegarde')
    parser.add_argument('--freeze_layers', type=int, default=4, help='Nombre de couches BERT à geler')
    parser.add_argument('--text_column', type=str, default='text', help='Nom de la colonne de texte')
    parser.add_argument('--label_column', type=str, default='label', help='Nom de la colonne de label')
    
    args = parser.parse_args()
    
    # Configuration du logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.save_path, f'training_{timestamp}.log')
    os.makedirs(args.save_path, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("=== Début de l'entraînement ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Vérifier la disponibilité du GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device utilisé: {device}")
    
    try:
        # Initialiser les composants
        logger.info("Initialisation des composants...")
        data_loader = EmailDataLoader(model_name=args.model_name)
        feature_extractor = FeatureExtractor()
        adversarial_attacker = AdversarialAttacker()
        
        # Charger et préparer les données
        logger.info("Chargement des données...")
        train_encodings, val_encodings, train_labels, val_labels = data_loader.prepare_dataset(
            args.data_path, 
            test_size=args.test_size,
            text_column=args.text_column,
            label_column=args.label_column
        )
        
        # Récupérer les textes originaux pour l'adversarial training
        df = pd.read_csv(args.data_path)
        all_texts = [data_loader.preprocess_text(str(text)) for text in df[args.text_column]]
        all_labels = []
        for label in df[args.label_column]:
            if isinstance(label, str):
                if label.lower() in ['phishing', 'spam', '1', 'true']:
                    all_labels.append(1)
                else:
                    all_labels.append(0)
            else:
                all_labels.append(int(label))
        
        # Split train/val pour les textes
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels_list, val_labels_list = train_test_split(
            all_texts, all_labels, test_size=args.test_size, random_state=42, stratify=all_labels
        )
        
        # Initialiser le modèle
        logger.info("Initialisation du modèle...")
        num_lexical_features = len(feature_extractor.get_feature_names())
        model = PhishingDetectorModel(
            model_name=args.model_name,
            num_lexical_features=num_lexical_features
        )
        
        # Geler certaines couches si demandé
        if args.freeze_layers > 0:
            model.freeze_bert_layers(args.freeze_layers)
        
        logger.info(f"Taille du modèle: {model.get_model_size()}")
        
        # Initialiser l'entraîneur adversarial
        adversarial_trainer = AdversarialTrainer(
            model=model,
            data_loader=data_loader,
            feature_extractor=feature_extractor,
            adversarial_attacker=adversarial_attacker,
            device=device
        )
        
        # Entraîner le modèle
        logger.info("Début de l'entraînement...")
        history = adversarial_trainer.train_with_adversarial_examples(
            train_texts=train_texts,
            train_labels=train_labels_list,
            val_texts=val_texts,
            val_labels=val_labels_list,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            adversarial_ratio=args.adversarial_ratio,
            save_path=args.save_path
        )
        
        # Sauvegarder la configuration finale
        final_config = {
            'model_config': {
                'model_name': args.model_name,
                'num_lexical_features': num_lexical_features,
                'model_size': model.get_model_size()
            },
            'training_config': vars(args),
            'final_metrics': {
                'best_val_accuracy': history['config']['best_val_acc'],
                'final_train_accuracy': history['train_history'][-1]['accuracy'],
                'final_val_accuracy': history['val_history'][-1]['accuracy']
            },
            'feature_names': feature_extractor.get_feature_names(),
            'timestamp': timestamp
        }
        
        with open(os.path.join(args.save_path, 'model_config.json'), 'w') as f:
            json.dump(final_config, f, indent=2)
        
        logger.info("=== Entraînement terminé avec succès ===")
        logger.info(f"Modèle sauvegardé dans: {args.save_path}")
        logger.info(f"Meilleure validation accuracy: {history['config']['best_val_acc']:.4f}")
        
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {e}")
        raise


if __name__ == "__main__":
    main()
