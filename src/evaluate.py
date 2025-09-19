"""
Script d'évaluation pour le modèle de détection de phishing.
Évalue la performance sur données clean et adversariales.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support,
                           roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# Imports locaux
from data_loader import EmailDataLoader
from features import FeatureExtractor
from adversarial import AdversarialAttacker
from model import PhishingDetectorModel, PhishingDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Classe pour évaluer la robustesse du modèle."""
    
    def __init__(self, 
                 model: PhishingDetectorModel,
                 data_loader: EmailDataLoader,
                 feature_extractor: FeatureExtractor,
                 adversarial_attacker: AdversarialAttacker,
                 device: torch.device = None):
        """
        Initialise l'évaluateur.
        
        Args:
            model: Modèle à évaluer
            data_loader: Chargeur de données
            feature_extractor: Extracteur de features
            adversarial_attacker: Générateur d'attaques
            device: Device pour l'évaluation
        """
        self.model = model
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor
        self.adversarial_attacker = adversarial_attacker
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_clean_data(self, 
                          texts: List[str], 
                          labels: List[int],
                          batch_size: int = 32) -> Dict:
        """
        Évalue le modèle sur des données propres.
        
        Args:
            texts: Textes à évaluer
            labels: Vrais labels
            batch_size: Taille des batches
            
        Returns:
            Dictionnaire avec les métriques
        """
        logger.info(f"Évaluation sur données propres: {len(texts)} exemples")
        
        # Préparer les données
        encodings = self.data_loader.tokenize_texts(texts)
        features = self.feature_extractor.extract_batch_features(texts)
        dataset = PhishingDataset(encodings, features, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Prédictions
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                lexical_features = batch['lexical_features'].to(self.device)
                batch_labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, lexical_features)
                predictions = torch.argmax(outputs['probabilities'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(outputs['probabilities'].cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculer les métriques
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        metrics['dataset_type'] = 'clean'
        
        return metrics
    
    def evaluate_adversarial_data(self, 
                                texts: List[str], 
                                labels: List[int],
                                attack_types: List[str] = None,
                                intensity: float = 0.1,
                                batch_size: int = 32) -> Dict:
        """
        Évalue le modèle sur des données adversariales.
        
        Args:
            texts: Textes originaux
            labels: Vrais labels
            attack_types: Types d'attaques à appliquer
            intensity: Intensité des attaques
            batch_size: Taille des batches
            
        Returns:
            Dictionnaire avec les métriques adversariales
        """
        if attack_types is None:
            attack_types = ['swap', 'delete', 'insert', 'homoglyph', 'url_obfuscation', 'combined']
        
        logger.info(f"Évaluation adversariale: {len(texts)} exemples")
        logger.info(f"Types d'attaques: {attack_types}, Intensité: {intensity}")
        
        results = {}
        
        for attack_type in attack_types:
            logger.info(f"Évaluation avec attaque: {attack_type}")
            
            # Générer les exemples adversariaux
            if attack_type == 'swap':
                adversarial_texts = [self.adversarial_attacker.character_swap_attack(text, intensity) for text in texts]
            elif attack_type == 'delete':
                adversarial_texts = [self.adversarial_attacker.character_deletion_attack(text, intensity) for text in texts]
            elif attack_type == 'insert':
                adversarial_texts = [self.adversarial_attacker.character_insertion_attack(text, intensity) for text in texts]
            elif attack_type == 'homoglyph':
                adversarial_texts = [self.adversarial_attacker.homoglyph_attack(text, intensity) for text in texts]
            elif attack_type == 'url_obfuscation':
                adversarial_texts = [self.adversarial_attacker.url_obfuscation_attack(text) for text in texts]
            elif attack_type == 'combined':
                adversarial_texts = [self.adversarial_attacker.combined_typo_attack(text, intensity) for text in texts]
            else:
                adversarial_texts = texts
            
            # Évaluer sur les données adversariales
            metrics = self.evaluate_clean_data(adversarial_texts, labels, batch_size)
            metrics['dataset_type'] = f'adversarial_{attack_type}'
            metrics['attack_intensity'] = intensity
            
            # Calculer les métriques d'attaque
            attack_metrics = self.adversarial_attacker.evaluate_attack_success(texts, adversarial_texts)
            metrics.update(attack_metrics)
            
            results[attack_type] = metrics
        
        return results
    
    def _calculate_metrics(self, 
                         true_labels: List[int], 
                         predictions: List[int], 
                         probabilities: List[List[float]]) -> Dict:
        """
        Calcule les métriques de classification.
        
        Args:
            true_labels: Vrais labels
            predictions: Prédictions du modèle
            probabilities: Probabilités prédites
            
        Returns:
            Dictionnaire des métriques
        """
        # Métriques de base
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        
        # Métriques par classe
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        # AUC-ROC
        probabilities_array = np.array(probabilities)
        if probabilities_array.shape[1] == 2:  # Classification binaire
            auc_roc = roc_auc_score(true_labels, probabilities_array[:, 1])
        else:
            auc_roc = roc_auc_score(true_labels, probabilities_array, multi_class='ovr')
        
        # Matrice de confusion
        cm = confusion_matrix(true_labels, predictions)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'support_per_class': support.tolist() if hasattr(support, 'tolist') else [int(support)]
        }
        
        return metrics
    
    def compare_clean_vs_adversarial(self, 
                                   clean_metrics: Dict, 
                                   adversarial_results: Dict) -> Dict:
        """
        Compare les performances sur données propres vs adversariales.
        
        Args:
            clean_metrics: Métriques sur données propres
            adversarial_results: Résultats sur données adversariales
            
        Returns:
            Dictionnaire de comparaison
        """
        comparison = {
            'clean_performance': {
                'accuracy': clean_metrics['accuracy'],
                'precision': clean_metrics['precision'],
                'recall': clean_metrics['recall'],
                'f1_score': clean_metrics['f1_score'],
                'auc_roc': clean_metrics['auc_roc']
            },
            'adversarial_performance': {},
            'robustness_metrics': {}
        }
        
        for attack_type, metrics in adversarial_results.items():
            comparison['adversarial_performance'][attack_type] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'auc_roc': metrics['auc_roc']
            }
            
            # Calcul de la dégradation
            accuracy_drop = clean_metrics['accuracy'] - metrics['accuracy']
            f1_drop = clean_metrics['f1_score'] - metrics['f1_score']
            
            comparison['robustness_metrics'][attack_type] = {
                'accuracy_drop': float(accuracy_drop),
                'f1_drop': float(f1_drop),
                'accuracy_retention': float(metrics['accuracy'] / clean_metrics['accuracy']),
                'f1_retention': float(metrics['f1_score'] / clean_metrics['f1_score']),
                'avg_edit_distance': metrics.get('avg_edit_distance', 0),
                'avg_similarity': metrics.get('avg_similarity', 1)
            }
        
        # Moyennes sur tous les types d'attaques
        avg_accuracy_drop = np.mean([m['accuracy_drop'] for m in comparison['robustness_metrics'].values()])
        avg_f1_drop = np.mean([m['f1_drop'] for m in comparison['robustness_metrics'].values()])
        
        comparison['overall_robustness'] = {
            'avg_accuracy_drop': float(avg_accuracy_drop),
            'avg_f1_drop': float(avg_f1_drop),
            'max_accuracy_drop': float(max([m['accuracy_drop'] for m in comparison['robustness_metrics'].values()])),
            'max_f1_drop': float(max([m['f1_drop'] for m in comparison['robustness_metrics'].values()]))
        }
        
        return comparison
    
    def plot_confusion_matrices(self, 
                              clean_metrics: Dict, 
                              adversarial_results: Dict, 
                              save_path: str):
        """
        Génère les matrices de confusion.
        
        Args:
            clean_metrics: Métriques sur données propres
            adversarial_results: Résultats adversariaux
            save_path: Chemin de sauvegarde
        """
        num_attacks = len(adversarial_results) + 1
        cols = min(3, num_attacks)
        rows = (num_attacks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_attacks == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Matrice de confusion pour données propres
        cm_clean = np.array(clean_metrics['confusion_matrix'])
        sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Clean Data')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Matrices pour données adversariales
        for i, (attack_type, metrics) in enumerate(adversarial_results.items(), 1):
            if i < len(axes):
                cm_adv = np.array(metrics['confusion_matrix'])
                sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Reds', ax=axes[i])
                axes[i].set_title(f'Adversarial: {attack_type}')
                axes[i].set_ylabel('True Label')
                axes[i].set_xlabel('Predicted Label')
        
        # Cacher les axes inutilisés
        for i in range(num_attacks, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_robustness_comparison(self, 
                                 comparison: Dict, 
                                 save_path: str):
        """
        Génère un graphique de comparaison de robustesse.
        
        Args:
            comparison: Résultats de comparaison
            save_path: Chemin de sauvegarde
        """
        attack_types = list(comparison['adversarial_performance'].keys())
        
        # Métriques à comparer
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        clean_values = [comparison['clean_performance'][metric] for metric in metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique 1: Comparaison des métriques
        x = np.arange(len(metrics))
        width = 0.15
        
        ax1.bar(x - width*2, clean_values, width, label='Clean', color='green', alpha=0.7)
        
        colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, attack_type in enumerate(attack_types):
            adv_values = [comparison['adversarial_performance'][attack_type][metric] for metric in metrics]
            ax1.bar(x - width + i*width, adv_values, width, 
                   label=f'Adv: {attack_type}', color=colors[i % len(colors)], alpha=0.7)
        
        ax1.set_xlabel('Métriques')
        ax1.set_ylabel('Score')
        ax1.set_title('Comparaison Clean vs Adversarial')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Dégradation des performances
        accuracy_drops = [comparison['robustness_metrics'][attack]['accuracy_drop'] for attack in attack_types]
        f1_drops = [comparison['robustness_metrics'][attack]['f1_drop'] for attack in attack_types]
        
        x2 = np.arange(len(attack_types))
        ax2.bar(x2 - 0.2, accuracy_drops, 0.4, label='Accuracy Drop', color='red', alpha=0.7)
        ax2.bar(x2 + 0.2, f1_drops, 0.4, label='F1 Drop', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Type d\'attaque')
        ax2.set_ylabel('Dégradation')
        ax2.set_title('Dégradation des performances')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(attack_types, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'robustness_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, 
                                 clean_metrics: Dict, 
                                 adversarial_results: Dict, 
                                 comparison: Dict, 
                                 save_path: str):
        """
        Génère un rapport d'évaluation complet.
        
        Args:
            clean_metrics: Métriques sur données propres
            adversarial_results: Résultats adversariaux
            comparison: Comparaison des résultats
            save_path: Chemin de sauvegarde
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'clean_evaluation': clean_metrics,
            'adversarial_evaluation': adversarial_results,
            'robustness_comparison': comparison,
            'summary': {
                'clean_accuracy': clean_metrics['accuracy'],
                'clean_f1': clean_metrics['f1_score'],
                'avg_adversarial_accuracy': np.mean([m['accuracy'] for m in adversarial_results.values()]),
                'avg_adversarial_f1': np.mean([m['f1_score'] for m in adversarial_results.values()]),
                'worst_accuracy_drop': comparison['overall_robustness']['max_accuracy_drop'],
                'worst_f1_drop': comparison['overall_robustness']['max_f1_drop'],
                'robustness_score': 1 - comparison['overall_robustness']['avg_accuracy_drop']
            }
        }
        
        # Sauvegarder le rapport JSON
        with open(os.path.join(save_path, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Générer les graphiques
        self.plot_confusion_matrices(clean_metrics, adversarial_results, save_path)
        self.plot_robustness_comparison(comparison, save_path)
        
        logger.info(f"Rapport d'évaluation sauvegardé dans: {save_path}")
        return report


def load_model_from_checkpoint(checkpoint_path: str, 
                             config_path: str, 
                             device: torch.device) -> PhishingDetectorModel:
    """
    Charge un modèle depuis un checkpoint.
    
    Args:
        checkpoint_path: Chemin vers le checkpoint
        config_path: Chemin vers la configuration
        device: Device pour le modèle
        
    Returns:
        Modèle chargé
    """
    # Charger la configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialiser le modèle
    model = PhishingDetectorModel(
        model_name=config['model_config']['model_name'],
        num_lexical_features=config['model_config']['num_lexical_features']
    )
    
    # Charger les poids
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Modèle chargé depuis: {checkpoint_path}")
    logger.info(f"Époque du checkpoint: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"Meilleure validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    
    return model


def main():
    """Fonction principale d'évaluation."""
    parser = argparse.ArgumentParser(description='Évaluation du modèle de détection de phishing')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--config_path', type=str, required=True, help='Chemin vers la configuration')
    parser.add_argument('--test_data_path', type=str, required=True, help='Chemin vers les données de test')
    parser.add_argument('--output_path', type=str, default='evaluation_results/', help='Dossier de sortie')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille des batches')
    parser.add_argument('--adversarial_intensity', type=float, default=0.1, help='Intensité des attaques')
    parser.add_argument('--text_column', type=str, default='text', help='Nom de la colonne de texte')
    parser.add_argument('--label_column', type=str, default='label', help='Nom de la colonne de label')
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie
    os.makedirs(args.output_path, exist_ok=True)
    
    # Configuration du logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_path, f'evaluation_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("=== Début de l'évaluation ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device utilisé: {device}")
    
    try:
        # Charger le modèle
        logger.info("Chargement du modèle...")
        model = load_model_from_checkpoint(args.model_path, args.config_path, device)
        
        # Charger la configuration pour récupérer le nom du modèle
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        
        # Initialiser les composants
        data_loader = EmailDataLoader(model_name=config['model_config']['model_name'])
        feature_extractor = FeatureExtractor()
        adversarial_attacker = AdversarialAttacker()
        
        # Initialiser l'évaluateur
        evaluator = ModelEvaluator(
            model=model,
            data_loader=data_loader,
            feature_extractor=feature_extractor,
            adversarial_attacker=adversarial_attacker,
            device=device
        )
        
        # Charger les données de test
        logger.info("Chargement des données de test...")
        test_texts, test_labels = data_loader.load_dataset(
            args.test_data_path, 
            text_column=args.text_column,
            label_column=args.label_column
        )
        
        # Évaluation sur données propres
        logger.info("Évaluation sur données propres...")
        clean_metrics = evaluator.evaluate_clean_data(test_texts, test_labels, args.batch_size)
        
        # Évaluation adversariale
        logger.info("Évaluation adversariale...")
        adversarial_results = evaluator.evaluate_adversarial_data(
            test_texts, test_labels, 
            intensity=args.adversarial_intensity,
            batch_size=args.batch_size
        )
        
        # Comparaison
        logger.info("Comparaison des résultats...")
        comparison = evaluator.compare_clean_vs_adversarial(clean_metrics, adversarial_results)
        
        # Génération du rapport
        logger.info("Génération du rapport...")
        report = evaluator.generate_evaluation_report(
            clean_metrics, adversarial_results, comparison, args.output_path
        )
        
        # Affichage des résultats principaux
        logger.info("=== RÉSULTATS D'ÉVALUATION ===")
        logger.info(f"Accuracy sur données propres: {clean_metrics['accuracy']:.4f}")
        logger.info(f"F1-Score sur données propres: {clean_metrics['f1_score']:.4f}")
        logger.info(f"Accuracy moyenne adversariale: {report['summary']['avg_adversarial_accuracy']:.4f}")
        logger.info(f"F1-Score moyen adversarial: {report['summary']['avg_adversarial_f1']:.4f}")
        logger.info(f"Pire dégradation accuracy: {report['summary']['worst_accuracy_drop']:.4f}")
        logger.info(f"Score de robustesse: {report['summary']['robustness_score']:.4f}")
        
        logger.info("=== Évaluation terminée avec succès ===")
        logger.info(f"Résultats sauvegardés dans: {args.output_path}")
        
    except Exception as e:
        logger.error(f"Erreur pendant l'évaluation: {e}")
        raise


if __name__ == "__main__":
    main()
