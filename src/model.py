"""
Module pour le modèle hybride DistilBERT + features lexicales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhishingDetectorModel(nn.Module):
    """
    Modèle hybride pour la détection de phishing.
    Combine DistilBERT avec des features lexicales.
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_lexical_features: int = 20,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.3,
                 num_classes: int = 2):
        """
        Initialise le modèle hybride.
        
        Args:
            model_name: Nom du modèle DistilBERT pré-entraîné
            num_lexical_features: Nombre de features lexicales
            hidden_dim: Dimension de la couche cachée
            dropout_rate: Taux de dropout
            num_classes: Nombre de classes (2 pour phishing/légitime)
        """
        super(PhishingDetectorModel, self).__init__()
        
        self.num_lexical_features = num_lexical_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Backbone DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.bert_hidden_size = self.distilbert.config.hidden_size  # 768 pour DistilBERT
        
        # Couches pour les features lexicales
        self.lexical_projection = nn.Linear(num_lexical_features, hidden_dim)
        self.lexical_bn = nn.BatchNorm1d(hidden_dim)
        
        # Couches de fusion
        self.fusion_dim = self.bert_hidden_size + hidden_dim
        self.fusion_layer = nn.Linear(self.fusion_dim, hidden_dim)
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)
        
        # Couches de classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids des couches personnalisées."""
        for module in [self.lexical_projection, self.fusion_layer]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                lexical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass du modèle.
        
        Args:
            input_ids: IDs des tokens (batch_size, seq_len)
            attention_mask: Masque d'attention (batch_size, seq_len)
            lexical_features: Features lexicales (batch_size, num_lexical_features)
            
        Returns:
            Dictionnaire avec logits et probabilités
        """
        batch_size = input_ids.size(0)
        
        # Embeddings DistilBERT
        bert_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Utiliser le token [CLS] pour la classification
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        cls_embedding = self.dropout(cls_embedding)
        
        # Traitement des features lexicales
        lexical_projected = self.lexical_projection(lexical_features)  # (batch_size, hidden_dim)
        
        # Normalisation batch si batch_size > 1
        if batch_size > 1:
            lexical_projected = self.lexical_bn(lexical_projected)
        
        lexical_projected = F.relu(lexical_projected)
        lexical_projected = self.dropout(lexical_projected)
        
        # Fusion des embeddings
        fused_features = torch.cat([cls_embedding, lexical_projected], dim=1)  # (batch_size, fusion_dim)
        
        # Couche de fusion
        fused_output = self.fusion_layer(fused_features)  # (batch_size, hidden_dim)
        
        # Normalisation batch si batch_size > 1
        if batch_size > 1:
            fused_output = self.fusion_bn(fused_output)
        
        fused_output = F.relu(fused_output)
        fused_output = self.dropout(fused_output)
        
        # Classification
        logits = self.classifier(fused_output)  # (batch_size, num_classes)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'cls_embedding': cls_embedding,
            'lexical_features': lexical_projected,
            'fused_features': fused_output
        }
    
    def predict(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                lexical_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Effectue une prédiction.
        
        Args:
            input_ids: IDs des tokens
            attention_mask: Masque d'attention
            lexical_features: Features lexicales
            
        Returns:
            Tuple (prédictions, probabilités)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, lexical_features)
            predictions = torch.argmax(outputs['probabilities'], dim=1)
            return predictions, outputs['probabilities']
    
    def get_attention_weights(self, 
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extrait les poids d'attention de DistilBERT.
        
        Args:
            input_ids: IDs des tokens
            attention_mask: Masque d'attention
            
        Returns:
            Poids d'attention
        """
        self.eval()
        with torch.no_grad():
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            return outputs.attentions
    
    def freeze_bert_layers(self, num_layers_to_freeze: int = 4):
        """
        Gèle les premières couches de DistilBERT.
        
        Args:
            num_layers_to_freeze: Nombre de couches à geler
        """
        logger.info(f"Gel de {num_layers_to_freeze} couches de DistilBERT")
        
        # Geler les embeddings
        for param in self.distilbert.embeddings.parameters():
            param.requires_grad = False
        
        # Geler les premières couches du transformer
        for i in range(min(num_layers_to_freeze, len(self.distilbert.transformer.layer))):
            for param in self.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False
    
    def unfreeze_all_layers(self):
        """Dégèle toutes les couches du modèle."""
        logger.info("Dégel de toutes les couches")
        for param in self.parameters():
            param.requires_grad = True
    
    def get_model_size(self) -> Dict[str, int]:
        """
        Retourne la taille du modèle.
        
        Returns:
            Dictionnaire avec les informations de taille
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }


class PhishingDataset(torch.utils.data.Dataset):
    """Dataset personnalisé pour les données de phishing."""
    
    def __init__(self, 
                 encodings: Dict,
                 lexical_features: np.ndarray,
                 labels: list):
        """
        Initialise le dataset.
        
        Args:
            encodings: Encodages des textes (sortie du tokenizer)
            lexical_features: Features lexicales
            labels: Labels des exemples
        """
        self.encodings = encodings
        self.lexical_features = torch.tensor(lexical_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __getitem__(self, idx):
        """Retourne un exemple du dataset."""
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'lexical_features': self.lexical_features[idx],
            'labels': self.labels[idx]
        }
        return item
    
    def __len__(self):
        """Retourne la taille du dataset."""
        return len(self.labels)


class ModelTrainer:
    """Classe utilitaire pour l'entraînement du modèle."""
    
    def __init__(self, 
                 model: PhishingDetectorModel,
                 device: torch.device = None):
        """
        Initialise le trainer.
        
        Args:
            model: Modèle à entraîner
            device: Device pour l'entraînement
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Modèle déplacé sur {self.device}")
        logger.info(f"Taille du modèle: {model.get_model_size()}")
    
    def train_epoch(self, 
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> Dict[str, float]:
        """
        Entraîne le modèle pour une époque.
        
        Args:
            dataloader: DataLoader d'entraînement
            optimizer: Optimiseur
            criterion: Fonction de perte
            
        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch in dataloader:
            # Déplacer les données sur le device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            lexical_features = batch['lexical_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, lexical_features)
            
            # Calcul de la perte
            loss = criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Métriques
            total_loss += loss.item()
            predictions = torch.argmax(outputs['probabilities'], dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self, 
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module) -> Dict[str, float]:
        """
        Évalue le modèle.
        
        Args:
            dataloader: DataLoader de validation
            criterion: Fonction de perte
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Déplacer les données sur le device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                lexical_features = batch['lexical_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, lexical_features)
                
                # Calcul de la perte
                loss = criterion(outputs['logits'], labels)
                
                # Métriques
                total_loss += loss.item()
                predictions = torch.argmax(outputs['probabilities'], dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }


if __name__ == "__main__":
    # Test du modèle
    logger.info("Test du modèle PhishingDetectorModel")
    
    # Paramètres de test
    batch_size = 2
    seq_length = 128
    num_lexical_features = 20
    
    # Créer des données de test
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    lexical_features = torch.randn(batch_size, num_lexical_features)
    
    # Initialiser le modèle
    model = PhishingDetectorModel(num_lexical_features=num_lexical_features)
    
    # Test forward pass
    outputs = model(input_ids, attention_mask, lexical_features)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Probabilities shape: {outputs['probabilities'].shape}")
    print(f"Model size: {model.get_model_size()}")
    
    # Test prédiction
    predictions, probs = model.predict(input_ids, attention_mask, lexical_features)
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probs}")
