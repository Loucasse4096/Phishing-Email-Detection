"""
Module pour le chargement et le préprocessing des données d'emails de phishing.
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailDataLoader:
    """Classe pour charger et préprocesser les données d'emails."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        """
        Initialise le data loader.
        
        Args:
            model_name: Nom du modèle HuggingFace pour le tokenizer
            max_length: Longueur maximale des séquences
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def clean_html(self, text: str) -> str:
        """
        Nettoie le HTML d'un texte.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte nettoyé
        """
        if pd.isna(text):
            return ""
        
        # Supprimer les balises HTML
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text()
        
        # Supprimer les caractères de contrôle et normaliser les espaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        return clean_text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalise les caractères Unicode.
        
        Args:
            text: Texte à normaliser
            
        Returns:
            Texte normalisé
        """
        if pd.isna(text):
            return ""
        
        # Convertir les caractères Unicode en ASCII
        normalized = unidecode(text)
        
        # Nettoyer les caractères spéciaux restants
        normalized = re.sub(r'[^\w\s\.\,\!\?\-\@\:\/]', '', normalized)
        
        return normalized
    
    def preprocess_text(self, text: str) -> str:
        """
        Préprocesse un texte complet.
        
        Args:
            text: Texte à préprocesser
            
        Returns:
            Texte préprocessé
        """
        # Nettoyage HTML
        clean_text = self.clean_html(text)
        
        # Normalisation Unicode
        normalized_text = self.normalize_unicode(clean_text)
        
        # Conversion en minuscules
        normalized_text = normalized_text.lower()
        
        # Suppression des espaces multiples
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        return normalized_text
    
    def load_dataset(self, file_path: str, text_column: str = "text", 
                    label_column: str = "label") -> Tuple[List[str], List[int]]:
        """
        Charge un dataset depuis un fichier CSV.
        
        Args:
            file_path: Chemin vers le fichier CSV
            text_column: Nom de la colonne contenant le texte
            label_column: Nom de la colonne contenant les labels
            
        Returns:
            Tuple (textes, labels)
        """
        logger.info(f"Chargement du dataset depuis {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset chargé: {len(df)} échantillons")
            
            # Vérifier les colonnes requises
            if text_column not in df.columns:
                raise ValueError(f"Colonne '{text_column}' non trouvée dans le dataset")
            if label_column not in df.columns:
                raise ValueError(f"Colonne '{label_column}' non trouvée dans le dataset")
            
            # Préprocesser les textes
            texts = []
            for text in df[text_column]:
                preprocessed = self.preprocess_text(str(text))
                texts.append(preprocessed)
            
            # Convertir les labels en entiers (0: légitimes, 1: phishing)
            labels = []
            for label in df[label_column]:
                if isinstance(label, str):
                    # Mapper les labels textuels vers des entiers
                    if label.lower() in ['phishing', 'spam', '1', 'true']:
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    labels.append(int(label))
            
            logger.info(f"Textes préprocessés: {len(texts)}")
            logger.info(f"Distribution des labels: {np.bincount(labels)}")
            
            return texts, labels
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise
    
    def tokenize_texts(self, texts: List[str]) -> Dict:
        """
        Tokenise une liste de textes.
        
        Args:
            texts: Liste des textes à tokeniser
            
        Returns:
            Dictionnaire avec les tokens, attention masks, etc.
        """
        logger.info(f"Tokenisation de {len(texts)} textes")
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return encoded
    
    def prepare_dataset(self, file_path: str, test_size: float = 0.2, 
                       random_state: int = 42) -> Tuple[Dict, Dict, List[int], List[int]]:
        """
        Prépare le dataset complet pour l'entraînement.
        
        Args:
            file_path: Chemin vers le fichier de données
            test_size: Proportion du test set
            random_state: Graine aléatoire
            
        Returns:
            Tuple (train_encodings, val_encodings, train_labels, val_labels)
        """
        # Charger les données
        texts, labels = self.load_dataset(file_path)
        
        # Split train/validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        logger.info(f"Train set: {len(train_texts)} échantillons")
        logger.info(f"Validation set: {len(val_texts)} échantillons")
        
        # Tokeniser
        train_encodings = self.tokenize_texts(train_texts)
        val_encodings = self.tokenize_texts(val_texts)
        
        return train_encodings, val_encodings, train_labels, val_labels


if __name__ == "__main__":
    # Test du data loader
    loader = EmailDataLoader()
    
    # Exemple de test avec des données fictives
    test_texts = [
        "<h1>Urgent!</h1> Your account will be suspended. Click <a href='http://fake-bank.com'>here</a>",
        "Hello, this is a normal business email about our meeting tomorrow.",
        "CONGRATULATIONS!!! You won $1,000,000! Click here to claim your prize!"
    ]
    
    for text in test_texts:
        preprocessed = loader.preprocess_text(text)
        print(f"Original: {text[:50]}...")
        print(f"Preprocessed: {preprocessed}")
        print("-" * 50)
