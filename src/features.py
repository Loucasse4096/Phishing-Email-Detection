"""
Module pour l'extraction de features lexicales des emails.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from collections import Counter
import math
import ipaddress
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Classe pour extraire des features lexicales des emails."""
    
    def __init__(self):
        """Initialise l'extracteur de features."""
        # Patterns regex pour différents types de contenu
        self.url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1-?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')
        self.ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        
        # Mots-clés suspects typiques du phishing
        self.suspicious_keywords = [
            'urgent', 'immediate', 'action required', 'suspended', 'verify', 'confirm',
            'click here', 'update', 'expire', 'limited time', 'congratulations',
            'winner', 'prize', 'free', 'bonus', 'claim', 'refund', 'tax',
            'security', 'alert', 'warning', 'locked', 'unauthorized', 'suspicious'
        ]
        
        # Domaines légitimes courants
        self.legitimate_domains = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
            'amazon.com', 'paypal.com', 'ebay.com', 'microsoft.com',
            'google.com', 'apple.com', 'facebook.com', 'twitter.com'
        ]
    
    def extract_url_features(self, text: str) -> Dict[str, float]:
        """
        Extrait les features liées aux URLs.
        
        Args:
            text: Texte de l'email
            
        Returns:
            Dictionnaire des features URL
        """
        urls = self.url_pattern.findall(text)
        
        features = {
            'num_urls': len(urls),
            'url_ratio': len(urls) / max(len(text.split()), 1),
            'has_suspicious_url': 0,
            'has_ip_url': 0,
            'has_shortened_url': 0,
            'avg_url_length': 0
        }
        
        if urls:
            # Longueur moyenne des URLs
            features['avg_url_length'] = np.mean([len(url) for url in urls])
            
            # Vérifier les URLs suspectes
            suspicious_patterns = ['bit.ly', 'tinyurl', 'goo.gl', 't.co']
            for url in urls:
                # URLs raccourcies
                if any(pattern in url.lower() for pattern in suspicious_patterns):
                    features['has_shortened_url'] = 1
                
                # URLs avec adresses IP
                try:
                    # Extraire le domaine de l'URL
                    domain_match = re.search(r'://([^/]+)', url)
                    if domain_match:
                        domain = domain_match.group(1)
                        # Vérifier si c'est une adresse IP
                        try:
                            ipaddress.ip_address(domain)
                            features['has_ip_url'] = 1
                        except ValueError:
                            pass
                        
                        # Vérifier si le domaine n'est pas dans les domaines légitimes
                        if not any(legit in domain.lower() for legit in self.legitimate_domains):
                            features['has_suspicious_url'] = 1
                except:
                    pass
        
        return features
    
    def extract_text_statistics(self, text: str) -> Dict[str, float]:
        """
        Extrait les statistiques textuelles.
        
        Args:
            text: Texte de l'email
            
        Returns:
            Dictionnaire des features statistiques
        """
        words = text.split()
        
        features = {
            'text_length': len(text),
            'num_words': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'num_sentences': len(re.split(r'[.!?]+', text)),
            'num_paragraphs': len(text.split('\n\n')),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        }
        
        return features
    
    def extract_suspicious_content(self, text: str) -> Dict[str, float]:
        """
        Extrait les features liées au contenu suspect.
        
        Args:
            text: Texte de l'email
            
        Returns:
            Dictionnaire des features de contenu suspect
        """
        text_lower = text.lower()
        
        features = {
            'num_suspicious_words': 0,
            'suspicious_word_ratio': 0,
            'has_exclamation': 1 if '!' in text else 0,
            'num_exclamations': text.count('!'),
            'has_all_caps_words': 0,
            'num_money_references': 0,
            'has_urgency_words': 0
        }
        
        # Compter les mots suspects
        suspicious_count = 0
        for keyword in self.suspicious_keywords:
            if keyword in text_lower:
                suspicious_count += text_lower.count(keyword)
        
        features['num_suspicious_words'] = suspicious_count
        features['suspicious_word_ratio'] = suspicious_count / max(len(text.split()), 1)
        
        # Mots en majuscules
        words = text.split()
        caps_words = [word for word in words if word.isupper() and len(word) > 2]
        features['has_all_caps_words'] = 1 if caps_words else 0
        
        # Références monétaires
        money_patterns = [r'\$\d+', r'\d+\s*dollars?', r'\d+\s*euros?', r'€\d+', r'£\d+']
        for pattern in money_patterns:
            features['num_money_references'] += len(re.findall(pattern, text_lower))
        
        # Mots d'urgence
        urgency_words = ['urgent', 'immediate', 'asap', 'hurry', 'quickly', 'now', 'today']
        features['has_urgency_words'] = 1 if any(word in text_lower for word in urgency_words) else 0
        
        return features
    
    def extract_contact_info(self, text: str) -> Dict[str, float]:
        """
        Extrait les features liées aux informations de contact.
        
        Args:
            text: Texte de l'email
            
        Returns:
            Dictionnaire des features de contact
        """
        features = {
            'num_emails': len(self.email_pattern.findall(text)),
            'num_phones': len(self.phone_pattern.findall(text)),
            'num_ips': len(self.ip_pattern.findall(text)),
            'has_multiple_contacts': 0
        }
        
        total_contacts = features['num_emails'] + features['num_phones']
        features['has_multiple_contacts'] = 1 if total_contacts > 2 else 0
        
        return features
    
    def calculate_text_entropy(self, text: str) -> float:
        """
        Calcule l'entropie du texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Entropie du texte
        """
        if not text:
            return 0
        
        # Compter la fréquence des caractères
        char_counts = Counter(text.lower())
        text_length = len(text)
        
        # Calculer l'entropie
        entropy = 0
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def extract_all_features(self, text: str) -> np.ndarray:
        """
        Extrait toutes les features d'un texte.
        
        Args:
            text: Texte de l'email
            
        Returns:
            Vecteur numpy des features
        """
        if pd.isna(text) or text == "":
            # Retourner un vecteur de zéros si le texte est vide
            return np.zeros(20)
        
        # Extraire toutes les catégories de features
        url_features = self.extract_url_features(text)
        text_stats = self.extract_text_statistics(text)
        suspicious_content = self.extract_suspicious_content(text)
        contact_info = self.extract_contact_info(text)
        
        # Ajouter l'entropie
        entropy = self.calculate_text_entropy(text)
        
        # Combiner toutes les features dans un vecteur
        feature_vector = []
        
        # Features URL (6 features)
        feature_vector.extend([
            url_features['num_urls'],
            url_features['url_ratio'],
            url_features['has_suspicious_url'],
            url_features['has_ip_url'],
            url_features['has_shortened_url'],
            url_features['avg_url_length']
        ])
        
        # Features statistiques (8 features)
        feature_vector.extend([
            text_stats['text_length'],
            text_stats['num_words'],
            text_stats['avg_word_length'],
            text_stats['num_sentences'],
            text_stats['uppercase_ratio'],
            text_stats['digit_ratio'],
            text_stats['special_char_ratio'],
            entropy
        ])
        
        # Features contenu suspect (6 features)
        feature_vector.extend([
            suspicious_content['num_suspicious_words'],
            suspicious_content['suspicious_word_ratio'],
            suspicious_content['has_exclamation'],
            suspicious_content['num_exclamations'],
            suspicious_content['has_all_caps_words'],
            suspicious_content['num_money_references']
        ])
        
        return np.array(feature_vector, dtype=np.float32)
    
    def extract_batch_features(self, texts: List[str]) -> np.ndarray:
        """
        Extrait les features d'une liste de textes.
        
        Args:
            texts: Liste des textes
            
        Returns:
            Matrice numpy des features (n_samples, n_features)
        """
        logger.info(f"Extraction des features pour {len(texts)} textes")
        
        features_list = []
        for text in texts:
            features = self.extract_all_features(text)
            features_list.append(features)
        
        feature_matrix = np.vstack(features_list)
        logger.info(f"Features extraites: {feature_matrix.shape}")
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """
        Retourne les noms des features.
        
        Returns:
            Liste des noms de features
        """
        return [
            # URL features
            'num_urls', 'url_ratio', 'has_suspicious_url', 'has_ip_url', 
            'has_shortened_url', 'avg_url_length',
            # Text statistics
            'text_length', 'num_words', 'avg_word_length', 'num_sentences',
            'uppercase_ratio', 'digit_ratio', 'special_char_ratio', 'text_entropy',
            # Suspicious content
            'num_suspicious_words', 'suspicious_word_ratio', 'has_exclamation',
            'num_exclamations', 'has_all_caps_words', 'num_money_references'
        ]


if __name__ == "__main__":
    # Test de l'extracteur de features
    extractor = FeatureExtractor()
    
    test_emails = [
        "URGENT! Your PayPal account has been suspended. Click here: http://fake-paypal.com to verify immediately!",
        "Hello John, I hope this email finds you well. Let's schedule our meeting for next week.",
        "CONGRATULATIONS!!! You've won $1,000,000! Call 1-800-FAKE-NUM or visit bit.ly/fakewinner to claim your prize!"
    ]
    
    for i, email in enumerate(test_emails):
        print(f"\n=== Email {i+1} ===")
        print(f"Text: {email[:60]}...")
        
        features = extractor.extract_all_features(email)
        feature_names = extractor.get_feature_names()
        
        print("Features:")
        for name, value in zip(feature_names, features):
            if value > 0:
                print(f"  {name}: {value:.3f}")
