"""
Module pour la génération d'attaques adversariales sur les emails.
"""

import random
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import Levenshtein
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdversarialAttacker:
    """Classe pour générer des attaques adversariales sur les emails."""
    
    def __init__(self, seed: int = 42):
        """
        Initialise l'attaquant adversarial.
        
        Args:
            seed: Graine aléatoire pour la reproductibilité
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Dictionnaire de substitutions homoglyphes
        self.homoglyphs = {
            'a': ['а', 'ɑ', 'α', '@'],  # cyrillique, grec, arobase
            'e': ['е', 'ε', '3'],       # cyrillique, grec
            'o': ['о', 'ο', '0'],       # cyrillique, grec, zéro
            'p': ['р', 'ρ'],            # cyrillique, grec
            'c': ['с', 'ϲ'],            # cyrillique, grec
            'x': ['х', 'χ'],            # cyrillique, grec
            'y': ['у', 'γ'],            # cyrillique, grec
            'i': ['і', 'ι', '1', 'l'],  # cyrillique, grec, un, L minuscule
            'n': ['η'],                 # grec
            's': ['ѕ', 'σ', '$'],       # cyrillique, grec, dollar
            'h': ['һ'],                 # cyrillique
            'b': ['Ь'],                 # cyrillique
            'g': ['ɡ'],                 # latin g sans serif
            'l': ['ӏ', '1', 'I'],       # cyrillique, un, I majuscule
            'm': ['м'],                 # cyrillique
            'r': ['г'],                 # cyrillique
            't': ['т'],                 # cyrillique
            'u': ['υ'],                 # grec
            'v': ['ν', 'v'],            # grec nu
            'w': ['ω'],                 # grec omega
            'z': ['ζ']                  # grec zeta
        }
        
        # Patterns pour l'obfuscation d'URLs et domaines
        self.url_obfuscation_patterns = {
            'paypal': ['paypa1', 'payp4l', 'p4ypal', 'paypaI', 'payp@l'],
            'amazon': ['amaz0n', 'amazom', 'amazon', 'am4zon', '@mazon'],
            'google': ['g00gle', 'googIe', 'g0ogle', 'goog1e', 'g@@gle'],
            'microsoft': ['micr0soft', 'microsoFt', 'micr0s0ft', 'micr@soft'],
            'apple': ['app1e', 'appl3', '@pple', 'app|e'],
            'bank': ['b@nk', 'b4nk', 'banк', 'b@n|<'],
            'secure': ['secur3', 's3cure', 'secur€', 's€cure'],
            'account': ['acc0unt', 'acc@unt', '4ccount', '@ccount'],
            'login': ['l0gin', 'log1n', 'l@gin', '|ogin'],
            'verify': ['v3rify', 'ver1fy', 'v€rify', 'verif¥']
        }
    
    def character_swap_attack(self, text: str, swap_rate: float = 0.02) -> str:
        """
        Attaque par échange de caractères adjacents.
        
        Args:
            text: Texte original
            swap_rate: Taux d'échange des caractères (0-1)
            
        Returns:
            Texte avec caractères échangés
        """
        if len(text) < 2:
            return text
        
        text_list = list(text)
        num_swaps = int(len(text) * swap_rate)
        
        for _ in range(num_swaps):
            # Choisir une position aléatoire (pas la dernière)
            pos = random.randint(0, len(text_list) - 2)
            
            # Échanger avec le caractère suivant
            if text_list[pos].isalpha() and text_list[pos + 1].isalpha():
                text_list[pos], text_list[pos + 1] = text_list[pos + 1], text_list[pos]
        
        return ''.join(text_list)
    
    def character_deletion_attack(self, text: str, delete_rate: float = 0.01) -> str:
        """
        Attaque par suppression de caractères.
        
        Args:
            text: Texte original
            delete_rate: Taux de suppression des caractères (0-1)
            
        Returns:
            Texte avec caractères supprimés
        """
        if len(text) == 0:
            return text
        
        text_list = list(text)
        num_deletions = int(len(text) * delete_rate)
        
        # Supprimer aléatoirement des caractères
        indices_to_delete = random.sample(range(len(text_list)), 
                                        min(num_deletions, len(text_list)))
        
        # Supprimer en ordre décroissant pour maintenir les indices
        for idx in sorted(indices_to_delete, reverse=True):
            if text_list[idx].isalpha():  # Ne supprimer que les lettres
                text_list.pop(idx)
        
        return ''.join(text_list)
    
    def character_insertion_attack(self, text: str, insert_rate: float = 0.01) -> str:
        """
        Attaque par insertion de caractères.
        
        Args:
            text: Texte original
            insert_rate: Taux d'insertion des caractères (0-1)
            
        Returns:
            Texte avec caractères insérés
        """
        text_list = list(text)
        num_insertions = int(len(text) * insert_rate)
        
        for _ in range(num_insertions):
            # Choisir une position aléatoire
            pos = random.randint(0, len(text_list))
            
            # Insérer un caractère aléatoire (lettre minuscule)
            char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
            text_list.insert(pos, char_to_insert)
        
        return ''.join(text_list)
    
    def homoglyph_attack(self, text: str, substitution_rate: float = 0.05) -> str:
        """
        Attaque par substitution d'homoglyphes.
        
        Args:
            text: Texte original
            substitution_rate: Taux de substitution des caractères (0-1)
            
        Returns:
            Texte avec homoglyphes substitués
        """
        text_list = list(text.lower())
        num_substitutions = int(len(text) * substitution_rate)
        
        # Trouver les positions des caractères substituables
        substitutable_positions = []
        for i, char in enumerate(text_list):
            if char in self.homoglyphs:
                substitutable_positions.append(i)
        
        # Effectuer les substitutions
        positions_to_substitute = random.sample(
            substitutable_positions, 
            min(num_substitutions, len(substitutable_positions))
        )
        
        for pos in positions_to_substitute:
            original_char = text_list[pos]
            if original_char in self.homoglyphs:
                # Choisir un homoglyphe aléatoire
                replacement = random.choice(self.homoglyphs[original_char])
                text_list[pos] = replacement
        
        return ''.join(text_list)
    
    def url_obfuscation_attack(self, text: str) -> str:
        """
        Attaque par obfuscation d'URLs et de domaines.
        
        Args:
            text: Texte original
            
        Returns:
            Texte avec URLs obfusquées
        """
        result = text.lower()
        
        # Appliquer les obfuscations définies
        for original, obfuscations in self.url_obfuscation_patterns.items():
            if original in result:
                # Choisir une obfuscation aléatoire
                obfuscated = random.choice(obfuscations)
                # Remplacer la première occurrence
                result = result.replace(original, obfuscated, 1)
        
        return result
    
    def combined_typo_attack(self, text: str, intensity: float = 0.1) -> str:
        """
        Attaque combinée avec plusieurs types de typos.
        
        Args:
            text: Texte original
            intensity: Intensité de l'attaque (0-1)
            
        Returns:
            Texte avec typos combinés
        """
        result = text
        
        # Appliquer différents types d'attaques avec des probabilités
        if random.random() < intensity:
            result = self.character_swap_attack(result, intensity * 0.5)
        
        if random.random() < intensity:
            result = self.character_deletion_attack(result, intensity * 0.3)
        
        if random.random() < intensity:
            result = self.character_insertion_attack(result, intensity * 0.2)
        
        if random.random() < intensity * 2:  # Plus probable
            result = self.homoglyph_attack(result, intensity * 0.8)
        
        if random.random() < intensity:
            result = self.url_obfuscation_attack(result)
        
        return result
    
    def generate_adversarial_examples(self, texts: List[str], 
                                    attack_types: List[str] = None,
                                    intensity: float = 0.1) -> List[str]:
        """
        Génère des exemples adversariaux pour une liste de textes.
        
        Args:
            texts: Liste des textes originaux
            attack_types: Types d'attaques à appliquer
            intensity: Intensité des attaques
            
        Returns:
            Liste des textes adversariaux
        """
        if attack_types is None:
            attack_types = ['swap', 'delete', 'insert', 'homoglyph', 'url_obfuscation', 'combined']
        
        logger.info(f"Génération d'exemples adversariaux pour {len(texts)} textes")
        logger.info(f"Types d'attaques: {attack_types}")
        
        adversarial_texts = []
        
        for text in texts:
            # Choisir un type d'attaque aléatoire
            attack_type = random.choice(attack_types)
            
            if attack_type == 'swap':
                adversarial_text = self.character_swap_attack(text, intensity)
            elif attack_type == 'delete':
                adversarial_text = self.character_deletion_attack(text, intensity)
            elif attack_type == 'insert':
                adversarial_text = self.character_insertion_attack(text, intensity)
            elif attack_type == 'homoglyph':
                adversarial_text = self.homoglyph_attack(text, intensity)
            elif attack_type == 'url_obfuscation':
                adversarial_text = self.url_obfuscation_attack(text)
            elif attack_type == 'combined':
                adversarial_text = self.combined_typo_attack(text, intensity)
            else:
                adversarial_text = text
            
            adversarial_texts.append(adversarial_text)
        
        return adversarial_texts
    
    def evaluate_attack_success(self, original_texts: List[str], 
                              adversarial_texts: List[str]) -> Dict[str, float]:
        """
        Évalue le succès des attaques en calculant la distance d'édition.
        
        Args:
            original_texts: Textes originaux
            adversarial_texts: Textes adversariaux
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        if len(original_texts) != len(adversarial_texts):
            raise ValueError("Les listes doivent avoir la même longueur")
        
        edit_distances = []
        similarity_scores = []
        
        for orig, adv in zip(original_texts, adversarial_texts):
            # Distance d'édition de Levenshtein
            edit_dist = Levenshtein.distance(orig, adv)
            edit_distances.append(edit_dist)
            
            # Score de similarité (0-1)
            similarity = 1 - (edit_dist / max(len(orig), len(adv), 1))
            similarity_scores.append(similarity)
        
        metrics = {
            'avg_edit_distance': np.mean(edit_distances),
            'avg_similarity': np.mean(similarity_scores),
            'min_similarity': np.min(similarity_scores),
            'max_edit_distance': np.max(edit_distances),
            'perturbation_rate': 1 - np.mean(similarity_scores)
        }
        
        return metrics
    
    def create_adversarial_dataset(self, texts: List[str], labels: List[int],
                                 adversarial_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """
        Crée un dataset augmenté avec des exemples adversariaux.
        
        Args:
            texts: Textes originaux
            labels: Labels originaux
            adversarial_ratio: Ratio d'exemples adversariaux à ajouter
            
        Returns:
            Tuple (textes_augmentés, labels_augmentés)
        """
        logger.info(f"Création d'un dataset adversarial avec ratio {adversarial_ratio}")
        
        # Calculer le nombre d'exemples adversariaux à générer
        num_adversarial = int(len(texts) * adversarial_ratio)
        
        # Sélectionner aléatoirement des textes pour les attaques
        indices = random.sample(range(len(texts)), num_adversarial)
        selected_texts = [texts[i] for i in indices]
        selected_labels = [labels[i] for i in indices]
        
        # Générer les exemples adversariaux
        adversarial_texts = self.generate_adversarial_examples(selected_texts)
        
        # Combiner avec le dataset original
        augmented_texts = texts + adversarial_texts
        augmented_labels = labels + selected_labels
        
        logger.info(f"Dataset augmenté: {len(augmented_texts)} exemples "
                   f"({len(texts)} originaux + {len(adversarial_texts)} adversariaux)")
        
        return augmented_texts, augmented_labels


if __name__ == "__main__":
    # Test de l'attaquant adversarial
    attacker = AdversarialAttacker()
    
    test_emails = [
        "Urgent! Your PayPal account has been suspended. Click here to verify: https://paypal.com/verify",
        "Hello, this is a legitimate business email about our meeting.",
        "CONGRATULATIONS! You won $1000! Visit our secure website: https://amazon.com/prize"
    ]
    
    print("=== Test des attaques adversariales ===\n")
    
    for i, email in enumerate(test_emails):
        print(f"Email {i+1} original:")
        print(f"'{email}'\n")
        
        # Test de chaque type d'attaque
        attacks = {
            'Character Swap': attacker.character_swap_attack(email),
            'Character Deletion': attacker.character_deletion_attack(email),
            'Homoglyph': attacker.homoglyph_attack(email),
            'URL Obfuscation': attacker.url_obfuscation_attack(email),
            'Combined Attack': attacker.combined_typo_attack(email)
        }
        
        for attack_name, attacked_email in attacks.items():
            print(f"{attack_name}:")
            print(f"'{attacked_email}'")
            
            # Calculer la similarité
            similarity = 1 - (Levenshtein.distance(email, attacked_email) / max(len(email), len(attacked_email)))
            print(f"Similarité: {similarity:.3f}\n")
        
        print("-" * 80)
