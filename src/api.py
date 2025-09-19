"""
API FastAPI pour le modèle de détection de phishing.
Fournit un endpoint /predict pour classifier les emails.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uvicorn

# Imports locaux
from data_loader import EmailDataLoader
from features import FeatureExtractor
from model import PhishingDetectorModel

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Phishing Email Detection API",
    description="API pour la détection d'emails de phishing utilisant DistilBERT et des features adversariales",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le modèle
model = None
data_loader = None
feature_extractor = None
device = None
model_config = None


class EmailRequest(BaseModel):
    """Modèle de requête pour la prédiction."""
    text: str = Field(..., description="Contenu de l'email à analyser", min_length=1, max_length=10000)
    return_probabilities: bool = Field(default=True, description="Retourner les probabilités détaillées")
    return_features: bool = Field(default=False, description="Retourner les features extraites")


class PredictionResponse(BaseModel):
    """Modèle de réponse pour la prédiction."""
    prediction: str = Field(..., description="Classification: 'phishing' ou 'legitimate'")
    confidence: float = Field(..., description="Score de confiance (0-1)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilités pour chaque classe")
    features: Optional[Dict[str, float]] = Field(None, description="Features extraites")
    processing_time_ms: float = Field(..., description="Temps de traitement en millisecondes")
    timestamp: str = Field(..., description="Timestamp de la prédiction")


class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check."""
    status: str
    model_loaded: bool
    model_info: Optional[Dict] = None
    timestamp: str


class BatchEmailRequest(BaseModel):
    """Modèle pour les prédictions en batch."""
    emails: List[str] = Field(..., description="Liste des emails à analyser", max_items=100)
    return_probabilities: bool = Field(default=True, description="Retourner les probabilités détaillées")


class BatchPredictionResponse(BaseModel):
    """Modèle de réponse pour les prédictions en batch."""
    predictions: List[PredictionResponse]
    total_count: int
    processing_time_ms: float
    timestamp: str


def load_model_and_components(model_path: str, config_path: str):
    """
    Charge le modèle et les composants nécessaires.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        config_path: Chemin vers la configuration
    """
    global model, data_loader, feature_extractor, device, model_config
    
    try:
        logger.info("Chargement du modèle et des composants...")
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device utilisé: {device}")
        
        # Charger la configuration
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Initialiser les composants
        data_loader = EmailDataLoader(model_name=model_config['model_config']['model_name'])
        feature_extractor = FeatureExtractor()
        
        # Initialiser et charger le modèle
        model = PhishingDetectorModel(
            model_name=model_config['model_config']['model_name'],
            num_lexical_features=model_config['model_config']['num_lexical_features']
        )
        
        # Charger les poids du modèle
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info("Modèle et composants chargés avec succès")
        logger.info(f"Taille du modèle: {model.get_model_size()}")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise


def predict_single_email(text: str, 
                        return_probabilities: bool = True, 
                        return_features: bool = False) -> Dict:
    """
    Prédit si un email est du phishing.
    
    Args:
        text: Contenu de l'email
        return_probabilities: Retourner les probabilités
        return_features: Retourner les features
        
    Returns:
        Dictionnaire avec la prédiction
    """
    start_time = datetime.now()
    
    try:
        # Préprocesser le texte
        preprocessed_text = data_loader.preprocess_text(text)
        
        # Tokeniser
        encoding = data_loader.tokenize_texts([preprocessed_text])
        
        # Extraire les features
        features = feature_extractor.extract_all_features(preprocessed_text)
        
        # Préparer les tenseurs
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        lexical_features = torch.tensor([features], dtype=torch.float32).to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, lexical_features)
            probabilities = outputs['probabilities'].cpu().numpy()[0]
            prediction_idx = np.argmax(probabilities)
        
        # Interpréter les résultats
        class_names = ['legitimate', 'phishing']
        prediction = class_names[prediction_idx]
        confidence = float(probabilities[prediction_idx])
        
        # Construire la réponse
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_names[i]: float(prob) for i, prob in enumerate(probabilities)
            }
        
        if return_features:
            feature_names = feature_extractor.get_feature_names()
            result['features'] = {
                name: float(value) for name, value in zip(feature_names, features)
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Événement de démarrage pour charger le modèle."""
    # Ces chemins peuvent être configurés via des variables d'environnement
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    config_path = os.getenv("CONFIG_PATH", "models/model_config.json")
    
    if os.path.exists(model_path) and os.path.exists(config_path):
        try:
            load_model_and_components(model_path, config_path)
            logger.info("API prête à recevoir des requêtes")
        except Exception as e:
            logger.error(f"Impossible de charger le modèle: {e}")
            logger.warning("L'API démarrera sans modèle chargé")
    else:
        logger.warning(f"Fichiers de modèle non trouvés: {model_path}, {config_path}")
        logger.warning("L'API démarrera sans modèle chargé")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint racine."""
    return {
        "message": "Phishing Email Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de vérification de l'état de santé."""
    model_info = None
    if model is not None and model_config is not None:
        model_info = {
            "model_name": model_config['model_config']['model_name'],
            "num_features": model_config['model_config']['num_lexical_features'],
            "device": str(device),
            "model_size": model.get_model_size() if model else None
        }
    
    return HealthResponse(
        status="healthy" if model is not None else "no_model_loaded",
        model_loaded=model is not None,
        model_info=model_info,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_email(request: EmailRequest):
    """
    Prédit si un email est du phishing.
    
    Args:
        request: Requête contenant l'email à analyser
        
    Returns:
        Résultat de la prédiction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Vérifiez la configuration.")
    
    try:
        result = predict_single_email(
            request.text, 
            request.return_probabilities, 
            request.return_features
        )
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_emails_batch(request: BatchEmailRequest):
    """
    Prédit plusieurs emails en batch.
    
    Args:
        request: Requête contenant la liste des emails
        
    Returns:
        Résultats des prédictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Vérifiez la configuration.")
    
    if len(request.emails) == 0:
        raise HTTPException(status_code=400, detail="Liste d'emails vide")
    
    start_time = datetime.now()
    predictions = []
    
    try:
        for email_text in request.emails:
            result = predict_single_email(
                email_text, 
                request.return_probabilities, 
                return_features=False  # Pas de features en batch pour limiter la taille
            )
            predictions.append(PredictionResponse(**result))
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            processing_time_ms=total_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement batch: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement batch: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Retourne les informations sur le modèle chargé."""
    if model is None or model_config is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_config": model_config['model_config'],
        "training_config": model_config.get('training_config', {}),
        "final_metrics": model_config.get('final_metrics', {}),
        "feature_names": model_config.get('feature_names', []),
        "model_size": model.get_model_size(),
        "device": str(device),
        "timestamp": model_config.get('timestamp', 'unknown')
    }


@app.post("/model/reload")
async def reload_model(model_path: str = None, config_path: str = None):
    """
    Recharge le modèle avec de nouveaux chemins.
    
    Args:
        model_path: Nouveau chemin vers le modèle
        config_path: Nouveau chemin vers la configuration
    """
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "models/model_config.json")
    
    try:
        load_model_and_components(model_path, config_path)
        return {
            "message": "Modèle rechargé avec succès",
            "model_path": model_path,
            "config_path": config_path,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du rechargement: {str(e)}")


# Fonction pour démarrer le serveur
def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Démarre le serveur FastAPI.
    
    Args:
        host: Adresse d'écoute
        port: Port d'écoute
        reload: Mode de rechargement automatique
    """
    uvicorn.run("api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Serveur API de détection de phishing')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Adresse d\'écoute')
    parser.add_argument('--port', type=int, default=8000, help='Port d\'écoute')
    parser.add_argument('--reload', action='store_true', help='Mode rechargement automatique')
    parser.add_argument('--model_path', type=str, help='Chemin vers le modèle')
    parser.add_argument('--config_path', type=str, help='Chemin vers la configuration')
    
    args = parser.parse_args()
    
    # Configurer les variables d'environnement si spécifiées
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.config_path:
        os.environ["CONFIG_PATH"] = args.config_path
    
    logger.info(f"Démarrage du serveur sur {args.host}:{args.port}")
    start_server(args.host, args.port, args.reload)
