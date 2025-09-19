# Dockerfile pour le système de détection de phishing
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models/best_model.pth
ENV CONFIG_PATH=/app/models/model_config.json

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Créer les dossiers nécessaires
RUN mkdir -p /app/src /app/models /app/data /app/logs

# Copier le code source
COPY src/ /app/src/

# Copier les modèles pré-entraînés (si disponibles)
# COPY models/ /app/models/

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Exposer le port
EXPOSE 8000

# Commande de santé
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Point d'entrée par défaut
CMD ["python", "src/api.py", "--host", "0.0.0.0", "--port", "8000"]
