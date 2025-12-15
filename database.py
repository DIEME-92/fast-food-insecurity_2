from sqlalchemy import create_engine, Column, Integer, String, Float, TIMESTAMP
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
import os

# Charger le fichier .env avec chemin absolu
load_dotenv(dotenv_path="D:/PROJET_DIT-20250506T153458Z-001/MES_PROJETS/.env")

# Récupérer la chaîne de connexion
DATABASE_URL = os.getenv("DATABASE_URL")
print("Chaine brute :", repr(DATABASE_URL))  # Sans accent

# Vérification de la chaîne
if DATABASE_URL is None:
    raise ValueError("La variable DATABASE_URL est introuvable. Verifie ton fichier .env.")  # Sans accent

# Créer l'engine SQLAlchemy
engine = create_engine(DATABASE_URL)

# Créer une session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Déclarer le modèle
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(TIMESTAMP, nullable=False)
    niveau = Column(String(50), nullable=False)
    score = Column(Float, nullable=False)

# Créer les tables si elles n'existent pas
try:
    Base.metadata.create_all(bind=engine)
    print("Connexion reussie et table creee")  # Sans accent
except Exception as e:
    print("Erreur de connexion ou de creation de table")
    print(e)

from datetime import datetime

# Créer une session
session = SessionLocal()

# Créer une prédiction
nouvelle_prediction = PredictionLog(
    date=datetime.utcnow(),     # Date actuelle en UTC
    niveau="eleve",             # Niveau de risque (ex: "faible", "moyen", "eleve")
    score=0.87                  # Score de prédiction (float)
)

# Ajouter et enregistrer
session.add(nouvelle_prediction)
session.commit()
session.close()

print("Prédiction enregistrée ✅")
