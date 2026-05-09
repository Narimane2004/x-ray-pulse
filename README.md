# PneumoScan AI 🫁

Détection IA de la pneumonie sur radiographie thoracique — frontend 3D moderne + backend FastAPI prêt à brancher sur votre modèle (TensorFlow `.h5` ou PyTorch `.pth`).

## Stack

- **Frontend** : React + TanStack Start, TailwindCSS v4, React Three Fiber (3D)
- **Backend** : FastAPI + Pillow + (TensorFlow ou PyTorch)
- **Communication** : `POST /predict` (multipart image) → `{ prediction, confidence }`

## Lancer le frontend

```bash
npm install
npm run dev
```

Par défaut, le frontend tourne en **mode démo** (prédiction simulée) tant que la variable d'environnement `BACKEND_URL` n'est pas définie.

Pour brancher le backend réel :

```bash
# .env
BACKEND_URL=http://localhost:8000
```

## Lancer le backend

```bash
cd backend
pip install -r requirements.txt
# décommentez tensorflow OU torch dans requirements.txt selon votre modèle
pip install tensorflow            # si model.h5
# pip install torch torchvision   # si model.pth

# Déposez votre modèle entraîné :
#   backend/model.h5     (Keras)
# OU backend/model.pth   (PyTorch state_dict)

python app.py
# -> http://localhost:8000
```

## Adapter l'architecture PyTorch

Si votre `.pth` n'est pas un DenseNet201 binaire, modifiez la section `_load_torch()` dans `backend/app.py` pour reconstruire votre architecture exacte avant `load_state_dict`.

## Préprocessing

Les images sont redimensionnées en **224×224** et normalisées dans `[0, 1]` — identique au pipeline du notebook d'entraînement Kaggle fourni.

## Disclaimer

Outil d'aide au diagnostic à but pédagogique. **Ne remplace pas un avis médical professionnel.**
