# Asthma Prediction

Simple ML project to predict Astma using features like family history, FeNO level, smoking status etc. The model is served via a FastAPI backend.

## Setup

1. Install dependencies:  
   `pip install -r requirements.txt`

2. Run the API server:  
   `uvicorn app:app --reload`

3. Test the API at:  
   `http://127.0.0.1:8000/docs`

## Usage

Send a POST request to `/predict` with details to get a prediction.

---

Made with Python, scikit-learn, and FastAPI.
