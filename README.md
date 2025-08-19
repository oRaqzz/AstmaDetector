Simple ML project to predict Asthma using features like Family history, Smoking status, FeNO level etc. The model is served via a FastAPI backend.
Setup

    Install dependencies:
    pip install -r requirements.txt

    Run the API server:
    uvicorn app:app --reload

    Test the API at:
    http://127.0.0.1:8000/docs

Usage

Send a POST request to /predict with house details to get a prediction.

Made with Python, scikit-learn, and FastAPI.
