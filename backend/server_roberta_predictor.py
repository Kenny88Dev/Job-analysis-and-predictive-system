import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import shap
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf 
import math

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # .../backend
ROOT_DIR = os.path.dirname(BASE_DIR)                  # .../ (Root)

# Robust path finding for Frontend
if os.path.exists(os.path.join(ROOT_DIR, "frontend")):
    FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
elif os.path.exists(os.path.join(ROOT_DIR, "FRONTEND")):
    FRONTEND_DIR = os.path.join(ROOT_DIR, "FRONTEND")
else:
    FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# Robust path finding for Dataset
if os.path.exists(os.path.join(ROOT_DIR, "ONET_FINAL_DATASET")):
    DATA_DIR = os.path.join(ROOT_DIR, "ONET_FINAL_DATASET")
else:
    DATA_DIR = os.path.join(ROOT_DIR, "ONET_DATASET")

MODEL_DIR = os.path.join(ROOT_DIR, "predictor_model")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# Global variables
df = None
roberta_model = None  
classifier = None     
shap_explainer = None
dashboard_stats = {}

# Stop words
STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'}

# --- LOADERS ---
def load_data():
    global df
    csv_path = os.path.join(DATA_DIR, "MERGED_Industry.csv")
    print(f"Loading dataset from: {csv_path}")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'Automation_Score' in df.columns:
                df['Automation_Score'] = pd.to_numeric(df['Automation_Score'], errors='coerce').fillna(0.0)
            print(f"Dataset loaded. Shape: {df.shape}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            df = pd.DataFrame()
    else:
        print(f"Warning: CSV not found at {csv_path}")
        df = pd.DataFrame()

def load_models():
    global roberta_model, classifier
    try:
        print("Loading RoBERTa...")
        roberta_model = SentenceTransformer('all-distilroberta-v1')
        print("RoBERTa loaded.")
    except Exception as e:
        print(f"Error loading RoBERTa: {e}")

    try:
        model_path = os.path.join(MODEL_DIR, "best_model_roberta_v2.keras")
        print(f"Loading Keras model from: {model_path}")
        classifier = tf.keras.models.load_model(model_path)
        print("Keras model loaded.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")

def setup_shap_explainer():
    global shap_explainer
    if roberta_model is None or classifier is None: return
    try:
        print("Setting up SHAP...")
        masker = shap.maskers.Text(r"\W+")
        def model_predict(texts):
            try:
                if isinstance(texts, str): texts = [texts]
                emb = roberta_model.encode(texts)
                pred = classifier.predict(emb, verbose=0)
                return np.hstack((1 - pred, pred))
            except: return np.array([[0.5, 0.5]] * len(texts))
        shap_explainer = shap.Explainer(model_predict, masker)
        print("SHAP initialized.")
    except Exception as e:
        print(f"Error setting up SHAP: {e}")

def precalculate_dashboard_stats():
    global dashboard_stats
    if df is None or df.empty: return
    print("Calculating dashboard stats...")
    # (Simplified logic for startup speed)
    if 'Automation_Score' in df:
        global_avg = float(df['Automation_Score'].mean())
        dashboard_stats['global_avg'] = global_avg
    print("Stats ready.")

# --- INITIALIZE ON STARTUP ---
# This ensures Gunicorn loads everything when it starts the worker
print("🔄 SERVER STARTUP: Initializing...")
load_data()
load_models()
setup_shap_explainer()
precalculate_dashboard_stats()
print("✅ SERVER STARTUP: Complete.")

# --- ROUTES ---
@app.route('/')
def index(): return app.send_static_file('homepage.html')

@app.route('/<path:path>')
def serve_static(path): return send_from_directory(FRONTEND_DIR, path)

@app.route('/api/stats', methods=['GET'])
def get_stats(): return jsonify(dashboard_stats)

@app.route('/predict', methods=['POST'])
def predict():
    if not roberta_model or not classifier: return jsonify({'error': 'Models loading'}), 503
    try:
        text = request.get_json().get('text', '')
        emb = roberta_model.encode([text])
        prob = float(classifier.predict(emb, verbose=0)[0][0])
        label = "High Risk" if prob >= 0.5 else "Low Risk"
        high_f, low_f = [], []
        if shap_explainer:
            try:
                sv = shap_explainer([text])
                vals = sv.values[0][:, 1] if len(sv.values.shape) >= 2 else sv.values[0]
                tokens = sv.data[0]
                factors = [{'word': t.strip(), 'score': float(v)} for t, v in zip(tokens, vals) if t.strip()]
                factors.sort(key=lambda x: abs(x['score']), reverse=True)
                high_f = [f for f in factors if f['score'] > 0]
                low_f = [f for f in factors if f['score'] < 0]
            except: pass
        return jsonify({'label': label, 'probability': prob, 'high_risk_factors': high_f, 'low_risk_factors': low_f})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/jobs_by_industry', methods=['GET'])
def get_jobs_by_industry():
    if df is None: return jsonify({'jobs': []})
    name = request.args.get('name', '').strip()
    try:
        return jsonify({'jobs': df[df['Industry'] == name]['Title'].unique().tolist()})
    except: return jsonify({'jobs': []})

@app.route('/api/compare', methods=['POST'])
def compare():
    if df is None: return jsonify({'error': 'No Data'}), 500
    q = request.get_json().get('query', '').strip()
    matches = df[df['Title'].astype(str).str.contains(q, case=False, na=False)]
    if matches.empty: return jsonify({'found': False, 'title': q, 'risk': 0.0})
    avg = float(matches['Automation_Score'].mean())
    task = str(matches['Task'].iloc[0]) if 'Task' in matches else "No details"
    return jsonify({'found': True, 'title': str(matches['Title'].iloc[0]), 'risk': avg, 'task': task})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    if df is None or not roberta_model: return jsonify({'error': 'Not ready'}), 503
    target = request.get_json().get('job_title', '').strip()
    thresh = request.get_json().get('threshold', 0.25)
    t_emb = roberta_model.encode([target])
    safe_df = df[df['Automation_Score'] < 2.05].drop_duplicates('Title')
    sims = cosine_similarity(t_emb, roberta_model.encode(safe_df['Title'].tolist()))[0]
    recs = []
    for idx in sims.argsort()[::-1]:
        if sims[idx] < thresh: break
        recs.append({'title': safe_df.iloc[idx]['Title'], 'score': float(safe_df.iloc[idx]['Automation_Score']), 'match_score': float(sims[idx]*100), 'task': str(safe_df.iloc[idx]['Task'])})
        if len(recs) >= 6: break
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
