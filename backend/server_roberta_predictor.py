import os
from flask import Flask, request, jsonify
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
df = None
roberta_model = None  
classifier = None     
shap_explainer = None
dashboard_stats = {}

# Stop words for word cloud
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
    'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
}

# --- HELPER: Find Project Root ---
def get_project_root():
    """
    Returns the root directory of the project.
    Assumes this script is inside a subfolder (e.g., /backend).
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to find the root where 'ONET_FINAL_DATASET' lives
    return os.path.dirname(current_script_dir)

def load_data():
    """Load the dataset dynamically relative to project root."""
    global df
    base_dir = get_project_root()
    
    # Path to your main dataset folder
    # NOTE: Ensure your repo folder name matches exactly (ONET_FINAL_DATASET or ONET_DATASET)
    dataset_folder = os.path.join(base_dir, "ONET_FINAL_DATASET")
    
    # Fallback check if folder wasn't found (common in local vs deployment structures)
    if not os.path.exists(dataset_folder):
        print(f"DEBUG: Dataset folder not found at {dataset_folder}, trying current dir...")
        dataset_folder = os.path.join(os.getcwd(), "ONET_FINAL_DATASET")

    csv_path = os.path.join(dataset_folder, "MERGED_Industry.csv")
    
    print(f"Loading dataset from: {csv_path}")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # Ensure Automation_Score is numeric
            if 'Automation_Score' in df.columns:
                df['Automation_Score'] = pd.to_numeric(df['Automation_Score'], errors='coerce').fillna(0.0)
            
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            df = pd.DataFrame()
            return False
    else:
        print(f"Warning: CSV file not found at {csv_path}")
        df = pd.DataFrame()
        return False

def load_models():
    """Load RoBERTa and Keras Models dynamically."""
    global roberta_model, classifier
    base_dir = get_project_root()
    
    # 1. Load RoBERTa (Downloads automatically, no path needed)
    try:
        print("Loading RoBERTa model (all-distilroberta-v1)...")
        roberta_model = SentenceTransformer('all-distilroberta-v1')
        print("RoBERTa model loaded successfully.")
    except Exception as e:
        print(f"Error loading RoBERTa: {e}")
        roberta_model = None
    
    # 2. Load Keras Model
    try:
        # Path to your model folder
        model_folder = os.path.join(base_dir, "predictor_model")
        
        # Fallback check
        if not os.path.exists(model_folder):
             model_folder = os.path.join(os.getcwd(), "predictor_model")
             
        model_path = os.path.join(model_folder, "best_model_roberta_v2.keras")
        print(f"Loading Keras classifier from: {model_path}")
        
        # Standard Loading (No hacks needed if envs match)
        classifier = tf.keras.models.load_model(model_path)
        print("Keras Neural Network loaded successfully.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        classifier = None

def setup_shap_explainer():
    """Initialize SHAP explainer."""
    global shap_explainer
    if roberta_model is None or classifier is None:
        return
    
    try:
        print("Setting up SHAP explainer...")
        masker = shap.maskers.Text(r"\W+")
        
        def model_predict(texts):
            try:
                if isinstance(texts, str): texts = [texts]
                embeddings = roberta_model.encode(texts)
                preds_1d = classifier.predict(embeddings, verbose=0)
                return np.hstack((1 - preds_1d, preds_1d))
            except:
                return np.array([[0.5, 0.5]] * len(texts))
        
        shap_explainer = shap.Explainer(model_predict, masker)
        print("SHAP explainer initialized successfully.")
    except Exception as e:
        print(f"Error setting up SHAP: {e}")
        shap_explainer = None

def precalculate_dashboard_stats():
    """Pre-calculate stats for the dashboard."""
    global dashboard_stats
    
    if df is None or df.empty:
        dashboard_stats = {'global_avg': 0.0, 'distribution': {'bins': [], 'counts': []}, 'top_risky_jobs': [], 'industry_stats': [], 'high_risk_words': [], 'low_risk_words': []}
        return
    
    print("Pre-calculating dashboard statistics...")
    if 'Automation_Score' not in df.columns: return

    global_avg = float(df['Automation_Score'].mean())
    counts, bins = np.histogram(df['Automation_Score'].dropna(), bins=50)
    distribution = {'bins': [float(b) for b in bins[:-1]], 'counts': [int(c) for c in counts]}
    
    top_risky_jobs = []
    if 'Title' in df.columns:
        job_risks = df.groupby('Title')['Automation_Score'].agg(['mean', 'count']).reset_index()
        job_risks = job_risks.sort_values('mean', ascending=False).head(1000)
        for _, row in job_risks.iterrows():
            val = float(row['mean']) if not math.isnan(row['mean']) else 0.0
            top_risky_jobs.append({'Title': str(row['Title']), 'Automation_Score': val, 'count': int(row['count'])})
            
    industry_stats = []
    if 'Industry' in df.columns:
        ind_stats = df.groupby('Industry').agg({'Automation_Score': 'mean', 'Title': 'nunique'}).reset_index()
        ind_stats = ind_stats.sort_values('mean_score', ascending=False)
        for _, row in ind_stats.iterrows():
            val = float(row['mean_score']) if not math.isnan(row['mean_score']) else 0.0
            industry_stats.append({'Industry': str(row['Industry']), 'Automation_Score': val, 'count': int(row['unique_jobs'])})

    high_risk_words = []
    low_risk_words = []
    if 'Task' in df.columns:
        threshold = df['Automation_Score'].median()
        high_tasks = df[df['Automation_Score'] >= threshold]['Task'].dropna()
        low_tasks = df[df['Automation_Score'] < threshold]['Task'].dropna()
        
        def get_top_words(tasks):
            words = []
            for t in tasks:
                words.extend([w for w in re.findall(r'\b[a-z]+\b', str(t).lower()) if w not in STOP_WORDS and len(w) > 2])
            return [[w, c] for w, c in Counter(words).most_common(80)]

        high_risk_words = get_top_words(high_tasks)
        low_risk_words = get_top_words(low_tasks)
    
    dashboard_stats = {
        'global_avg': global_avg, 'distribution': distribution,
        'top_risky_jobs': top_risky_jobs, 'industry_stats': industry_stats,
        'high_risk_words': high_risk_words, 'low_risk_words': low_risk_words
    }
    print("Dashboard stats ready.")

# --- API ENDPOINTS ---

@app.route('/predict', methods=['POST'])
def predict():
    if roberta_model is None or classifier is None:
        return jsonify({'error': 'Models not loaded'}), 500
    try:
        text = request.get_json().get('text', '')
        if not text: return jsonify({'error': 'No text'}), 400
        
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

@app.route('/api/stats', methods=['GET'])
def get_stats(): return jsonify(dashboard_stats)

@app.route('/api/cloud', methods=['GET'])
def get_cloud():
    t = request.args.get('type', 'high').lower()
    return jsonify(dashboard_stats.get('high_risk_words' if t == 'high' else 'low_risk_words', []))

@app.route('/api/jobs_by_industry', methods=['GET'])
def get_jobs_by_industry():
    if df is None or df.empty: return jsonify({'jobs': []}), 500
    name = request.args.get('name', '').strip()
    if not name: return jsonify({'jobs': []}), 400
    try:
        jobs = df[df['Industry'] == name]['Title'].unique().tolist()
        return jsonify({'jobs': jobs})
    except: return jsonify({'jobs': []}), 500

@app.route('/api/compare', methods=['POST'])
def compare_job():
    if df is None: return jsonify({'error': 'No Data'}), 500
    try:
        q = request.get_json().get('query', '').strip()
        if not q: return jsonify({'error': 'No query'}), 400
        matches = df[df['Title'].astype(str).str.contains(q, case=False, na=False)]
        if matches.empty: return jsonify({'found': False, 'title': q, 'risk': 0.0})
        
        avg_risk = float(matches['Automation_Score'].mean())
        if math.isnan(avg_risk): avg_risk = 0.0
        
        task = str(matches['Task'].dropna().iloc[0]) if not matches['Task'].dropna().empty else "No details"
        return jsonify({'found': True, 'title': str(matches['Title'].iloc[0]), 'risk': avg_risk, 'task_count': len(matches), 'task': task})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_jobs():
    if df is None or roberta_model is None: return jsonify({'error': 'Not ready'}), 500
    try:
        data = request.get_json()
        target = data.get('job_title', '').strip()
        thresh = data.get('threshold', 0.25)
        if not target: return jsonify([])
        
        target_emb = roberta_model.encode([target])
        safe_df = df[df['Automation_Score'] < 2.05].drop_duplicates('Title') if 'Automation_Score' in df else pd.DataFrame()
        if safe_df.empty: return jsonify([])
        
        safe_emb = roberta_model.encode(safe_df['Title'].tolist())
        sims = cosine_similarity(target_emb, safe_emb)[0]
        
        recs = []
        for idx in sims.argsort()[::-1]:
            if sims[idx] < thresh: break
            row = safe_df.iloc[idx]
            recs.append({
                'title': row['Title'], 
                'score': float(row['Automation_Score']), 
                'match_score': float(sims[idx]*100),
                'task': str(row['Task']) if 'Task' in row else "No details"
            })
            if len(recs) >= 6: break
        return jsonify(recs)
    except Exception as e: return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_data()
    load_models()
    setup_shap_explainer()
    precalculate_dashboard_stats()
    # Use 0.0.0.0 for external access (e.g. Docker/Render)
    app.run(debug=True, host='0.0.0.0', port=5000)
