from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import shap
from collections import Counter
import re
import os
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

def load_data():
    """Load the dataset with error handling"""
    global df
    base_dir = r"C:\Users\User\source\repos\ONET_DATASET_V3\ONET_PREPROCESSING_STEPS"
    preferred_files = ["MERGED_Industry.csv", "MERGED_1.csv"]
    
    for fname in preferred_files:
        data_path = os.path.join(base_dir, fname)
        if os.path.exists(data_path):
            try:
                print(f"Loading dataset from: {data_path}")
                df = pd.read_csv(data_path)
                
                # Ensure Automation_Score is numeric
                if 'Automation_Score' in df.columns:
                    df['Automation_Score'] = pd.to_numeric(df['Automation_Score'], errors='coerce').fillna(0.0)
                
                print(f"Dataset loaded successfully from {fname}. Shape: {df.shape}")
                return True
            except Exception as e:
                print(f"Error loading dataset {fname}: {e}")
                df = pd.DataFrame()
                return False
    
    print(f"Warning: No preferred dataset found in {base_dir}. Using empty DataFrame.")
    df = pd.DataFrame()
    return False

def load_models():
    """Load RoBERTa and Keras Models (Standard Loading)"""
    global roberta_model, classifier
    
    # 1. Load the Sentence Transformer (RoBERTa)
    try:
        print("Loading RoBERTa model (all-distilroberta-v1)...")
        roberta_model = SentenceTransformer('all-distilroberta-v1')
        print("RoBERTa model loaded successfully.")
    except Exception as e:
        print(f"Error loading RoBERTa model: {e}")
        roberta_model = None
    
    # 2. Load the Keras Neural Network
    try:
        model_path = r"C:\Users\User\source\repos\ONET_DATASET_V3\ONET_PREPROCESSING_STEPS\best_model_roberta_v2.keras"
        print(f"Loading Keras classifier from: {model_path}")
        
        # STANDARD LOADING (No hacks needed anymore!)
        classifier = tf.keras.models.load_model(model_path)
        
        print("Keras Neural Network loaded successfully.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        classifier = None

def setup_shap_explainer():
    """Initialize SHAP explainer for RoBERTa + Keras"""
    global shap_explainer
    if roberta_model is None or classifier is None:
        # Don't print warning here to keep console clean, just return
        return
    
    try:
        print("Setting up SHAP explainer...")
        
        masker = shap.maskers.Text(r"\W+")
        
        def model_predict(texts):
            try:
                if isinstance(texts, str):
                    texts = [texts]
                
                embeddings = roberta_model.encode(texts)
                preds_1d = classifier.predict(embeddings, verbose=0)
                preds_2d = np.hstack((1 - preds_1d, preds_1d))
                return preds_2d
                
            except Exception as e:
                # Silent fail for robustness during prediction
                return np.array([[0.5, 0.5]] * len(texts))
        
        shap_explainer = shap.Explainer(model_predict, masker)
        print("SHAP explainer initialized successfully.")
        
    except Exception as e:
        print(f"Error setting up SHAP explainer: {e}")
        shap_explainer = None

def precalculate_dashboard_stats():
    """Pre-calculate dashboard statistics on startup"""
    global dashboard_stats
    
    if df is None or df.empty:
        dashboard_stats = {'global_avg': 0.0, 'distribution': {'bins': [], 'counts': []}, 'top_risky_jobs': [], 'industry_stats': [], 'high_risk_words': [], 'low_risk_words': []}
        return
    
    print("Pre-calculating dashboard statistics...")
    
    if 'Automation_Score' not in df.columns:
        return

    # 1. Global average & Histogram
    global_avg = float(df['Automation_Score'].mean())
    counts, bins = np.histogram(df['Automation_Score'].dropna(), bins=50)
    distribution = {
        'bins': [float(b) for b in bins[:-1]],
        'counts': [int(c) for c in counts]
    }
    
    # 2. Risky Job Titles
    top_risky_jobs = []
    if 'Title' in df.columns:
        job_risks = df.groupby('Title')['Automation_Score'].agg(['mean', 'count']).reset_index()
        job_risks = job_risks.sort_values('mean', ascending=False).head(1000)
        for _, row in job_risks.iterrows():
            val = float(row['mean']) if not math.isnan(row['mean']) else 0.0
            top_risky_jobs.append({
                'Title': str(row['Title']),
                'Automation_Score': val,
                'count': int(row['count'])
            })
            
    # 3. Industry Stats
    industry_stats = []
    if 'Industry' in df.columns:
        ind_stats = df.groupby('Industry').agg({
            'Automation_Score': 'mean',
            'Title': 'nunique'
        }).reset_index()
        ind_stats.columns = ['Industry', 'mean_score', 'unique_jobs']
        ind_stats = ind_stats.sort_values('mean_score', ascending=False)
        for _, row in ind_stats.iterrows():
            val = float(row['mean_score']) if not math.isnan(row['mean_score']) else 0.0
            industry_stats.append({
                'Industry': str(row['Industry']),
                'Automation_Score': val,
                'count': int(row['unique_jobs'])
            })

    # 4. Word Clouds
    high_risk_words = []
    low_risk_words = []
    if 'Task' in df.columns:
        threshold = df['Automation_Score'].median()
        high_risk_tasks = df[df['Automation_Score'] >= threshold]['Task'].dropna()
        low_risk_tasks = df[df['Automation_Score'] < threshold]['Task'].dropna()
        
        def get_top_words(tasks):
            words = []
            for task in tasks:
                w_list = re.findall(r'\b[a-z]+\b', str(task).lower())
                words.extend([w for w in w_list if w not in STOP_WORDS and len(w) > 2])
            return [[w, c] for w, c in Counter(words).most_common(80)]

        high_risk_words = get_top_words(high_risk_tasks)
        low_risk_words = get_top_words(low_risk_tasks)
    
    dashboard_stats = {
        'global_avg': global_avg,
        'distribution': distribution,
        'top_risky_jobs': top_risky_jobs,
        'industry_stats': industry_stats,
        'high_risk_words': high_risk_words,
        'low_risk_words': low_risk_words
    }
    print("Dashboard statistics pre-calculated successfully.")

# --- API ENDPOINTS ---

@app.route('/predict', methods=['POST'])
def predict():
    if roberta_model is None or classifier is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text: return jsonify({'error': 'No text provided'}), 400
        
        embedding = roberta_model.encode([text])
        prediction_raw = classifier.predict(embedding, verbose=0)
        probability = float(prediction_raw[0][0])
        label = "High Risk" if probability >= 0.5 else "Low Risk"
        
        high_factors = []
        low_factors = []
        
        if shap_explainer:
            try:
                shap_values = shap_explainer([text])
                if len(shap_values.values.shape) >= 2:
                    scores = shap_values.values[0][:, 1]
                else:
                    scores = shap_values.values[0]
                tokens = shap_values.data[0]
                word_scores = []
                for word, score in zip(tokens, scores):
                    if word and word.strip() and abs(score) > 0.001:
                        word_scores.append({'word': word.strip(), 'score': float(score)})
                word_scores.sort(key=lambda x: abs(x['score']), reverse=True)
                high_factors = [x for x in word_scores if x['score'] > 0]
                low_factors = [x for x in word_scores if x['score'] < 0]
            except Exception:
                pass 

        return jsonify({
            'label': label,
            'probability': probability,
            'high_risk_factors': high_factors,
            'low_risk_factors': low_factors
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(dashboard_stats)

@app.route('/api/cloud', methods=['GET'])
def get_word_cloud():
    cloud_type = request.args.get('type', 'high').lower()
    return jsonify(dashboard_stats.get('high_risk_words', []) if cloud_type == 'high' else dashboard_stats.get('low_risk_words', []))

@app.route('/api/jobs_by_industry', methods=['GET'])
def get_jobs_by_industry():
    global df
    if df is None or df.empty:
        return jsonify({'jobs': [], 'error': 'Dataset not loaded'}), 500
    
    industry_name = request.args.get('name', '').strip()
    if not industry_name:
        return jsonify({'jobs': [], 'error': 'Missing name'}), 400
        
    try:
        if 'Industry' not in df.columns or 'Title' not in df.columns:
             return jsonify({'jobs': []})

        industry_jobs = df[df['Industry'] == industry_name]
        if industry_jobs.empty:
            return jsonify({'jobs': []})
            
        job_titles = industry_jobs['Title'].unique().tolist()
        return jsonify({'jobs': job_titles})
    except Exception as e:
        print(f"Error fetching jobs: {e}")
        return jsonify({'jobs': [], 'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_job():
    if df is None or df.empty: return jsonify({'error': 'Dataset not loaded'}), 500
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        if not query: return jsonify({'error': 'No query'}), 400
        
        matches = df[df['Title'].astype(str).str.contains(query, case=False, na=False)]
        if matches.empty:
            return jsonify({'found': False, 'title': query, 'risk': 0.0})
            
        avg_risk_raw = matches['Automation_Score'].mean()
        avg_risk = float(avg_risk_raw) if not math.isnan(avg_risk_raw) else 0.0
        count = int(len(matches))
        job_title = str(matches['Title'].iloc[0])
        
        task_desc = "No description available."
        if 'Task' in matches.columns:
            valid = matches['Task'].dropna()
            if not valid.empty: task_desc = str(valid.iloc[0])

        return jsonify({'found': True, 'title': job_title, 'risk': avg_risk, 'task_count': count, 'task': task_desc})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_jobs():
    if df is None or roberta_model is None: return jsonify({'error': 'System not ready'}), 500
    try:
        data = request.get_json()
        current_job_title = data.get('job_title', '').strip()
        threshold = data.get('threshold', 0.25)
        
        if not current_job_title: return jsonify([])

        target_embedding = roberta_model.encode([current_job_title])
        
        if 'Automation_Score' in df.columns:
            safe_jobs_df = df[df['Automation_Score'] < 2.05].drop_duplicates(subset=['Title'])
        else:
            safe_jobs_df = pd.DataFrame()
            
        if safe_jobs_df.empty: return jsonify([])

        safe_titles = safe_jobs_df['Title'].tolist()
        safe_embeddings = roberta_model.encode(safe_titles)
        
        similarities = cosine_similarity(target_embedding, safe_embeddings)[0]
        sorted_indices = similarities.argsort()[::-1]
        
        recommendations = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            if score >= threshold:
                row = safe_jobs_df.iloc[idx]
                rec_score = float(row['Automation_Score']) if 'Automation_Score' in df.columns else 0.0
                task_txt = str(row['Task']) if 'Task' in row else "No details"
                
                recommendations.append({
                    'title': row['Title'],
                    'score': rec_score,
                    'match_score': float(score * 100),
                    'task': task_txt
                })
                if len(recommendations) >= 6: break
        
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_data()
    load_models()
    setup_shap_explainer()
    precalculate_dashboard_stats()
    app.run(debug=True, host='127.0.0.1', port=5000)