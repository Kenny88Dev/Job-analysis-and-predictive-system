# AI Workforce Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A comprehensive AI-powered system designed to analyze labor market trends, predict automation risks, and provide data-driven career guidance.**



[Image of Dashboard Screenshot placeholder]


## 📖 Overview

The **AI Workforce Intelligence Platform** is a web-based tool leveraging **RoBERTa embeddings** and **Deep Learning** to predict the susceptibility of occupations to automation. Unlike simple keyword matching, this system analyzes the *semantic meaning* of job descriptions using the **O*NET dataset**.

It features **Explainable AI (XAI)** integration via **SHAP**, allowing users to understand exactly *why* a job is considered high or low risk by visualizing the impact of specific tasks and keywords (e.g., "creativity" vs. "repetition").

## ✨ Key Features

* **🤖 Automation Risk Predictor:** A RoBERTa-based Neural Network that calculates a granular risk score (0-5) for any job description.
* **🧠 Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) to visualize positive and negative contributors to the risk score.
* **🧭 Career Pathfinder:** A semantic recommendation engine that suggests "safe" career alternatives based on skill similarity but lower automation risk.
* **📊 Market Index:** Interactive visualizations (Chart.js) showing risk distribution across major industries.
* **⚖️ Job Comparator:** Head-to-head analysis of two occupations to compare risk scores and task snapshots.
* **Flip-Card UI:** Engaging frontend interface for exploring job details and recommendations.

## 🛠️ Tech Stack

### Machine Learning & Backend
* **Python 3.11**
* **TensorFlow / Keras 2.14:** Core neural network training and inference.
* **Sentence-Transformers (RoBERTa):** Generating high-dimensional vector embeddings for NLP tasks.
* **SHAP:** Model interpretability and feature importance.
* **Flask:** RESTful API backend handling predictions and data retrieval.
* **Pandas & NumPy:** Data manipulation and analysis.

### Frontend
* **HTML5 / CSS3:** Modern, responsive UI with glassmorphism effects.
* **JavaScript (Vanilla):** Dynamic DOM manipulation and API integration.
* **Chart.js:** Data visualization for the Market Index.
* **D3.js:** Force-directed graphs for the Risk Cloud.

## ⚙️ Installation & Setup

> **⚠️ Important:** This project requires specific library versions to load the pre-trained Keras model correctly. Please follow the setup below strictly.

1. Clone the Repository
   
2. Create a Virtual Environment (Python 3.11 Recommended)
  * **It is highly recommended to use Python 3.11 to avoid compatibility issues with TensorFlow 2.14.
  * **# Windows
  * **python -m venv venv
  * **.\venv\Scripts\activate

  * **# Mac/Linux
  * **python3 -m venv venv
  * **source venv/bin/activate
   
3. Install Dependencies
   * **Install the required packages. Note the specific versions for numpy and sentence-transformers to ensure stability.
   * **pip install tensorflow==2.14.0
   * **pip install sentence-transformers==2.7.0
   * **pip install numpy<2.0
   * **pip install shap==0.44.1
   * **pip install pandas flask flask-cors scikit-learn

4. Data Setup
   * **Ensure the O*NET CSV files are located in the ONET_PREPROCESSING_STEPS directory:
   * **MERGED_Industry.csv
   * **best_model_roberta.keras (Pre-trained model)

####################################################

🚀 Usage
1. Start the Backend Server:
   python server_roberta_predictor.py
   The server will start at http://127.0.0.1:5000.

2. Launch the Frontend: Open homepage.html directly in your web browser.
   📂 Project Structure
    ├── ONET_PREPROCESSING_STEPS/   # Dataset and Model files
    │   ├── MERGED_Industry.csv
    │   └── best_model_roberta.keras
    ├── FRONTEND/                   # Web Interface
    │   ├── homepage.html
    │   ├── predictor.html
    │   ├── pathfinder.html
    │   ├── style.css
    │   └── script.js
    ├── server_roberta_predictor.py # Flask Backend
    └── README.md

#####################################################

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📊 Dataset Attribution
This application utilizes data from the O*NET database, sponsored by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA).
