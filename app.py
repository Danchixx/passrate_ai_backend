from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Directory for saving trained models
MODEL_DIR = "models"
REGRESSOR_PATH = os.path.join(MODEL_DIR, "regressor.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Global model storage
ml_models = {
    'scaler': None,
    'regressor': None,
    'classifier': None
}

# Initialize or load saved models
def initialize_models():
    """Load ML models if available, otherwise train and save."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if all(os.path.exists(p) for p in [REGRESSOR_PATH, CLASSIFIER_PATH, SCALER_PATH]):
        print("📦 Loading saved ML models...")
        ml_models['regressor'] = joblib.load(REGRESSOR_PATH)
        ml_models['classifier'] = joblib.load(CLASSIFIER_PATH)
        ml_models['scaler'] = joblib.load(SCALER_PATH)
        print("✓ Models loaded successfully")
    else:
        print("⚙️ Training models for the first time (this may take a bit)...")
        training_data = generate_synthetic_training_data(200)
        train_models(training_data)
        print("💾 Saving trained models...")
        joblib.dump(ml_models['regressor'], REGRESSOR_PATH)
        joblib.dump(ml_models['classifier'], CLASSIFIER_PATH)
        joblib.dump(ml_models['scaler'], SCALER_PATH)
        print("✓ Models saved for next startup")

def generate_synthetic_training_data(n_samples=200):
    """Generate synthetic training data based on educational patterns"""
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        # Simulate student performance
        num_sets = np.random.randint(2, 6)
        
        # Student ability level (latent variable)
        ability = np.random.normal(70, 15)
        
        # Generate set scores with some noise
        set_scores = []
        for i in range(num_sets):
            # Add learning effect and random variation
            learning_bonus = i * np.random.uniform(0, 3)
            score = np.clip(ability + learning_bonus + np.random.normal(0, 10), 0, 100)
            set_scores.append(score)
        
        # Calculate features
        avg_score = np.mean(set_scores)
        min_score = np.min(set_scores)
        max_score = np.max(set_scores)
        std_score = np.std(set_scores)
        
        # Calculate improvement trend (linear regression slope)
        if num_sets > 1:
            x = np.arange(num_sets)
            slope = np.polyfit(x, set_scores, 1)[0]
        else:
            slope = 0
        
        # True pass rate based on educational research
        # Strong correlation with avg score, min score, and consistency
        true_pass_rate = (
            0.85 * avg_score +  # Weighted heavily on average
            0.10 * min_score +  # Safety net - weakest performance
            0.05 * (100 - std_score) +  # Consistency bonus
            0.05 * max(0, slope * 10)  # Improvement trend bonus
        ) / 1.05  # Normalize
        
        # Add realistic noise
        true_pass_rate = np.clip(true_pass_rate + np.random.normal(0, 5), 10, 98)
        
        # Binary outcome for classification
        passed = 1 if true_pass_rate > 60 else 0
        
        data.append({
            'avg_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'std_score': std_score,
            'score_range': max_score - min_score,
            'improvement_trend': slope,
            'num_sets': num_sets,
            'consistency': 100 - (std_score * 2),
            'pass_rate': true_pass_rate,
            'passed': passed
        })
    
    return pd.DataFrame(data)

def train_models(training_df):
    """Train ML models on historical data"""
    features = ['avg_score', 'min_score', 'max_score', 'std_score', 
                'score_range', 'improvement_trend', 'num_sets', 'consistency']
    
    X = training_df[features]
    y_regression = training_df['pass_rate']
    y_classification = training_df['passed']
    
    # Train scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train regression model for pass rate prediction
    regressor = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    regressor.fit(X_scaled, y_regression)
    
    # Train classifier for pass/fail prediction
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    classifier.fit(X_scaled, y_classification)
    
    # Store models
    ml_models['scaler'] = scaler
    ml_models['regressor'] = regressor
    ml_models['classifier'] = classifier
    
    print("✓ ML Models trained successfully")
    print(f"  - Training samples: {len(training_df)}")
    print(f"  - Features: {len(features)}")
    print(f"  - Regressor R² score: {regressor.score(X_scaled, y_regression):.3f}")
    print(f"  - Classifier accuracy: {classifier.score(X_scaled, y_classification):.3f}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_models_loaded': all(v is not None for v in ml_models.values()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict-pass-rate', methods=['POST'])
def predict_pass_rate():
    """
    Main prediction endpoint
    Expected JSON input:
    {
        "set_scores": [75.5, 82.0, 78.5],
        "total_questions": 15,
        "correct_answers": 12,
        "student_id": "12345",
        "material_id": "67890"
    }
    """
    try:
        print(f"🧩 Received prediction request at {datetime.now().strftime('%H:%M:%S')}")
        
        data = request.get_json()
        
        # Validate input
        if 'set_scores' not in data or len(data['set_scores']) < 2:
            return jsonify({
                'success': False,
                'error': 'Minimum 2 set scores required'
            }), 400
        
        set_scores = data['set_scores']
        
        # Extract features
        features = extract_features(set_scores)
        
        # Prepare feature vector
        feature_vector = np.array([[
            features['avg_score'],
            features['min_score'],
            features['max_score'],
            features['std_score'],
            features['score_range'],
            features['improvement_trend'],
            features['num_sets'],
            features['consistency']
        ]])
        
        # Scale features
        feature_vector_scaled = ml_models['scaler'].transform(feature_vector)
        
        # Make predictions
        predicted_pass_rate = ml_models['regressor'].predict(feature_vector_scaled)[0]
        pass_probability = ml_models['classifier'].predict_proba(feature_vector_scaled)[0][1]
        
        # Ensemble prediction (combine both models)
        final_pass_rate = (predicted_pass_rate * 0.7) + (pass_probability * 100 * 0.3)
        final_pass_rate = np.clip(final_pass_rate, 10, 98)
        
        # Calculate confidence based on data quality
        confidence = calculate_confidence(features, set_scores)
        
        # Generate feature importance scores
        feature_scores = {
            'consistency': features['consistency'],
            'learningCurve': min(100, 50 + (features['improvement_trend'] * 8)),
            'stability': max(0, 100 - (features['std_score'] * 2)),
            'readiness': final_pass_rate
        }
        
        # Generate recommendations
        recommendations = generate_recommendations(features, final_pass_rate)
        
        result = {
            'success': True,
            'prediction': {
                'passRate': round(final_pass_rate, 2),
                'confidence': round(confidence, 2),
                'features': feature_scores,
                'recommendations': recommendations,
                'modelDetails': f"Gradient Boosting + Random Forest • {features['num_sets']} sets • Confidence: {confidence:.1f}%"
            },
            'raw_features': features,
            'model_info': {
                'type': 'Ensemble (GBR + RFC)',
                'training_samples': 1000,
                'pass_probability': round(pass_probability * 100, 2)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def extract_features(set_scores):
    """Extract ML features from test set scores"""
    set_scores = np.array(set_scores)
    
    features = {
        'avg_score': float(np.mean(set_scores)),
        'min_score': float(np.min(set_scores)),
        'max_score': float(np.max(set_scores)),
        'std_score': float(np.std(set_scores)),
        'score_range': float(np.max(set_scores) - np.min(set_scores)),
        'num_sets': len(set_scores)
    }
    
    # Calculate improvement trend (linear regression slope)
    if len(set_scores) > 1:
        x = np.arange(len(set_scores))
        slope = np.polyfit(x, set_scores, 1)[0]
        features['improvement_trend'] = float(slope)
    else:
        features['improvement_trend'] = 0.0
    
    # Calculate consistency score
    features['consistency'] = max(0.0, 100.0 - (features['std_score'] * 2))
    
    return features

def calculate_confidence(features, set_scores):
    """Calculate prediction confidence based on data quality"""
    confidence = 50.0  # Base confidence
    
    # More test sets = higher confidence
    confidence += min(30, (features['num_sets'] - 2) * 12)
    
    # High consistency = higher confidence
    if features['std_score'] < 8:
        confidence += 15
    elif features['std_score'] < 15:
        confidence += 8
    
    # Sufficient data points for trend analysis
    if features['num_sets'] >= 4:
        confidence += 10
    
    # Strong performance (low variance in top scores)
    if features['min_score'] > 60 and features['std_score'] < 10:
        confidence += 10
    
    # Penalize very low sample size
    if features['num_sets'] < 3:
        confidence -= 15
    
    return np.clip(confidence, 40, 95)

def generate_recommendations(features, pass_rate):
    """Generate personalized study recommendations"""
    recs = []
    
    # Overall readiness
    if pass_rate >= 85:
        recs.append("✅ Excellent preparation! High probability of success.")
    elif pass_rate >= 70:
        recs.append("✅ Good preparation level. Light review recommended.")
    elif pass_rate >= 55:
        recs.append("⚠️ Moderate readiness. Focus on weak areas to improve.")
    else:
        recs.append("🔴 Additional preparation strongly recommended.")
    
    # Learning trend
    if features['improvement_trend'] > 3:
        recs.append("📈 Strong upward trend - you're learning effectively!")
    elif features['improvement_trend'] < -3:
        recs.append("📉 Declining scores detected. Review study methods and rest.")
    
    # Consistency analysis
    if features['std_score'] > 20:
        recs.append("🎯 High score variance. Focus on consistent fundamentals.")
    elif features['std_score'] < 8:
        recs.append("🎯 Excellent consistency! Reliable performance pattern.")
    
    # Weak point identification
    if features['min_score'] < 50:
        recs.append(f"⚡ Lowest score ({features['min_score']:.0f}%) shows knowledge gaps. Target those topics.")
    
    # Data quality
    if features['num_sets'] < 3:
        recs.append("📊 Take more practice tests for better prediction accuracy.")
    
    return ' '.join(recs)

@app.route('/retrain-model', methods=['POST'])
def retrain_model():
    """
    Retrain models with new data
    Expected JSON: List of historical performance records
    """
    try:
        data = request.get_json()
        
        if 'training_data' not in data:
            return jsonify({
                'success': False,
                'error': 'No training data provided'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['training_data'])
        
        # Train new models
        train_models(df)
        
        return jsonify({
            'success': True,
            'message': f'Models retrained with {len(df)} samples'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting ML Pass Rate Prediction Server...")
    print("=" * 50)
    
    # Initialize models on startup
    initialize_models()
    
    print("=" * 50)
    print("✓ Server ready!")
    print("📡 Endpoints:")
    print("  - POST /predict-pass-rate")
    print("  - POST /retrain-model")
    print("  - GET  /health")
    print("=" * 50)
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True)