"""
Model Utilities for Crop Yield Prediction App
=============================================
Handles model loading, preprocessing, prediction, and recommendations.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Model configuration
MODEL_PATH = 'models/rf_model_tuned.pkl'
ENCODERS_PATH = 'models/label_encoders.pkl'

# Feature columns (same as training)
NUMERIC_FEATURES = [
    'is_irrigated',
    'flooding_occurrence',
    'total_rainfall_mm',
    'water_retention_score',
    'inorganic_fert_qty_kg',
    'organic_fert_qty_kg',
    'pest_attack_occurred',
    'weeks_before_attack',
    'farm_size_ha',
    'experience_years',
    'weeding_frequency',
    'pesticide_used',
    'irrigation_vol_ltrs',
]

CATEGORICAL_FEATURES = ['seed_type', 'maize_variety', 'soil_type']

# Optimal model parameters (from tuning)
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'max_features': 0.5,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': 42,
    'n_jobs': -1
}


def train_and_save_model():
    """Train the model and save it for the app to use."""
    print("Training model for app...")

    # Load data
    df = pd.read_csv('farm_data_v2.csv')

    # Encode categorical features
    df_encoded = df.copy()
    label_encoders = {}

    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Prepare features
    feature_cols = NUMERIC_FEATURES + [col + '_encoded' for col in CATEGORICAL_FEATURES]
    X = df_encoded[feature_cols]
    y = df_encoded['yield_bags_per_ha']

    # Train model
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X, y)

    # Save model and encoders
    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)

    print(f"Model saved to {MODEL_PATH}")
    return model, label_encoders


def load_model_and_encoders():
    """Load the trained model and label encoders."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training new model...")
        train_and_save_model()

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)

    return model, label_encoders


def preprocess_input(input_data, label_encoders):
    """Preprocess user input for prediction."""
    # Create dataframe from input
    df = pd.DataFrame([input_data])

    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        col_encoded = col + '_encoded'
        le = label_encoders[col]
        df[col_encoded] = le.transform(df[col])

    # Select features in correct order
    feature_cols = NUMERIC_FEATURES + [col + '_encoded' for col in CATEGORICAL_FEATURES]
    X = df[feature_cols]

    return X


def predict_yield(input_data):
    """
    Predict crop yield for given input.

    Args:
        input_data: dict with all required features

    Returns:
        dict with prediction and confidence interval
    """
    model, label_encoders = load_model_and_encoders()

    # Preprocess
    X = preprocess_input(input_data, label_encoders)

    # Predict
    prediction = model.predict(X)[0]

    # Calculate confidence interval (approximate using std of predictions)
    # For a more robust method, we'd use quantile regression
    predictions_std = np.std([tree.predict(X)[0] for tree in model.estimators_])
    confidence_interval = (
        max(0, prediction - 1.5 * predictions_std),
        prediction + 1.5 * predictions_std
    )

    return {
        'yield': round(prediction, 2),
        'confidence_low': round(confidence_interval[0], 2),
        'confidence_high': round(confidence_interval[1], 2),
        'model_version': 'v1.0'
    }


def get_recommendations(input_data, predicted_yield):
    """
    Generate recommendations based on input data and predicted yield.

    Args:
        input_data: dict with farmer inputs
        predicted_yield: float predicted yield

    Returns:
        list of recommendation dicts
    """
    recommendations = []
    avg_yield = 16.79  # From EDA

    # Compare with average
    if predicted_yield > avg_yield:
        recommendations.append({
            'type': 'success',
            'title': 'Above Average Prediction',
            'message': f'Your predicted yield of {predicted_yield:.1f} bags/ha is above the average of {avg_yield:.1f} bags/ha!'
        })

    # Seed type recommendation
    if input_data.get('seed_type') == 'local':
        recommendations.append({
            'type': 'tip',
            'title': 'Switch to Improved Seeds',
            'message': 'Using improved seeds could increase your yield by approximately 31% (from ~14.7 to ~19.2 bags/ha).'
        })
    else:
        recommendations.append({
            'type': 'success',
            'title': 'Great Choice!',
            'message': 'Using improved seeds is an excellent decision for maximizing yield.'
        })

    # Fertilizer recommendations
    has_inorganic = input_data.get('inorganic_fert_used', False)
    has_organic = input_data.get('organic_fert_used', False)

    if not has_inorganic and not has_organic:
        recommendations.append({
            'type': 'warning',
            'title': 'Consider Using Fertilizers',
            'message': 'Using both organic and inorganic fertilizers together could increase your yield by up to 51%.'
        })
    elif has_inorganic and not has_organic:
        recommendations.append({
            'type': 'tip',
            'title': 'Add Organic Fertilizer',
            'message': 'Adding organic fertilizer alongside inorganic could boost your yield further.'
        })
    elif has_organic and not has_inorganic:
        recommendations.append({
            'type': 'tip',
            'title': 'Consider Inorganic Fertilizer',
            'message': 'Adding inorganic fertilizer could complement your organic fertilizer use.'
        })
    else:
        recommendations.append({
            'type': 'success',
            'title': 'Optimal Fertilizer Use',
            'message': 'Using both organic and inorganic fertilizers is optimal for maximum yield.'
        })

    # Planting month recommendation
    month = input_data.get('planting_month', '')
    if month in ['June', 'July']:
        recommendations.append({
            'type': 'success',
            'title': 'Optimal Planting Time',
            'message': f'Planting in {month} is ideal for maximum yield.'
        })
    elif month == 'May':
        recommendations.append({
            'type': 'warning',
            'title': 'Consider Adjusting Planting Time',
            'message': 'May planting shows lower yields. Consider June or July for better results.'
        })

    # Irrigation recommendation
    if not input_data.get('is_irrigated', False):
        recommendations.append({
            'type': 'tip',
            'title': 'Consider Irrigation',
            'message': 'Irrigation can significantly boost yield, especially during dry periods.'
        })

    # Flooding risk warning
    if input_data.get('flooding_risk') == 'High':
        recommendations.append({
            'type': 'warning',
            'title': 'High Flooding Risk!',
            'message': 'Flooding can reduce yield by up to 71%. Consider drainage solutions or alternative fields.'
        })

    # Pest management
    if not input_data.get('pesticide_used', False) and input_data.get('pest_attack_occurred', False):
        recommendations.append({
            'type': 'warning',
            'title': 'Pest Management Needed',
            'message': 'Using pesticides can help prevent yield loss from pest attacks.'
        })

    return recommendations


# For testing
if __name__ == '__main__':
    # Test prediction
    test_input = {
        'is_irrigated': 1,
        'flooding_occurrence': 0,
        'total_rainfall_mm': 1500,
        'water_retention_score': 4,
        'inorganic_fert_used': 1,
        'inorganic_fert_qty_kg': 500,
        'inorganic_fert_type': 'npk',
        'organic_fert_used': 1,
        'organic_fert_qty_kg': 300,
        'pest_attack_occurred': 0,
        'weeks_before_attack': 0,
        'farm_size_ha': 10,
        'experience_years': 15,
        'weeding_frequency': 3,
        'pesticide_used': 1,
        'irrigation_vol_ltrs': 3000,
        'seed_type': 'improved',
        'maize_variety': 'white',
        'soil_type': 'loamy',
        'planting_month': 'July',
        'flooding_risk': 'Low'
    }

    result = predict_yield(test_input)
    print("Prediction:", result)

    recs = get_recommendations(test_input, result['yield'])
    print("\nRecommendations:")
    for r in recs:
        print(f"  - {r['title']}: {r['message']}")
