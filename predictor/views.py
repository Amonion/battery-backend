import os
import pickle
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

MODELS = {
    'Neural Network': joblib.load(os.path.join(MODEL_PATH, 'ANN.pkl')),
    'Convolutional Neural Network': joblib.load(os.path.join(MODEL_PATH, 'CNN_LSTM.pkl')),
    'Decision Tree': joblib.load(os.path.join(MODEL_PATH, 'DECISION_TREE.pkl')),
    'Gradient Boosting': joblib.load(os.path.join(MODEL_PATH, 'GRADIENT_BOOST.pkl')),
    'Linear Regression': joblib.load(os.path.join(MODEL_PATH, 'LINEAR_REGRESSION.pkl')),
    'Random Forest': joblib.load(os.path.join(MODEL_PATH, 'RANDOM_FOREST.pkl')),
}

def map_to_battery_health(rul):
    if rul >= 741:
        return 'Excellent Battery Health'
    elif rul >= 371:
        return 'Average Battery Health'
    else:
        return 'Low or Poor Battery Health'

# Function to suggest optimization strategies based on battery health category
def suggest_optimization(category):
    if category == 'Excellent Battery Health':
        suggestions = [
            {"title": "Regular Monitoring", "explanation": "Continuously monitor the battery’s performance parameters."},
            {"title": "Optimal Charging Practices", "explanation": "Avoid overcharging and deep discharging. Use smart chargers."},
            {"title": "Temperature Management", "explanation": "Ensure the battery operates within the optimal temperature range."},
            {"title": "Balanced Usage", "explanation": "Ensure cells are balanced if the battery is part of a pack."},
            {"title": "Software Updates", "explanation": "Keep battery management software updated."}
        ]
    elif category == 'Average Battery Health':
        suggestions = [
            {"title": "Conditioning Cycles", "explanation": "Perform controlled charging and discharging cycles."},
            {"title": "Reduced Load", "explanation": "Lower the load on the battery whenever possible."},
            {"title": "Partial Charging", "explanation": "Maintain the battery’s state of charge between 20% and 80%."},
            {"title": "Routine Maintenance", "explanation": "Regularly clean and check connections and terminals."},
            {"title": "Usage Adjustments", "explanation": "Adjust usage patterns to avoid high-drain scenarios."}
        ]
    else:  # Low or Poor Battery Health
        suggestions = [
            {"title": "Capacity Testing", "explanation": "Regularly test the battery’s capacity."},
            {"title": "Load Reduction", "explanation": "Significantly reduce the load to prevent sudden failures."},
            {"title": "Refurbishment", "explanation": "Consider battery refurbishment options."},
            {"title": "Preemptive Replacement", "explanation": "Replace the battery to avoid unexpected failures."},
            {"title": "Recycling", "explanation": "Ensure proper recycling procedures for disposed batteries."}
        ]
    return suggestions

@api_view(['GET'])
def get_models(request):
    return Response({'models': list(MODELS.keys())})


@api_view(['POST'])
def predict(request):
    data = request.data
    model_name = data.get('model_name') 
    features = data.get('features')  
    
    print("Received data:", data)
    
    if model_name not in MODELS:
        return Response({'error': 'Model not found'}, status=status.HTTP_400_BAD_REQUEST)
    
    model = MODELS[model_name]

    try:
        processed_features = [
            float(features.get('Cycle_Index', 0) or 0),
            float(features.get('Discharge_Time', 0) or 0),
            float(features.get('Decrement', 0) or 0),
            float(features.get('Max_Voltage_Discharge', 0) or 0),
            float(features.get('Min_Voltage_Charge', 0) or 0),
            float(features.get('Time', 0) or 0),
            float(features.get('Time_constant_current', 0) or 0),
            float(features.get('Charging_time', 0) or 0),
        ]
        print("Processed features:", processed_features) 
    except ValueError as e:
        return Response({'error': f'Invalid feature values: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        prediction = model.predict([processed_features])[0]

        battery_health_category = map_to_battery_health(prediction)

        optimization_suggestion = suggest_optimization(battery_health_category)

        return Response({
            'prediction': prediction,
            'battery_health_category': battery_health_category,
            'optimization_suggestion': optimization_suggestion
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)