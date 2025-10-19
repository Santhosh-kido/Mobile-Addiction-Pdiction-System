from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from data_processor import process_data_file, convert_api_features
from ml_models import run_ml_algorithms

app = Flask(__name__)
CORS(app)



# Load data at startup
try:
    training_data, labels = process_data_file('Training.csv')
    test_data, test_labels = process_data_file('Testing.csv')
    print(f"Loaded {len(training_data)} training and {len(test_data)} test samples")
except Exception as e:
    print(f"Error loading data: {e}")
    training_data = np.array([[25, 1, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 3, 0, 1]])
    labels = np.array([2])
    test_data, test_labels = training_data, labels



@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        print(f"Received data: {input_data}")
        
        if not input_data:
            return jsonify({"error": "No data received"}), 400
            
        user_features = [convert_api_features(input_data)]
        
        X_train, y_train = training_data, labels
        X_test, y_test = test_data, test_labels
        
        results = run_ml_algorithms(X_train, y_train, X_test, y_test, user_features)
        
        # Ensemble
        avg_percentage = sum(r['addictionPercentage'] for r in results) / len(results)
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        
        ensemble_prediction = "Low Risk" if avg_percentage < 30 else "Moderate Risk" if avg_percentage < 65 else "High Risk"
        
        ensemble_result = {
            "algorithm": "Ensemble (6 Models)",
            "prediction": ensemble_prediction,
            "accuracy": round(avg_accuracy),
            "addictionPercentage": round(avg_percentage)
        }
        
        # Print final results to console
        print("\n==================== PREDICTION RESULTS ====================")
        print(f"Final Prediction: {ensemble_prediction}")
        print(f"Addiction Percentage: {round(avg_percentage)}%")
        print(f"Average Accuracy: {round(avg_accuracy)}%")
        print("\nIndividual Algorithm Results:")
        for result in results:
            print(f"  {result['algorithm']} - Accuracy: {result['accuracy']}%")
        print("============================================================\n")
        
        return jsonify({
            "results": results,
            "ensembleResult": ensemble_result
        })
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)