import os
import sys

from starlette.testclient import TestClient
import app

client = TestClient(app.app)

# 1. Test GET /
res = client.get('/')
print('GET / status:', res.status_code)
assert res.status_code == 200
assert 'Calculus &amp; Math Explainer' in res.text or 'Calculus & Math Explainer' in res.text
assert 'id="view-console"' in res.text
assert 'id="view-explainer"' in res.text

# 2. Test POST /api/predict
pred_payload = {
    'Rainfall_mm': 850.0,
    'Temperature_C': 23.5,
    'Soil_pH': 6.5,
    'Humidity_pct': 65.0,
    'Fertilizer_Used_kg': 135.0
}
res_pred = client.post('/api/predict', json=pred_payload)
print('POST /api/predict status:', res_pred.status_code)
pred_json = res_pred.json()
print('Prediction Momentum:', pred_json['prediction_momentum'], 'ton/ha')
assert res_pred.status_code == 200
assert len(pred_json['feature_contributions']) >= 5

# 3. Test POST /api/retrain
retrain_payload = {'epochs': 50, 'learning_rate': 0.005, 'momentum_factor': 0.90}
res_retrain = client.post('/api/retrain', json=retrain_payload)
print('POST /api/retrain status:', res_retrain.status_code, 'Test MSE Mom:', res_retrain.json()['test_mse_mom'])
assert res_retrain.status_code == 200

# 4. Test POST /api/calculate_math
math_payload = {
    'rainfall': 820.0,
    'temperature': 22.0,
    'soil_ph': 6.5,
    'humidity': 68.0,
    'fertilizer': 140.0,
    'actual_yield': 5.50
}
res_math = client.post('/api/calculate_math', json=math_payload)
print('POST /api/calculate_math status:', res_math.status_code, 'Yield:', res_math.json()['predicted_yield'])
assert res_math.status_code == 200

# 5. Test GET /api/dataset/info
res_info = client.get('/api/dataset/info')
print('GET /api/dataset/info status:', res_info.status_code, 'Dataset Rows:', res_info.json()['dataset_rows'])
assert res_info.status_code == 200

print('\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY WITH PERFECT FIDELITY!')
