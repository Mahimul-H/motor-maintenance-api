from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API is Online"

def test_predict_healthy():
    response = client.post("/predict", json={
        "voltage_v": 230,
        "current_a": 12,
        "temp_c": 65,
        "vibration_g": 0.05
    })
    assert response.status_code == 200
    assert response.json()["status"] == "NORMAL"

def test_predict_failure():
    response = client.post("/predict", json={
        "voltage_v": 180,  # Out of range
        "current_a": 25,
        "temp_c": 90,
        "vibration_g": 0.35
    })
    assert response.status_code == 200
    assert response.json()["status"] == "FAILURE RISK"