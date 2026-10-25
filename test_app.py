# -*- coding: utf-8 -*-
"""
Тесты для Loan Default Prediction API
"""

import pytest
from fastapi.testclient import TestClient
from app_final import app

client = TestClient(app)

# Валидные тестовые данные
VALID_REQUEST = {
    "customer_age": 35,
    "customer_income": 50000,
    "employment_duration": 24,
    "loan_amnt": 10000,
    "loan_int_rate": 10.5,
    "term_years": 5,
    "cred_hist_length": 10,
    "start_cred_history": 25,
    "home_ownership": "MORTGAGE",
    "loan_intent": "DEBTCONSOLIDATION",
    "loan_grade": "B",
    "historical_default": "N"
}


def test_health_endpoint():
    """Тест проверки работоспособности сервиса"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "error"]
    assert "service" in data
    assert "model_loaded" in data


def test_version_endpoint():
    """Тест проверки версии сервиса"""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "service_name" in data
    assert data["version"] == "1.0.0"


def test_predict_success():
    """Тест успешного прогноза дефолта"""
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "default_probability" in data
    assert "prediction_label" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["default_probability"] <= 1


def test_predict_with_high_risk_params():
    """Тест прогноза с параметрами высокого риска"""
    request_data = VALID_REQUEST.copy()
    request_data["customer_age"] = 22
    request_data["customer_income"] = 15000
    request_data["historical_default"] = "Y"
    request_data["loan_grade"] = "E"
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]


def test_invalid_home_ownership():
    """Тест невалидного статуса домовладения"""
    request_data = VALID_REQUEST.copy()
    request_data["home_ownership"] = "INVALID_VALUE"
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 400


def test_invalid_loan_grade():
    """Тест невалидной оценки кредита"""
    request_data = VALID_REQUEST.copy()
    request_data["loan_grade"] = "Z"
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 400


def test_age_validation_too_young():
    """Тест валидации возраста — слишком молодой"""
    request_data = VALID_REQUEST.copy()
    request_data["customer_age"] = 15
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422


def test_info_endpoint():
    """Тест информации о сервисе"""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "features" in data
    assert "metrics" in data
    assert len(data["features"]) == 12
