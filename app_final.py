# app_final.py
# -*- coding: utf-8 -*-
"""
Loan Default Prediction API
Сервис прогнозирования дефолта заемщика
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from ml import inference
import os
import logging
from datetime import datetime

# Инициализация приложения
app = FastAPI(
    title="Loan Default Prediction API",
    description="Сервис для прогнозирования вероятности дефолта заемщика",
    version="1.0.0"
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Переменные окружения
SERVICE_NAME = os.getenv("SERVICE_NAME", "loan-default-api")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# Схемы данных (Pydantic models)
class LoanRequest(BaseModel):
    """Запрос на прогнозирование дефолта"""
    customer_age: int = Field(..., ge=18, le=80, description="Возраст клиента")
    customer_income: float = Field(..., gt=0, description="Годовой доход клиента")
    employment_duration: int = Field(..., description="Продолжительность трудоустройства в месяцах (999 если неизвестно)")
    loan_amnt: float = Field(..., gt=0, description="Запрашиваемая сумма кредита")
    loan_int_rate: float = Field(..., ge=0, description="Процентная ставка по кредиту")
    term_years: int = Field(..., ge=1, le=30, description="Срок кредита в годах")
    cred_hist_length: int = Field(..., ge=0, le=80, description="Продолжительность кредитной истории в годах")
    start_cred_history: int = Field(..., ge=16, le=80, description="Возраст при оформлении первого кредита")
    home_ownership: str = Field(..., description="Статус домовладения")
    loan_intent: str = Field(..., description="Цель кредита")
    loan_grade: str = Field(..., description="Оценка, присвоенная кредиту")
    historical_default: str = Field(..., description="Допускал ли ранее дефолт")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    """Ответ с прогнозом"""
    prediction: int
    prediction_label: str
    default_probability: float
    service_name: str
    timestamp: str

class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str
    service: str
    model_loaded: bool

class VersionResponse(BaseModel):
    """Ответ с версией"""
    version: str
    model_version: str
    service_name: str

# Валидация категориальных значений
VALID_HOME_OWNERSHIP = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
VALID_LOAN_INTENT = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']
VALID_LOAN_GRADE = ['A', 'B', 'C', 'D', 'E']
VALID_HISTORICAL_DEFAULT = ['Y', 'N', 'no_info']

# ============== ENDPOINTS ==============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Проверка работоспособности сервиса
    """
    logger.info("Health check requested")
    try:
        # Проверяем, что модель загружена (просто пробуем сделать импорт)
        from ml import inference
        model_loaded = True
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        model_loaded = False
    
    return HealthResponse(
        status="ok" if model_loaded else "error",
        service=SERVICE_NAME,
        model_loaded=model_loaded
    )

@app.get("/version", response_model=VersionResponse, tags=["System"])
async def get_version():
    """
    Информация о версии сервиса
    """
    logger.info("Version requested")
    return VersionResponse(
        version="1.0.0",
        model_version=MODEL_VERSION,
        service_name=SERVICE_NAME
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_default(request: LoanRequest):
    """
    Прогнозирование вероятности дефолта заемщика
    
    Принимает параметры займа и возвращает прогноз с вероятностью
    """
    logger.info(f"Prediction request received for age={request.customer_age}, income={request.customer_income}")
    
    # Валидация категориальных значений
    if request.home_ownership not in VALID_HOME_OWNERSHIP:
        raise HTTPException(status_code=400, detail=f"Invalid home_ownership. Must be one of: {VALID_HOME_OWNERSHIP}")
    if request.loan_intent not in VALID_LOAN_INTENT:
        raise HTTPException(status_code=400, detail=f"Invalid loan_intent. Must be one of: {VALID_LOAN_INTENT}")
    if request.loan_grade not in VALID_LOAN_GRADE:
        raise HTTPException(status_code=400, detail=f"Invalid loan_grade. Must be one of: {VALID_LOAN_GRADE}")
    if request.historical_default not in VALID_HISTORICAL_DEFAULT:
        raise HTTPException(status_code=400, detail=f"Invalid historical_default. Must be one of: {VALID_HISTORICAL_DEFAULT}")
    
    try:
        # Подготовка данных для модели
        params = [[
            request.customer_age,
            request.customer_income,
            request.employment_duration,
            request.loan_amnt,
            request.loan_int_rate,
            request.term_years,
            request.cred_hist_length,
            request.start_cred_history,
            request.home_ownership,
            request.loan_intent,
            request.loan_grade,
            request.historical_default
        ]]
        
        # Получение прогноза
        y_pr, probabilities = inference(params)
        
        # Извлечение вероятности дефолта (класс 1)
        default_probability = float(probabilities[0][1])
        prediction = int(y_pr[0])
        prediction_label = "Дефолт вероятен" if prediction == 1 else "Дефолт маловероятен"
        
        logger.info(f"Prediction completed: {prediction_label}, probability={default_probability:.4f}")
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            default_probability=round(default_probability, 4),
            service_name=SERVICE_NAME,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/info", tags=["System"])
async def get_info():
    """
    Информация о сервисе и модели
    """
    logger.info("Service info requested")
    return {
        "service": SERVICE_NAME,
        "description": "API для прогнозирования дефолта заемщика на основе параметров кредита",
        "features": [
            "customer_age",
            "customer_income", 
            "employment_duration",
            "loan_amnt",
            "loan_int_rate",
            "term_years",
            "cred_hist_length",
            "start_cred_history",
            "home_ownership",
            "loan_intent",
            "loan_grade",
            "historical_default"
        ],
        "metrics": {
            "precision": 0.98,
            "recall": 0.92,
            "f1_score": 0.95
        },
        "data_source": "Данные заемщиков Великобритании"
    }

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_final:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )