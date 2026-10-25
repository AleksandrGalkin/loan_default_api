# Loan Default Prediction API

API для прогнозирования вероятности дефолта заёмщика на основе параметров кредита.

## Описание
Сервис принимает параметры заёмщика и кредита, возвращает прогноз о вероятности дефолта.
Модель обучена на данных заёмщиков Великобритании и построена на алгоритме **CatBoost**.

**Метрики модели:**
- Precision: 0.98
- Recall: 0.92
- F1-score: 0.95

## Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/AleksandrGalkin/loan_default_api.git
cd loan_default_api
```

## 2. Создайте и активируйте виртуальное окружение:
```bash
python -m venv venv
source venv/Scripts/activate
```

## 3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск приложения
Режим разработки (с автоперезагрузкой при изменениях кода):
```bash
python -m uvicorn app_final:app --reload
```

## Продакшен-режим (команда запуска БЕЗ --reload, как требуется в задании):
```bash
python -m uvicorn app_final:app --host 0.0.0.0 --port 8000
```

После запуска сервера интерактивная API документация (Swagger) доступна по адресу:
http://127.0.0.1:8000/docs
Основные endpoints и диагностика
>GET /health — Проверка работоспособности сервиса.
Проверка: curl http://127.0.0.1:8000/health
Ответ: {"status": "ok", "service": "loan-default-api", "model_loaded": true}
>GET /version — Информация о версии сервиса и модели.
Проверка: curl http://127.0.0.1:8000/version
>POST /predict — Прогнозирование вероятности дефолта (принимает JSON с параметрами клиента).
>GET /info — Информация о признаках и метриках модели.

## Логирование:
Логи выводятся в стандартный поток вывода (stdout) в терминале. Уровень логирования — INFO. Фиксируются: запуск сервиса, входящие запросы с параметрами, результаты прогноза и ошибки.
Тесты

## Запуск автоматических тестов локально:
```bash
python -m pytest -v
```

## Тесты также автоматически запускаются в GitHub Actions при каждом push и pull request.
Переменные окружения
Для настройки сервиса используются следующие переменные (файл .env создается на основе .env.example):
>Переменная SERVICE_NAME
>Описание Имя сервиса для идентификации в логах и ответах
>Значение по умолчанию loan-default-api

>Переменная MODEL_VERSION
>Описание Версия используемой ML-модели
>Значение по умолчанию  1.0.0

##  Пояснения к параметрам запроса (POST /predict)
> home_ownership (статус домовладения): RENT (аренда), OWN (собственность), MORTGAGE (ипотека), OTHER (остальное)
> loan_intent (цель кредита): EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT
> loan_grade (оценка кредита): A, B, C, D, E (от лучшей к худшей)
> historical_default (исторический дефолт): Y (да), N (нет), no_info (нет информации)

##  Структура проекта
```text
loan_default_api/
├── app_final.py              # Основной код FastAPI приложения
── ml.py                     # Логика загрузки модели и предсказания
├── model.pkl                 # Файл с весами обученной модели CatBoost
├── test_app.py               # Автоматические тесты (pytest)
├── requirements.txt          # Зависимости для запуска
├── requirements-dev.txt      # Зависимости для разработки и тестов
├── .env.example              # Пример файла переменных окружения
├── .gitignore                # Исключения для Git
├── README.md                 # Эта документация
└── .github/workflows/ci.yml  # Конфигурация GitHub Actions CI/CD
```

##  Автор
Aleksandr Galkin