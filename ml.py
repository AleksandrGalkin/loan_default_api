# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:46:03 2024

@author: kexib
"""

from pickle import load
from catboost import CatBoostClassifier
import pandas as pd

def inference(data: list):
    if isinstance(data, list):
        data = pd.DataFrame(data, columns=['customer_age', 'customer_income', 'employment_duration', 'loan_amnt',
                                           'loan_int_rate', 'term_years', 'cred_hist_length', 'start_cred_history',
                                           'home_ownership', 'loan_intent', 'loan_grade', 'historical_default'])

    loaded_model = CatBoostClassifier().load_model("src/model.pcl")
    probabilities = loaded_model.predict_proba(data)  # Получаем вероятности
    y_pr = probabilities.argmax(axis=1)  # Получаем предсказанные классы
    return y_pr, probabilities  # Возвращаем и классы, и вероятности

def main():
    test_sample = [[22, 24000.0, 2.0, 6000.0, 12.18, 2, 3, 19, 'RENT', 'HOMEIMPROVEMENT', 'A', 'Y']]
    y_pr, probabilities = inference(test_sample)
    
    if (y_pr[0] == 0):
        print('Дефолт маловероятен')
    else:
        print('Вероятно будет дефолт')


if __name__ == "__main__":
    main()
    
    
    
    
    
    