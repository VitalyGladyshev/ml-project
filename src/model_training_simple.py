import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os
from datetime import datetime

def classification_metrics(y_true, y_pred):
    """Расчет метрик классификации"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro")
    }

def train_models_without_mlflow(X_train, X_test, y_train, y_test, feature_names):
    """Обучение моделей без MLflow"""
    print("\n" + "=" * 60)
    print("=== Обучение моделей (без MLflow) ===")
    
    models = {
        "LogisticRegression": make_pipeline(LogisticRegression(solver="liblinear", random_state=42, max_iter=1000)),
        "SVC": make_pipeline(SVC(gamma="auto", random_state=42, probability=True)),
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=15)),
        "DecisionTree": make_pipeline(DecisionTreeClassifier(max_depth=7, random_state=42)),
        "RandomForest": make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        "GradientBoosting": make_pipeline(GradientBoostingClassifier(random_state=42))
    }
    
    best_f1 = 0
    best_model_info = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs("models", exist_ok=True)
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"Обучение модели: {model_name}")
        
        # Обучение модели
        model.fit(X_train, y_train)
        print("   Модель обучена")
        
        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Расчет метрик
        train_metrics = classification_metrics(y_train, y_train_pred)
        test_metrics = classification_metrics(y_test, y_test_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Вывод метрик
        print(f"   Метрики (test):")
        print(f"      • Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"      • F1-score (macro): {test_metrics['f1_macro']:.4f}")
        print(f"      • Precision (macro): {test_metrics['precision_macro']:.4f}")
        print(f"      • Recall (macro): {test_metrics['recall_macro']:.4f}")
        print(f"   Кросс-валидация F1: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Сохранение модели
        model_path = f"models/{model_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        print(f"   💾 Модель сохранена: {os.path.abspath(model_path)}")
        
        # Сохранение результатов
        results.append({
            "model_name": model_name,
            "test_f1": test_metrics['f1_macro'],
            "cv_f1": cv_mean,
            "model_path": model_path
        })
        
        # Лучшая модель
        if test_metrics['f1_macro'] > best_f1:
            best_f1 = test_metrics['f1_macro']
            best_model_info = {
                "model_name": model_name,
                "best_f1": best_f1,
                "model": model,
                "model_path": model_path
            }
    
    # Вывод итогов
    print(f"\n{'=' * 50}")
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (без MLflow):")
    print("-" * 50)
    for result in sorted(results, key=lambda x: x['test_f1'], reverse=True):
        print(f"{result['model_name']}:")
        print(f"  • F1-score (test): {result['test_f1']:.4f}")
        print(f"  • CV F1: {result['cv_f1']:.4f}")
        print(f"  • Модель: {os.path.basename(result['model_path'])}")
    
    print(f"\nЛУЧШАЯ МОДЕЛЬ:")
    print(f"  • Название: {best_model_info['model_name']}")
    print(f"  • F1-score (test): {best_model_info['best_f1']:.4f}")
    print(f"  • Путь к модели: {os.path.basename(best_model_info['model_path'])}")
    
    return best_model_info