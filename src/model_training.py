import mlflow
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

def classification_metrics(y_true, y_pred, prefix=""):
    """Расчет и возврат метрик классификации"""
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}precision_macro": precision_score(y_true, y_pred, average="macro"),
        f"{prefix}recall_macro": recall_score(y_true, y_pred, average="macro")
    }
    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Обучение и оценка различных моделей с логированием в MLflow"""
    print("📈 Начало обучения моделей...")
    
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
    
    # Создание директорий для моделей
    os.makedirs("models", exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"🔍 Обучение модели: {model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_{timestamp}") as run:
            # Логирование параметров
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("timestamp", timestamp)
            
            # Специфичные параметры для некоторых моделей
            if model_name == "KNN":
                mlflow.log_param("n_neighbors", 15)
            elif model_name == "DecisionTree":
                mlflow.log_param("max_depth", 7)
            elif model_name == "RandomForest":
                mlflow.log_param("n_estimators", 100)
            
            # Обучение модели
            print("   ⏳ Обучение модели...")
            model.fit(X_train, y_train)
            print("   ✅ Модель обучена")
            
            # Предсказания
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Расчет метрик
            train_metrics = classification_metrics(y_train, y_train_pred, "train_")
            test_metrics = classification_metrics(y_test, y_test_pred, "test_")
            
            # Кросс-валидация
            print("   ⏳ Выполнение кросс-валидации...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Логирование метрик
            for metric_name, value in {**train_metrics, **test_metrics}.items():
                mlflow.log_metric(metric_name, value)
            
            mlflow.log_metric("cv_f1_mean", float(cv_mean))
            mlflow.log_metric("cv_f1_std", float(cv_std))
            
            # Вывод метрик в консоль
            print(f"   📊 Метрики (test):")
            print(f"      • Accuracy: {test_metrics['test_accuracy']:.4f}")
            print(f"      • F1-score (macro): {test_metrics['test_f1_macro']:.4f}")
            print(f"      • Precision (macro): {test_metrics['test_precision_macro']:.4f}")
            print(f"      • Recall (macro): {test_metrics['test_recall_macro']:.4f}")
            print(f"   🔄 Кросс-валидация F1: {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Сохранение модели
            model_path = f"models/{model_name}_{timestamp}.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            print(f"   💾 Модель сохранена: {model_path}")
            
            # Сохранение лучших результатов
            if test_metrics["test_f1_macro"] > best_f1:
                best_f1 = test_metrics["test_f1_macro"]
                best_model_info = {
                    "model_name": model_name,
                    "best_f1": best_f1,
                    "model": model,
                    "run_id": run.info.run_id,
                    "model_path": model_path
                }
            
            print(f"   ✅ Завершено: {model_name}")
    
    print(f"\n{'=' * 50}")
    print("🏆 ИТОГИ ОБУЧЕНИЯ:")
    print(f"Лучшая модель: {best_model_info['model_name']}")
    print(f"Лучший F1-score (test): {best_model_info['best_f1']:.4f}")
    print(f"ID запуска MLflow: {best_model_info['run_id']}")
    print(f"Путь к модели: {best_model_info['model_path']}")
    
    return best_model_info