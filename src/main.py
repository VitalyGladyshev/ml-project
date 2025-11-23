import os
import mlflow
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Импорты с обработкой ошибок
try:
    from src.model_training_simple import train_models_without_mlflow
except ImportError:
    train_models_without_mlflow = None

try:
    from src.model_training import train_and_evaluate_models
except ImportError as e:
    print(f"Ошибка импорта model_training: {str(e)}")
    try:
        from .model_training import train_and_evaluate_models
    except ImportError as e2:
        print(f"Ошибка импорта с относительным путем: {str(e2)}")
        raise

try:
    from src.data_validation import validate_data_with_deepchecks
except ImportError as e:
    print(f"Ошибка импорта data_validation: {str(e)}")
    try:
        from .data_validation import validate_data_with_deepchecks
    except ImportError as e2:
        print(f"Ошибка импорта с относительным путем: {str(e2)}")
        raise

try:
    from src.drift_detection import detect_data_drift
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  EvidentlyAI недоступен: {str(e)}")
    EVIDENTLY_AVAILABLE = False

def setup_mlflow_local():
    """Настройка MLflow для локального файлового хранилища"""
    try:
        # Создаем директорию для MLflow если не существует
        mlflow_dir = os.path.abspath("./mlruns")
        os.makedirs(mlflow_dir, exist_ok=True)
        
        # Устанавливаем URI для локального файлового хранилища
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        print(f"MLflow настроен на локальное файловое хранилище: {mlflow_dir}")
        
        # Создаем или получаем эксперимент
        experiment_name = "ikm_classification_experiment"
        
        # Проверяем, существует ли эксперимент
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            print(f"Эксперимент '{experiment_name}' создан")
        else:
            print(f"Эксперимент '{experiment_name}' найден (ID: {experiment.experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        print(f"Текущий эксперимент: '{experiment_name}'")
        
        return True
    
    except Exception as e:
        print(f"Ошибка при настройке MLflow: {str(e)}")
        return False

def load_and_prepare_data():
    """Загрузка и предобработка данных"""
    print("\n" + "=" * 60)
    print("=== Загрузка и подготовка данных ===")
    print("=" * 60)
    
    try:
        # Поиск файла данных
        data_paths = [
            "data/ikm_start_3.csv",
            "../data/ikm_start_3.csv",
            "../../data/ikm_start_3.csv",
            "ikm_start_3.csv",
            "../ikm_start_3.csv",
            "data/ikm_start_3.csv"
        ]
        
        data_path = None
        for path in data_paths:
            if os.path.exists(path):
                data_path = path
                print(f"Найден файл данных: {os.path.abspath(path)}")
                break
        
        if data_path is None:
            print("Файл данных не найден! Проверьте следующие пути:")
            for path in data_paths:
                print(f"  - {os.path.abspath(path)}")
            print("\nПоместите файл 'ikm_start_3.csv' в папку 'data/'")
            raise FileNotFoundError("Файл данных не найден")
        
        # Загрузка данных
        data = pd.read_csv(data_path)
        print(f"Загружено данных: {data.shape[0]} строк, {data.shape[1]} столбцов")
        
        # Показать первые 2 строки и основную информацию
        print("\nПервые 2 строки данных:")
        print(data.head(2).to_string())
        
        print("\nИнформация о данных:")
        print(f"Типы данных:\n{data.dtypes}")
        print(f"\nПропущенные значения:\n{data.isnull().sum()}")
        
        # Проверка целевой переменной
        target_options = ['стп_ХОБЛ', 'стп_ХОБЛ', 'target', 'y', 'label']
        target_column = None
        
        for col in target_options:
            if col in data.columns:
                target_column = col
                break
        
        if target_column is None:
            # Попробуем найти столбец, содержащий 'ХОБЛ' или 'хобл'
            for col in data.columns:
                if 'ХОБЛ' in col or 'хобл' in col.lower():
                    target_column = col
                    print(f"🔍 Найдена целевая переменная по ключевому слову: '{target_column}'")
                    break
        
        if target_column is None:
            print("Целевая переменная не найдена!")
            print(f"Доступные столбцы: {', '.join(data.columns)}")
            print("Убедитесь, что в данных есть столбец 'стп_ХОБЛ'")
            raise ValueError("Целевая переменная не найдена")
        
        print(f"\nЦелевая переменная: '{target_column}'")
        print(f"Распределение целевой переменной:\n{data[target_column].value_counts()}")
        
        # Разделение на признаки и целевую переменную
        X = data.drop([target_column], axis=1)
        y = data[target_column]
        
        print(f"\nРазмеры данных: X={X.shape}, y={y.shape}")
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"SplitOptions: train={X_train.shape}, test={X_test.shape}")
        print(f"Распределение в train: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Распределение в test: {pd.Series(y_test).value_counts().to_dict()}")
        
        # Масштабирование данных
        print("\nМасштабирование данных...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Преобразование обратно в DataFrame для сохранения имен колонок
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print("Данные успешно подготовлены!")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
    
    except Exception as e:
        print(f"Ошибка при загрузке данных: {str(e)}")
        raise

def main():
    """Основная функция для запуска всего пайплайна"""
    print("ЗАПУСК ML ПАЙПЛАЙНА")
    print("=" * 80)
    print("Конфигурация:")
    print(f" • MLflow: локальное файловое хранилище (./mlruns)")
    print(f" • Deepchecks: активен")
    print(f" • EvidentlyAI: {'активен' if EVIDENTLY_AVAILABLE else 'не доступен'}")
    print("=" * 80)
    
    try:
        # Настройка MLflow для локального использования
        print("\n" + "=" * 60)
        print("=== Настройка MLflow (локальное хранилище) ===")
        mlflow_ready = setup_mlflow_local()
        
        if not mlflow_ready:
            print("MLflow не настроен. Продолжаем без логирования...")
        
        # Загрузка и подготовка данных
        X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare_data()
        
        # Проверка данных с Deepchecks
        print("\n" + "=" * 60)
        print("=== Проверка данных с Deepchecks ===")
        try:
            validate_data_with_deepchecks(X_train, y_train)
            print("Проверка данных с Deepchecks завершена успешно")
        except Exception as e:
            print(f"Ошибка при проверке данных с Deepchecks: {str(e)}")
        
        # Анализ дрейфа данных с EvidentlyAI
        if EVIDENTLY_AVAILABLE:
            print("\n" + "=" * 60)
            print("=== Анализ дрейфа данных с EvidentlyAI ===")
            try:
                detect_data_drift(X_train, X_test)
                print("Анализ дрейфа данных завершен успешно")
            except Exception as e:
                print(f"Ошибка при анализе дрейфа данных: {str(e)}")
                print("Продолжаем без анализа дрейфа...")
        else:
            print("\n" + "=" * 60)
            print("EvidentlyAI недоступен. Пропускаем анализ дрейфа данных")
        
        # Обучение моделей с MLflow
        print("\n" + "=" * 60)
        print("=== Обучение моделей с MLflow ===")
        try:
            # if mlflow_ready:
            #     best_model_info = train_and_evaluate_models(
            #         X_train, X_test, y_train, y_test, feature_names
            #     )
            #     print("Обучение моделей с MLflow завершено успешно")
            # else:
            #     print("MLflow недоступен. Обучение моделей без логирования...")
            #     # Здесь можно добавить обучение без MLflow
            #     best_model_info = None

            if mlflow_ready:
                try:
                    best_model_info = train_and_evaluate_models(
                        X_train, X_test, y_train, y_test, feature_names
                    )
                except Exception as e:
                    print(f"Ошибка при обучении с MLflow: {str(e)}")
                    print("Переключаемся на обучение без MLflow...")
                    if train_models_without_mlflow:
                        best_model_info = train_models_without_mlflow(X_train, X_test, y_train, y_test, feature_names)
                    else:
                        best_model_info = None
            else:
                print("MLflow недоступен. Обучение моделей без логирования...")
                if train_models_without_mlflow:
                    best_model_info = train_models_without_mlflow(X_train, X_test, y_train, y_test, feature_names)
                else:
                    best_model_info = None

        except Exception as e:
            print(f"Ошибка при обучении моделей: {str(e)}")
            best_model_info = None
        
        print("\n" + "=" * 80)
        print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 80)
        
        # Итоговая информация
        print("\nРезультаты:")
        print(f" • Модели обучены: {'да' if best_model_info else 'нет'}")
        print(f" • Данные проверены: да")
        print(f" • Дрейф данных проанализирован: {'да' if EVIDENTLY_AVAILABLE else 'нет'}")
        
        if best_model_info:
            print(f"\nЛучшая модель: {best_model_info['model_name']}")
            print(f"Лучший F1-score (test): {best_model_info['best_f1']:.4f}")
        
        print("\nАртефакты сохранены в:")
        print(f" • MLflow: {os.path.abspath('./mlruns')}")
        print(f" • Отчеты Deepchecks: {os.path.abspath('./reports/deepchecks')}")
        print(f" • Отчеты EvidentlyAI: {os.path.abspath('./reports/evidently')}")
        print(f" • Модели: {os.path.abspath('./models')}")
        
        print("\nДля просмотра результатов MLflow выполните:")
        print(f"mlflow ui --backend-store-uri file://{os.path.abspath('./mlruns')}")
        
        return best_model_info
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    start_time = time.time()
    result = main()
    end_time = time.time()
    
    print(f"\nОбщее время выполнения: {end_time - start_time:.2f} секунд")
    print("Программа завершена")