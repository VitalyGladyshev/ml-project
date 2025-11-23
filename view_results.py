import webbrowser
import os
from pathlib import Path

def open_reports():
    """Открывает все отчеты в браузере"""
    print("Открываем отчеты в браузере...")
    
    # Открываем отчеты Deepchecks
    deepchecks_dir = Path("reports/deepchecks")
    if deepchecks_dir.exists():
        html_files = list(deepchecks_dir.glob("*.html"))
        if html_files:
            latest = max(html_files, key=lambda f: f.stat().st_mtime)
            print(f"Deepchecks отчет: {latest}")
            webbrowser.open(f"file://{latest.absolute()}")
    
    # Открываем отчеты EvidentlyAI
    evidently_dir = Path("reports/evidently")
    if evidently_dir.exists():
        html_files = list(evidently_dir.glob("*.html"))
        if html_files:
            latest = max(html_files, key=lambda f: f.stat().st_mtime)
            print(f"EvidentlyAI отчет: {latest}")
            webbrowser.open(f"file://{latest.absolute()}")
        
        # Открываем сводный отчет
        summary_files = list(evidently_dir.glob("summary_*.png"))
        if summary_files:
            latest_summary = max(summary_files, key=lambda f: f.stat().st_mtime)
            print(f"Сводная визуализация: {latest_summary}")
            webbrowser.open(f"file://{latest_summary.absolute()}")
    
    # Открываем папку с моделями
    models_dir = Path("models")
    if models_dir.exists() and any(models_dir.glob("*.pkl")):
        print(f"Модели сохранены в: {models_dir.absolute()}")
    
    # Открываем MLflow если есть данные
    mlruns_dir = Path("mlruns")
    if mlruns_dir.exists() and any(mlruns_dir.glob("*")):
        print("\nДля просмотра MLflow результатов выполните:")
        mlruns_path = os.getcwd().replace('\\', '/')
        print(f"mlflow ui --backend-store-uri file:///{mlruns_path}/mlruns")
    else:
        print("\nMLflow данные отсутствуют или не сохранились")

if __name__ == "__main__":
    open_reports()