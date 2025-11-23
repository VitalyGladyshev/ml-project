import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
    ColumnSummaryMetric
)
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def detect_data_drift(reference_data, current_data):
    """Анализ дрейфа данных с помощью EvidentlyAI"""
    print("Запуск анализа дрейфа данных с EvidentlyAI...")
    
    # Создание директорий для отчетов
    os.makedirs("reports/evidently", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Создание отчета о дрейфе данных
    print("   Создание основного отчета о дрейфе...")
    data_drift_report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])
    
    # Запуск отчета
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Сохранение отчета
    report_path = f"reports/evidently/data_drift_{timestamp}.html"
    data_drift_report.save_html(report_path)
    print(f"Основной отчет о дрейфе сохранен: {os.path.abspath(report_path)}")
    
    # Детальный анализ для числовых признаков
    numerical_features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nДетальный анализ для {len(numerical_features)} числовых признаков...")
    
    drift_results = []
    
    for i, feature in enumerate(numerical_features[:10]):  # Анализ первых 10 признаков
        print(f"   Анализ признака: {feature} ({i+1}/{min(10, len(numerical_features))})")
        
        # Создание отчета для отдельного признака
        feature_report = Report(metrics=[
            ColumnDriftMetric(column_name=feature),
            ColumnSummaryMetric(column_name=feature)
        ])
        
        feature_report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Сохранение отчета для признака
        feature_path = f"reports/evidently/{feature}_drift_{timestamp}.html"
        feature_report.save_html(feature_path)
        
        # Получение результатов
        report_dict = feature_report.as_dict()
        drift_score = None
        drift_detected = False
        
        for metric in report_dict.get('metrics', []):
            if metric.get('metric') == 'ColumnDriftMetric':
                result = metric.get('result', {})
                drift_score = result.get('drift_score')
                drift_detected = result.get('drift_detected', False)
                break
        
        if drift_score is not None:
            drift_results.append({
                'feature': feature,
                'drift_score': drift_score,
                'drift_detected': drift_detected,
                'report_path': feature_path
            })
    
    # Создание сводного отчета
    print("\nСоздание сводного отчета...")
    create_summary_report(drift_results, reference_data, current_data, timestamp)
    
    # Вывод ключевых результатов
    print("\nКлючевые выводы по анализу дрейфа:")
    if drift_results:
        significant_drift = [r for r in drift_results if r['drift_score'] > 0.1 or r['drift_detected']]
        
        if significant_drift:
            print(f"🚨 Обнаружен значительный дрейф в {len(significant_drift)} признаках:")
            for result in sorted(significant_drift, key=lambda x: x['drift_score'], reverse=True)[:5]:
                print(f"   • {result['feature']}: drift_score={result['drift_score']:.4f}, drift_detected={result['drift_detected']}")
        else:
            print("Значительного дрейфа данных не обнаружено")
            print("   Все признаки показывают стабильное распределение")
    else:
        print("Недостаточно числовых признаков для детального анализа")
    
    print(f"\nВсе отчеты сохранены в: {os.path.abspath('reports/evidently/')}")
    return data_drift_report

def create_summary_report(drift_results, reference_data, current_data, timestamp):
    """Создание сводного отчета с визуализацияциями"""
    if not drift_results:
        return
    
    plt.figure(figsize=(15, 10))
    
    # 1. График drift scores
    plt.subplot(2, 2, 1)
    features = [r['feature'] for r in drift_results]
    scores = [r['drift_score'] for r in drift_results]
    
    bars = plt.barh(features, scores, color=['red' if s > 0.1 else 'green' for s in scores])
    plt.axvline(x=0.1, color='orange', linestyle='--', label='Порог (0.1)')
    plt.xlabel('Drift Score')
    plt.title('Drift Score по признакам')
    plt.legend()
    
    # Добавление значений на бары
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{scores[i]:.4f}', 
                ha='left', va='center')
    
    # 2. Сравнение распределений для признака с наибольшим дрейфом
    if drift_results:
        max_drift = max(drift_results, key=lambda x: x['drift_score'])
        feature = max_drift['feature']
        
        plt.subplot(2, 2, 2)
        sns.histplot(reference_data[feature], kde=True, color='blue', alpha=0.5, label='Reference')
        sns.histplot(current_data[feature], kde=True, color='red', alpha=0.5, label='Current')
        plt.title(f'Распределение: {feature}\nDrift Score: {max_drift["drift_score"]:.4f}')
        plt.legend()
    
    # 3. Boxplot для сравнения
    plt.subplot(2, 2, 3)
    data_to_plot = pd.DataFrame({
        'value': pd.concat([reference_data[feature], current_data[feature]]),
        'dataset': ['Reference'] * len(reference_data) + ['Current'] * len(current_data)
    })
    sns.boxplot(x='dataset', y='value', data=data_to_plot)
    plt.title(f'Boxplot: {feature}')
    
    # 4. Статистика
    plt.subplot(2, 2, 4)
    stats_data = {
        'Статистика': ['Среднее (Reference)', 'Среднее (Current)', 'Стандартное отклонение (Reference)', 'Стандартное отклонение (Current)'],
        'Значение': [
            reference_data[feature].mean(),
            current_data[feature].mean(),
            reference_data[feature].std(),
            current_data[feature].std()
        ]
    }
    stats_df = pd.DataFrame(stats_data)

    col_labels = list(stats_df.columns)
    plt.table(
        cellText=stats_df.values,
        colLabels=col_labels,
        loc='center'
    )
    plt.axis('off')
    plt.title(f'Статистика: {feature}')
    
    plt.tight_layout()
    summary_path = f"reports/evidently/summary_drift_{timestamp}.png"
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    
    print(f"Сводная визуализация сохранена: {os.path.abspath(summary_path)}")
