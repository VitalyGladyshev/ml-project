from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
import pandas as pd
import os
import json
from datetime import datetime

def validate_data_with_deepchecks(X, y):
    """Проверка качества данных с помощью Deepchecks"""
    print("Запуск проверки данных с Deepchecks...")
    
    # Создание Dataset для Deepchecks
    df = pd.concat([X.reset_index(drop=True), pd.Series(y, name='target').reset_index(drop=True)], axis=1)
    
    # Определение категориальных признаков
    cat_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].nunique() < 10:
            cat_features.append(col)
    
    print(f"Обнаружено категориальных признаков: {len(cat_features)}")
    if cat_features:
        print(f"   • {', '.join(cat_features[:5])}{'...' if len(cat_features) > 5 else ''}")
    
    dataset = Dataset(df, label='target', cat_features=cat_features)
    
    # Запуск проверки целостности данных
    print("   Выполнение проверок...")
    suite = data_integrity()
    suite_result = suite.run(dataset)
    
    # Создание директорий для отчетов
    os.makedirs("reports/deepchecks", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Сохранение отчета
    report_path = f"reports/deepchecks/data_validation_{timestamp}.html"
    suite_result.save_as_html(report_path)
    
    # Сохранение результатов в JSON
    json_path = f"reports/deepchecks/data_validation_{timestamp}.json"
    save_results_to_json(suite_result, json_path)
    
    print(f"Отчет Deepchecks сохранен: {os.path.abspath(report_path)}")
    print(f"JSON отчет сохранен: {os.path.abspath(json_path)}")
    
    # Анализ результатов
    print("\nКлючевые выводы из проверки данных:")
    analyze_suite_results(suite_result)
    
    return suite_result

def save_results_to_json(suite_result, json_path: str):
    """Сохранение результатов в JSON формате"""
    results_data = []
    
    for check_result in suite_result.results:
        result_item = {
            "name": check_result.get_header(),
            "status": str(check_result.get_result()),
            "summary": check_result.get_description() if hasattr(check_result, 'get_description') else "",
            "conditions_results": [],
            "details": {}
        }
        
        # Получение результатов условий
        if hasattr(check_result, 'conditions_results'):
            for condition in check_result.conditions_results:
                result_item["conditions_results"].append({
                    "name": condition.name,
                    "status": str(condition.status),
                    "details": condition.details
                })
        
        # Получение детальной информации
        try:
            if hasattr(check_result, 'value'):
                result_item["details"]["value"] = str(check_result.value)
            if hasattr(check_result, 'display') and check_result.display:
                result_item["details"]["display_items"] = len(check_result.display)
        except Exception as e:
            result_item["details"]["error"] = str(e)
        
        results_data.append(result_item)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

def analyze_suite_results(suite_result):
    """Анализ результатов проверок Deepchecks"""
    failed_checks = 0
    warning_checks = 0
    passed_checks = 0
    
    for check_result in suite_result.results:
        status = str(check_result.get_result())
        
        if status == 'FAIL':
            failed_checks += 1
            print(f"Ошибка {check_result.get_header()}: FAIL")
            if hasattr(check_result, 'conditions_results'):
                for cond in check_result.conditions_results:
                    if cond.status == 'FAIL':
                        print(f"   • {cond.name}: {cond.details}")
        
        elif status == 'WARN':
            warning_checks += 1
            print(f"Предупреждение {check_result.get_header()}: WARN")
            if hasattr(check_result, 'conditions_results'):
                for cond in check_result.conditions_results:
                    if cond.status == 'WARN':
                        print(f"   • {cond.name}: {cond.details}")
        
        elif status == 'PASS':
            passed_checks += 1
    
    print(f"\nСтатистика проверок:")
    print(f"   • Пройдено: {passed_checks}")
    print(f"   • Предупреждений: {warning_checks}")
    print(f"   • Не пройдено: {failed_checks}")
    
    if failed_checks > 0:
        print("\nОбнаружены критические проблемы с данными!")
        print("   Рекомендуется исправить их перед обучением модели")
    elif warning_checks > 0:
        print("\nОбнаружены предупреждения")
        print("   Рекомендуется проверить данные на эти проблемы")
    else:
        print("\nВсе проверки пройдены успешно!")
        print("   Данные готовы для обучения модели")
