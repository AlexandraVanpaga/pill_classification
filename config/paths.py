import os

BASE_DIR = r'/home/ubuntu/pill_classification'
PATHS = {
    # Сырые данные
    'raw_data': os.path.join(BASE_DIR, 'dataset', 'raw_dataset.zip'),
    
    # Разархивированные данные (исходные)
    'extracted_data': os.path.join(BASE_DIR, 'dataset', 'extracted'),
    
    # Обработанные данные (после препроцессинга)
    'processed_data': os.path.join(BASE_DIR, 'dataset', 'processed'),
    'train_dataset': os.path.join(BASE_DIR, 'dataset', 'processed', 'train'),
    'test_dataset': os.path.join(BASE_DIR, 'dataset', 'processed', 'test'),
    
    # Модели
    'models_dir': os.path.join(BASE_DIR, 'models'),
    'best_model': os.path.join(BASE_DIR, 'models', 'meds_classifier.pt'),
    
    # Результаты
    'results_dir': os.path.join(BASE_DIR, 'results'),
    'test_examples': os.path.join(BASE_DIR, 'results', 'test_examples.json'),
}