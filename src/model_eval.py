"""
Модуль для оценки качества обученной модели на тестовых данных.
Включает тестирование с TTA и без, сохранение результатов.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime


class ModelEvaluator:
    """
    Класс для оценки модели на тестовых данных
    
    Args:
        model: обученная модель
        test_loader: DataLoader для тестовых данных
        classes: список названий классов
        device: устройство (cuda/cpu)
        save_dir: директория для сохранения результатов
    """
    def __init__(self, model, test_loader, classes, device, save_dir):
        self.model = model
        self.test_loader = test_loader
        self.classes = classes
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
    
    def evaluate_baseline(self):
        """
        Оценка модели без TTA (baseline)
        
        Returns:
            tuple: (labels_true, labels_predicted, accuracy)
        """
        print("="*60)
        print("ТЕСТИРОВАНИЕ БЕЗ TTA (BASELINE)")
        print("="*60)
        
        labels_predicted = []
        labels_true = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                labels_predicted.extend(predicted.cpu().numpy())
                labels_true.extend(labels.numpy())
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"Обработано батчей: {batch_idx + 1}/{len(self.test_loader)}")
        
        accuracy = accuracy_score(labels_true, labels_predicted)
        
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ БЕЗ TTA")
        print(f"{'='*60}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        
        if accuracy >= 0.75:
            print("\n🎉 ЦЕЛЬ ДОСТИГНУТА БЕЗ TTA!")
        else:
            gap = 75 - accuracy*100
            print(f"\nДо цели: {gap:.2f}%")
        
        print(f"{'='*60}\n")
        
        return labels_true, labels_predicted, accuracy
    
    def evaluate_with_tta(self, n_augmentations=4):
        """
        Оценка модели с Test-Time Augmentation
        
        Args:
            n_augmentations (int): количество аугментаций (4 или 10)
            
        Returns:
            tuple: (labels_true, labels_predicted, accuracy)
        """
        print("="*60)
        print(f"ТЕСТИРОВАНИЕ С TTA ({n_augmentations} АУГМЕНТАЦИЙ)")
        print("="*60)
        
        if n_augmentations == 4:
            print("Применяем аугментации:")
            print("  1. Оригинал")
            print("  2. Horizontal flip ↔")
            print("  3. Vertical flip ↕")
            print("  4. Both flips ⤡\n")
        
        labels_predicted = []
        labels_true = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                batch_predictions = []
                
                # 1. Оригинал
                imgs = images.to(self.device)
                outputs = self.model(imgs)
                probs = F.softmax(outputs, dim=1)
                batch_predictions.append(probs.cpu().numpy())
                
                # 2. Horizontal flip
                imgs_hflip = torch.flip(images, dims=[3]).to(self.device)
                outputs = self.model(imgs_hflip)
                probs = F.softmax(outputs, dim=1)
                batch_predictions.append(probs.cpu().numpy())
                
                # 3. Vertical flip
                imgs_vflip = torch.flip(images, dims=[2]).to(self.device)
                outputs = self.model(imgs_vflip)
                probs = F.softmax(outputs, dim=1)
                batch_predictions.append(probs.cpu().numpy())
                
                # 4. Both flips
                imgs_both = torch.flip(images, dims=[2, 3]).to(self.device)
                outputs = self.model(imgs_both)
                probs = F.softmax(outputs, dim=1)
                batch_predictions.append(probs.cpu().numpy())
                
                # Усредняем предсказания
                avg_probs = np.mean(batch_predictions, axis=0)
                predicted = np.argmax(avg_probs, axis=1)
                
                labels_predicted.extend(predicted)
                labels_true.extend(labels.numpy())
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"Обработано: {batch_idx + 1}/{len(self.test_loader)}")
        
        accuracy = accuracy_score(labels_true, labels_predicted)
        
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ С TTA")
        print(f"{'='*60}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        
        if accuracy >= 0.75:
            print("\n🎉🎉🎉 ЦЕЛЬ ДОСТИГНУТА С TTA! 🎉🎉🎉")
        else:
            gap = 75 - accuracy*100
            print(f"\nДо цели: {gap:.2f}%")
        
        print(f"{'='*60}\n")
        
        return labels_true, labels_predicted, accuracy
    
    def print_classification_report(self, labels_true, labels_predicted, title="Classification Report"):
        """
        Выводит подробный classification report
        
        Args:
            labels_true: истинные метки
            labels_predicted: предсказанные метки
            title: заголовок отчёта
        """
        print(f"\n{title}:\n")
        print(classification_report(labels_true, labels_predicted, 
                                   target_names=self.classes, digits=3))
    
    def plot_confusion_matrix(self, labels_true, labels_predicted, filename='confusion_matrix.png'):
        """
        Строит и сохраняет матрицу ошибок
        
        Args:
            labels_true: истинные метки
            labels_predicted: предсказанные метки
            filename: имя файла для сохранения
        """
        conf_matrix = confusion_matrix(labels_true, labels_predicted)
        
        # Берём топ-20 классов
        class_counts = conf_matrix.sum(axis=1)
        top_indices = np.argsort(class_counts)[-20:]
        
        conf_matrix_top = conf_matrix[top_indices][:, top_indices]
        classes_top = [self.classes[i] for i in top_indices]
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            conf_matrix_top, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=classes_top,
            yticklabels=classes_top,
            cbar_kws={'label': 'Count'}
        )
        
        accuracy = accuracy_score(labels_true, labels_predicted)
        plt.title(f'Confusion Matrix - Top 20 Classes\nTest Accuracy: {accuracy*100:.2f}%', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Матрица ошибок сохранена: {save_path}")
    
    def save_results(self, baseline_acc, tta_acc, labels_true_tta, labels_predicted_tta):
        """
        Сохраняет результаты оценки
        
        Args:
            baseline_acc: accuracy без TTA
            tta_acc: accuracy с TTA
            labels_true_tta: истинные метки (TTA)
            labels_predicted_tta: предсказанные метки (TTA)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Основные результаты
        results = {
            'model': 'EfficientNet-B4',
            'timestamp': timestamp,
            'test_baseline': {
                'accuracy': float(baseline_acc * 100),
                'method': 'Single prediction'
            },
            'test_tta': {
                'accuracy': float(tta_acc * 100),
                'method': 'TTA (4 augmentations)',
                'augmentations': ['original', 'h_flip', 'v_flip', 'both_flips']
            },
            'improvement': {
                'tta_boost': float((tta_acc - baseline_acc) * 100)
            },
            'goal_achieved': tta_acc >= 0.75,
            'dataset': {
                'num_classes': len(self.classes),
                'test_samples': len(labels_true_tta)
            }
        }
        
        results_path = os.path.join(self.save_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Результаты сохранены: {results_path}")
        
        # Classification report
        report_dict = classification_report(
            labels_true_tta, 
            labels_predicted_tta, 
            target_names=self.classes,
            output_dict=True
        )
        
        report_path = os.path.join(self.save_dir, 'classification_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Classification report: {report_path}")
        
        # Примеры предсказаний
        test_examples = []
        for i in range(min(100, len(labels_true_tta))):
            test_examples.append({
                'index': i,
                'true_label': self.classes[labels_true_tta[i]],
                'predicted_label': self.classes[labels_predicted_tta[i]],
                'correct': bool(labels_true_tta[i] == labels_predicted_tta[i])
            })
        
        examples_path = os.path.join(self.save_dir, 'test_examples.json')
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump(test_examples, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Примеры: {examples_path}")
    
    def full_evaluation(self):
        """
        Полная оценка модели: baseline + TTA + сохранение результатов
        
        Returns:
            dict: словарь с результатами
        """
        # Baseline
        labels_true_base, labels_pred_base, acc_base = self.evaluate_baseline()
        
        # TTA
        labels_true_tta, labels_pred_tta, acc_tta = self.evaluate_with_tta()
        
        # Сравнение
        print(f"{'='*60}")
        print(f"СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print(f"{'='*60}")
        print(f"БЕЗ TTA:  {acc_base*100:.2f}%")
        print(f"С TTA:    {acc_tta*100:.2f}%")
        print(f"{'='*60}")
        print(f"Улучшение: {(acc_tta - acc_base)*100:+.2f}%")
        print(f"{'='*60}\n")
        
        # Classification report
        self.print_classification_report(labels_true_tta, labels_pred_tta, 
                                        "Подробный Classification Report С TTA")
        
        # Confusion matrix
        self.plot_confusion_matrix(labels_true_tta, labels_pred_tta, 
                                   'confusion_matrix_tta.png')
        
        # Сохранение
        self.save_results(acc_base, acc_tta, labels_true_tta, labels_pred_tta)
        
        print(f"\nВсе результаты сохранены в: {self.save_dir}")
        
        return {
            'baseline_accuracy': acc_base,
            'tta_accuracy': acc_tta,
            'improvement': acc_tta - acc_base,
            'goal_achieved': acc_tta >= 0.75
        }


def evaluate_model(model, test_loader, classes, device, save_dir):
    """
    Удобная функция для оценки модели
    
    Args:
        model: обученная модель
        test_loader: DataLoader для тестовых данных
        classes: список классов
        device: устройство
        save_dir: директория для сохранения результатов
        
    Returns:
        dict: результаты оценки
    """
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        classes=classes,
        device=device,
        save_dir=save_dir
    )
    
    results = evaluator.full_evaluation()
    
    return results


def load_and_evaluate(model_path, model_class, num_classes, test_loader, 
                     classes, device, save_dir):
    """
    Загружает модель из checkpoint и оценивает её
    
    Args:
        model_path: путь к сохранённой модели
        model_class: класс модели
        num_classes: количество классов
        test_loader: DataLoader для тестов
        classes: список классов
        device: устройство
        save_dir: директория для результатов
        
    Returns:
        dict: результаты оценки
    """
    # Создаём модель
    model = model_class(num_classes=num_classes).to(device)
    
    # Загружаем веса
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Модель загружена: {model_path}")
    print(f"  Эпоха: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%\n")
    
    # Оценка
    results = evaluate_model(model, test_loader, classes, device, save_dir)
    
    return results


if __name__ == "__main__":
    print("Модуль evaluate.py готов к использованию")
    print("\nПример использования:")
    print("""
from src.model_efficientnet import PillClassifierEfficientNetB4
from src.prepare_dataloaders import create_test_loader
from src.evaluate import load_and_evaluate
from config import PATHS

# Загрузка тестовых данных
test_loader, classes = create_test_loader(PATHS['extracted_data'])

# Оценка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = load_and_evaluate(
    model_path=PATHS['best_model'],
    model_class=PillClassifierEfficientNetB4,
    num_classes=len(classes),
    test_loader=test_loader,
    classes=classes,
    device=device,
    save_dir=PATHS['results_dir']
)
    """)