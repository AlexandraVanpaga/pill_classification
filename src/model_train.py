"""
Модуль для обучения модели EfficientNet-B4 с мониторингом Gap и Early Stopping.
"""

import torch
import torch.nn as nn
from datetime import datetime
import os
import json


class ModelTrainer:
    """
    Класс для обучения модели классификации таблеток
    
    Args:
        model: модель для обучения
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
        criterion: функция потерь
        optimizer: оптимизатор
        scheduler: scheduler для learning rate
        device: устройство (cuda/cpu)
        save_dir: директория для сохранения моделей
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        # История обучения
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'gap': [],
            'lr': []
        }
        
        # Лучшие метрики
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_gap = float('inf')
        
        # Early stopping
        self.patience = 0
        self.patience_limit = 10
        
        # Создаём директорию для сохранения
        os.makedirs(save_dir, exist_ok=True)
    
    def train_one_epoch(self, epoch):
        """
        Обучает модель одну эпоху
        
        Args:
            epoch (int): номер текущей эпохи
            
        Returns:
            tuple: (train_loss, train_acc)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Промежуточный вывод каждые 10 батчей
            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100 * correct / total
                print(f"  Батч {batch_idx+1}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f}, Acc={batch_acc:.2f}%")
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100 * correct / total
        
        return train_loss, train_acc
    
    def validate(self):
        """
        Валидация модели
        
        Returns:
            tuple: (val_loss, val_acc)
        """
        self.model.eval()
        running_vloss = 0.0
        vcorrect = 0
        vtotal = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_vloss += loss.item()
                _, predicted = torch.max(outputs, 1)
                vtotal += labels.size(0)
                vcorrect += (predicted == labels).sum().item()
        
        val_loss = running_vloss / len(self.val_loader)
        val_acc = 100 * vcorrect / vtotal
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Сохраняет checkpoint модели
        
        Args:
            epoch (int): номер эпохи
            is_best (bool): является ли это лучшей моделью
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': self.history['val_acc'][-1],
            'val_loss': self.history['val_loss'][-1],
            'gap': self.history['gap'][-1],
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_gap': self.best_gap
        }
        
        if is_best:
            model_path = os.path.join(self.save_dir, 'best_efficientnet_b4.pt')
            torch.save(checkpoint, model_path)
            return model_path
        else:
            model_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, model_path)
            return model_path
    
    def train(self, epochs=40):
        """
        Полный цикл обучения
        
        Args:
            epochs (int): количество эпох
            
        Returns:
            dict: история обучения
        """
        print(f"\n{'='*60}")
        print("ОБУЧЕНИЕ EfficientNet-B4")
        print(f"{'='*60}")
        print(f"Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Эпох: {epochs}")
        print(f"Early Stopping: {self.patience_limit} эпох")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"{'='*60}")
            print(f"Эпоха {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Обучение
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Валидация
            val_loss, val_acc = self.validate()
            
            # Метрики
            gap = train_acc - val_acc
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['gap'].append(gap)
            self.history['lr'].append(current_lr)
            
            # Обновление Learning Rate
            self.scheduler.step()
            
            # Вывод
            gap_status = "✅" if gap < 10 else "⚠️" if gap < 15 else "🔴"
            
            print(f"\n{'─'*60}")
            print(f"ИТОГИ ЭПОХИ {epoch+1}")
            print(f"{'─'*60}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"{gap_status} Gap:      {gap:.2f}%")
            print(f"Learning Rate: {current_lr:.7f}")
            
            # Сохранение лучшей модели
            improved = False
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_gap = gap
                self.patience = 0
                improved = True
                
                model_path = self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Лучшая модель сохранена!")
                print(f"  Val Acc: {val_acc:.2f}%, Gap: {gap:.2f}%")
            
            elif val_acc == self.best_val_acc and gap < self.best_gap:
                self.best_gap = gap
                self.patience = 0
                improved = True
                
                model_path = self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Модель с меньшим Gap сохранена!")
                print(f"  Val Acc: {val_acc:.2f}%, Gap: {gap:.2f}%")
            
            else:
                self.patience += 1
                print(f"Без улучшений: {self.patience}/{self.patience_limit}")
            
            # Early stopping
            if self.patience >= self.patience_limit:
                print(f"\n{'='*60}")
                print(f"⏹ EARLY STOPPING на эпохе {epoch+1}")
                print(f"Нет улучшений {self.patience_limit} эпох подряд")
                print(f"{'='*60}")
                break
            
            # Предупреждения
            if gap > 20:
                print(f"\n🚨 ВНИМАНИЕ: Gap > 20% - критическое переобучение!")
            
            if val_acc >= 75:
                print(f"\n🎯 Достигнута цель 75%!")
            
            print()
        
        # Итоги обучения
        self._print_summary()
        
        return self.history
    
    def _print_summary(self):
        """Выводит итоговую статистику обучения"""
        print(f"\n{'='*60}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*60}")
        print(f"Окончание: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Обучено эпох: {len(self.history['train_acc'])}")
        print(f"\nЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print(f"  Val Accuracy:  {self.best_val_acc:.2f}%")
        print(f"  Val Loss:      {self.best_val_loss:.4f}")
        print(f"  Gap:           {self.best_gap:.2f}%")
        
        if self.best_val_acc >= 75:
            print(f"\n🎉 Цель достигнута! (>= 75%)")
        if self.best_gap < 10:
            print(f"🏆 Отличная генерализация (Gap < 10%)")
        elif self.best_gap < 15:
            print(f"✅ Хорошая генерализация (Gap < 15%)")
        
        print(f"\nМодель сохранена: {self.save_dir}/best_efficientnet_b4.pt")
        print(f"{'='*60}")
    
    def save_history(self, filename='training_history.json'):
        """
        Сохраняет историю обучения в JSON
        
        Args:
            filename (str): имя файла
        """
        history_path = os.path.join(self.save_dir, filename)
        
        history_data = {
            'history': self.history,
            'best_metrics': {
                'val_acc': self.best_val_acc,
                'val_loss': self.best_val_loss,
                'gap': self.best_gap
            },
            'training_info': {
                'total_epochs': len(self.history['train_acc']),
                'early_stopped': self.patience >= self.patience_limit
            }
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        print(f"✓ История сохранена: {history_path}")


def train_model(model, train_loader, val_loader, device, save_dir, 
                epochs=40, lr=0.001, weight_decay=0.05, label_smoothing=0.15):
    """
    Удобная функция для запуска обучения
    
    Args:
        model: модель для обучения
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
        device: устройство
        save_dir: директория для сохранения
        epochs (int): количество эпох
        lr (float): learning rate
        weight_decay (float): weight decay
        label_smoothing (float): label smoothing
        
    Returns:
        tuple: (trainer, history)
    """
    # Создаём criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Создаём optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Создаём scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # Создаём trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir
    )
    
    # Запускаем обучение
    history = trainer.train(epochs=epochs)
    
    # Сохраняем историю
    trainer.save_history()
    
    return trainer, history


if __name__ == "__main__":
    print("Модуль train.py готов к использованию")
    print("\nПример использования:")
    print("""
from src.model_efficientnet import create_model
from src.prepare_dataloaders import create_dataloaders
from src.train import train_model
from config import PATHS

# Загрузка данных
train_loader, val_loader, classes = create_dataloaders(PATHS['extracted_data'])

# Создание модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=len(classes), device=device)

# Обучение
trainer, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    save_dir=PATHS['models_dir'],
    epochs=40,
    lr=0.001,
    weight_decay=0.05
)
    """)