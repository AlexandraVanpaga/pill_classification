"""
Скрипт для обучения модели EfficientNet-B4.
"""

import torch
import torch.nn as nn
from datetime import datetime
import os
import json

from config import PATHS, MODEL_CONFIG, TRAINING_CONFIG
from src.data_preprocessing import create_dataloaders
from src.model_efficientnet import create_model


class ModelTrainer:
    """Класс для обучения модели"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, save_dir, config=None):
        if config is None:
            config = TRAINING_CONFIG
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.config = config
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'gap': [], 'lr': []}
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_gap = float('inf')
        self.patience = 0
        self.patience_limit = config['patience_limit']
        
        os.makedirs(save_dir, exist_ok=True)
    
    def train_one_epoch(self, epoch):
        """Обучает модель одну эпоху"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % self.config['log_batch_interval'] == 0:
                batch_acc = 100 * correct / total
                print(f"  Батч {batch_idx+1}/{len(self.train_loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.2f}%")
        
        return running_loss / len(self.train_loader), 100 * correct / total
    
    def validate(self):
        """Валидация модели"""
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
        
        return running_vloss / len(self.val_loader), 100 * vcorrect / vtotal
    
    def save_checkpoint(self, epoch, is_best=False):
        """Сохраняет checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': self.history['val_acc'][-1],
            'val_loss': self.history['val_loss'][-1],
            'gap': self.history['gap'][-1],
            'history': self.history
        }
        
        filename = self.config['model_filename'] if is_best else f'checkpoint_epoch_{epoch}.pt'
        model_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, model_path)
        return model_path
    
    def train(self):
        """Полный цикл обучения"""
        epochs = self.config['epochs']
        
        print(f"\n{'='*60}")
        print("ОБУЧЕНИЕ EfficientNet-B4")
        print(f"{'='*60}")
        print(f"Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Эпох: {epochs} | Early Stopping: {self.patience_limit}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"{'='*60}\nЭпоха {epoch+1}/{epochs}\n{'='*60}")
            
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            gap = train_acc - val_acc
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['gap'].append(gap)
            self.history['lr'].append(current_lr)
            
            self.scheduler.step()
            
            gap_status = "✅" if gap < 10 else "⚠️" if gap < 15 else "🔴"
            print(f"\n{'─'*60}\nИТОГИ ЭПОХИ {epoch+1}\n{'─'*60}")
            print(f"Train: Loss={train_loss:.4f} Acc={train_acc:.2f}%")
            print(f"Val:   Loss={val_loss:.4f} Acc={val_acc:.2f}%")
            print(f"{gap_status} Gap={gap:.2f}% | LR={current_lr:.7f}")
            
            if val_acc > self.best_val_acc or (val_acc == self.best_val_acc and gap < self.best_gap):
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_gap = gap
                self.patience = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Лучшая модель сохранена: Val Acc={val_acc:.2f}%, Gap={gap:.2f}%")
            else:
                self.patience += 1
                print(f"Без улучшений: {self.patience}/{self.patience_limit}")
            
            if self.patience >= self.patience_limit:
                print(f"\n{'='*60}\n⏹ EARLY STOPPING на эпохе {epoch+1}\n{'='*60}")
                break
            
            print()
        
        self._print_summary()
        self.save_history()
        return self.history
    
    def _print_summary(self):
        """Выводит итоговую статистику"""
        print(f"\n{'='*60}\nОБУЧЕНИЕ ЗАВЕРШЕНО\n{'='*60}")
        print(f"Окончание: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Обучено эпох: {len(self.history['train_acc'])}")
        print(f"\nЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print(f"  Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"  Val Loss: {self.best_val_loss:.4f}")
        print(f"  Gap: {self.best_gap:.2f}%")
        print(f"\nМодель: {self.save_dir}/{self.config['model_filename']}")
        print(f"{'='*60}")
    
    def save_history(self, filename='training_history.json'):
        """Сохраняет историю обучения"""
        history_path = os.path.join(self.save_dir, filename)
        history_data = {
            'history': self.history,
            'best_metrics': {
                'val_acc': self.best_val_acc,
                'val_loss': self.best_val_loss,
                'gap': self.best_gap
            }
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=4)
        print(f"✓ История сохранена: {history_path}")


def main():
    """Главная функция для запуска обучения"""
    
    print(f"\n{'='*60}")
    print("ИНИЦИАЛИЗАЦИЯ ОБУЧЕНИЯ")
    print(f"{'='*60}")
    print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Загрузка данных (без config)
    print(f"\n{'─'*60}")
    print("ЗАГРУЗКА ДАННЫХ")
    print(f"{'─'*60}")
    train_loader, val_loader, classes = create_dataloaders(
        extracted_path=PATHS['extracted_data'],
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    print(f"\nКоличество классов: {len(classes)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Создание модели
    print(f"\n{'─'*60}")
    print("СОЗДАНИЕ МОДЕЛИ")
    print(f"{'─'*60}")
    model = create_model(config=MODEL_CONFIG, device=device)
    
    # Создание optimizer и scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=MODEL_CONFIG['label_smoothing'])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=MODEL_CONFIG['weight_decay'],
        betas=MODEL_CONFIG['betas']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MODEL_CONFIG['scheduler_T_max'],
        eta_min=MODEL_CONFIG['scheduler_eta_min']
    )
    
    # Обучение
    print(f"\n{'─'*60}")
    print("ЗАПУСК ОБУЧЕНИЯ")
    print(f"{'─'*60}")
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=PATHS['models_dir'],
        config=TRAINING_CONFIG
    )
    
    history = trainer.train()
    
    print(f"\n{'='*60}")
    print("ВСЕ ГОТОВО!")
    print(f"{'='*60}")
    print(f"Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Лучшая Val Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Лучший Gap: {trainer.best_gap:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()