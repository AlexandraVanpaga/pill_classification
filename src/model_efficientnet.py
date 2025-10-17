"""
Модуль с архитектурой EfficientNet-B4 для классификации таблеток.
Включает защиту от переобучения: Dropout, BatchNorm, регуляризацию.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PillClassifierEfficientNetB4(nn.Module):
    """
    EfficientNet-B4 с усиленной регуляризацией для борьбы с переобучением
    
    Архитектура:
    - Pretrained EfficientNet-B4 backbone (ImageNet)
    - Частичная разморозка последних 3 блоков
    - Многослойный classifier с Dropout и BatchNorm
    - Постепенное уменьшение размерности: 1792 -> 896 -> 448 -> 224 -> num_classes
    
    Args:
        num_classes (int): количество классов для классификации
        dropout_rate (float): базовый уровень dropout (по умолчанию 0.5)
    """
    def __init__(self, num_classes=84, dropout_rate=0.5):
        super().__init__()
        
        # Загружаем pretrained EfficientNet-B4
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        
        # Замораживаем все слои backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Размораживаем последние 3 блока для fine-tuning
        total_blocks = len(self.backbone.features)
        unfreeze_from = total_blocks - 3
        
        for i in range(unfreeze_from, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True
        
        print(f"EfficientNet-B4 архитектура:")
        print(f"  Заморожено блоков: {unfreeze_from}/{total_blocks}")
        print(f"  Разморожено блоков: {total_blocks - unfreeze_from}/{total_blocks}")
        
        # Заменяем classifier с сильной регуляризацией
        num_features = self.backbone.classifier[1].in_features  # 1792 для B4
        
        self.backbone.classifier = nn.Sequential(
            # Слой 1: 1792 -> 896
            nn.Dropout(dropout_rate),           # 0.5
            nn.Linear(num_features, 896),
            nn.BatchNorm1d(896),
            nn.ReLU(),
            
            # Слой 2: 896 -> 448
            nn.Dropout(dropout_rate - 0.1),     # 0.4
            nn.Linear(896, 448),
            nn.BatchNorm1d(448),
            nn.ReLU(),
            
            # Слой 3: 448 -> 224
            nn.Dropout(dropout_rate - 0.2),     # 0.3
            nn.Linear(448, 224),
            nn.BatchNorm1d(224),
            nn.ReLU(),
            
            # Выходной слой: 224 -> num_classes
            nn.Dropout(dropout_rate - 0.3),     # 0.2
            nn.Linear(224, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: входной тензор изображений [batch_size, 3, 224, 224]
            
        Returns:
            логиты для каждого класса [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def get_num_params(self):
        """
        Возвращает количество параметров модели
        
        Returns:
            dict: {'total': всего, 'trainable': обучаемых, 'frozen': замороженных}
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_percent': trainable / total * 100
        }
    
    def unfreeze_more_layers(self, num_blocks=2):
        """
        Размораживает дополнительные блоки для fine-tuning
        
        Args:
            num_blocks (int): количество дополнительных блоков для разморозки
        """
        total_blocks = len(self.backbone.features)
        
        # Определяем текущее количество разморожженных блоков
        frozen_count = sum(1 for i in range(total_blocks) 
                          if not any(p.requires_grad for p in self.backbone.features[i].parameters()))
        
        # Размораживаем дополнительные блоки
        unfreeze_from = max(0, frozen_count - num_blocks)
        
        for i in range(unfreeze_from, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True
        
        print(f"Разморожено дополнительно {num_blocks} блоков")
        print(f"Теперь обучается с блока {unfreeze_from}/{total_blocks}")


def create_model(num_classes, dropout_rate=0.5, device='cuda'):
    """
    Фабричная функция для создания модели
    
    Args:
        num_classes (int): количество классов
        dropout_rate (float): уровень dropout
        device (str): устройство ('cuda' или 'cpu')
        
    Returns:
        model: готовая к обучению модель
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = PillClassifierEfficientNetB4(
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)
    
    # Выводим статистику
    params = model.get_num_params()
    
    print(f"\n{'='*60}")
    print("СТАТИСТИКА МОДЕЛИ")
    print(f"{'='*60}")
    print(f"Модель: EfficientNet-B4")
    print(f"Устройство: {device}")
    print(f"Всего параметров:      {params['total']:,}")
    print(f"Обучаемых параметров:  {params['trainable']:,}")
    print(f"Замороженных:          {params['frozen']:,}")
    print(f"Процент обучаемых:     {params['trainable_percent']:.1f}%")
    print(f"{'='*60}")
    
    print("\nМеры против переобучения:")
    print("  ✓ Dropout слои (0.5 -> 0.4 -> 0.3 -> 0.2)")
    print("  ✓ BatchNorm после каждого Linear")
    print("  ✓ Частичная разморозка (последние 3 блока)")
    print("  ✓ Постепенное уменьшение размерности")
    print("  ✓ Pretrained веса (ImageNet)")
    
    return model


def get_optimizer_and_scheduler(model, lr=0.001, weight_decay=0.05, epochs=40):
    """
    Создаёт оптимизатор и scheduler для модели
    
    Args:
        model: модель для обучения
        lr (float): начальный learning rate
        weight_decay (float): коэффициент L2 регуляризации
        epochs (int): количество эпох для scheduler
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ ОПТИМИЗАЦИИ")
    print(f"{'='*60}")
    print(f"Optimizer:       AdamW")
    print(f"Learning Rate:   {lr:.6f}")
    print(f"Weight Decay:    {weight_decay:.2f}")
    print(f"Scheduler:       CosineAnnealingLR (T_max={epochs})")
    print(f"{'='*60}\n")
    
    return optimizer, scheduler


if __name__ == "__main__":
    # Тестирование модуля
    print("Тестирование модуля model_efficientnet.py\n")
    
    # Создание модели
    num_classes = 84
    model = create_model(num_classes=num_classes, dropout_rate=0.5, device='cuda')
    
    # Создание оптимизатора
    optimizer, scheduler = get_optimizer_and_scheduler(model, lr=0.001, weight_decay=0.05)
    
    # Тестовый forward pass
    print("\nТестовый forward pass:")
    dummy_input = torch.randn(2, 3, 224, 224).to(next(model.parameters()).device)
    output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    print("\n✓ Модуль работает корректно!")