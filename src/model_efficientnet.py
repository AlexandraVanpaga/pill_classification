"""
Модуль с архитектурой EfficientNet-B4 для классификации таблеток.
Включает защиту от переобучения: Dropout, BatchNorm, регуляризацию.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_CONFIG


class PillClassifierEfficientNetB4(nn.Module):
    """
    EfficientNet-B4 с усиленной регуляризацией для борьбы с переобучением
    
    Архитектура:
    - Pretrained EfficientNet-B4 backbone (ImageNet)
    - Частичная разморозка последних N блоков
    - Многослойный classifier с Dropout и BatchNorm
    - Постепенное уменьшение размерности
    
    Args:
        config (dict): конфигурация модели (если None, используется MODEL_CONFIG)
    """
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = MODEL_CONFIG
        
        # Загружаем pretrained EfficientNet-B4
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        
        # Замораживаем все слои backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Размораживаем последние N блоков для fine-tuning
        total_blocks = len(self.backbone.features)
        unfreeze_from = total_blocks - config['unfreeze_last_n_blocks']
        
        for i in range(unfreeze_from, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True
        
        print(f"EfficientNet-B4 архитектура:")
        print(f"  Заморожено блоков: {unfreeze_from}/{total_blocks}")
        print(f"  Разморожено блоков: {total_blocks - unfreeze_from}/{total_blocks}")
        
        # Заменяем classifier с сильной регуляризацией
        num_features = self.backbone.classifier[1].in_features
        hidden_dims = config['classifier_hidden_dims']
        dropouts = config['classifier_dropouts']
        
        layers = []
        in_features = num_features
        
        # Создаем скрытые слои
        for i, (hidden_dim, dropout) in enumerate(zip(hidden_dims, dropouts[:-1])):
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            in_features = hidden_dim
        
        # Финальный слой
        layers.extend([
            nn.Dropout(dropouts[-1]),
            nn.Linear(in_features, config['num_classes'])
        ])
        
        self.backbone.classifier = nn.Sequential(*layers)
    
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
        
        # Определяем текущее количество замороженных блоков
        frozen_count = sum(1 for i in range(total_blocks) 
                          if not any(p.requires_grad for p in self.backbone.features[i].parameters()))
        
        # Размораживаем дополнительные блоки
        unfreeze_from = max(0, frozen_count - num_blocks)
        
        for i in range(unfreeze_from, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True
        
        print(f"Разморожено дополнительно {num_blocks} блоков")
        print(f"Теперь обучается с блока {unfreeze_from}/{total_blocks}")


def create_model(config=None, device='cuda'):
    """
    Фабричная функция для создания модели
    
    Args:
        config (dict): конфигурация модели (если None, используется MODEL_CONFIG)
        device (str): устройство ('cuda' или 'cpu')
        
    Returns:
        model: готовая к обучению модель
    """
    if config is None:
        config = MODEL_CONFIG
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = PillClassifierEfficientNetB4(config=config).to(device)
    
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
    print(f"  ✓ Dropout слои {config['classifier_dropouts']}")
    print("  ✓ BatchNorm после каждого Linear")
    print(f"  ✓ Частичная разморозка (последние {config['unfreeze_last_n_blocks']} блока)")
    print("  ✓ Постепенное уменьшение размерности")
    print("  ✓ Pretrained веса (ImageNet)")
    
    return model


def get_optimizer_and_scheduler(model, config=None):
    """
    Создаёт оптимизатор и scheduler для модели
    
    Args:
        model: модель для обучения
        config (dict): конфигурация (если None, используется MODEL_CONFIG)
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    if config is None:
        config = MODEL_CONFIG
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=config['betas']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['scheduler_T_max'],
        eta_min=config['scheduler_eta_min']
    )
    
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ ОПТИМИЗАЦИИ")
    print(f"{'='*60}")
    print(f"Optimizer:       AdamW")
    print(f"Learning Rate:   {config['learning_rate']:.6f}")
    print(f"Weight Decay:    {config['weight_decay']:.2f}")
    print(f"Scheduler:       CosineAnnealingLR (T_max={config['scheduler_T_max']})")
    print(f"{'='*60}\n")
    
    return optimizer, scheduler


if __name__ == "__main__":
    # Тестирование модуля
    print("Тестирование модуля model_efficientnet.py\n")
    
    # Создание модели
    model = create_model(config=MODEL_CONFIG, device='cuda')
    
    # Создание оптимизатора
    optimizer, scheduler = get_optimizer_and_scheduler(model, config=MODEL_CONFIG)
    
    # Тестовый forward pass
    print("\nТестовый forward pass:")
    dummy_input = torch.randn(2, 3, 224, 224).to(next(model.parameters()).device)
    output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    print("\n✓ Модуль работает корректно!")