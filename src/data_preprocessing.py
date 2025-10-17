"""
Модуль для создания DataLoader для обучения и валидации модели классификации таблеток.
С усиленной аугментацией для малого датасета.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path


def get_transforms():
    """
    Возвращает трансформации для обучающей и валидационной выборок
    С усиленной аугментацией для малого датасета
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        # Resize с запасом для crop
        transforms.Resize((256, 256)),
        
        # Случайный crop (добавляет вариативность)
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        
        # Геометрические трансформации
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        
        # Цветовые трансформации
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1
        ),
        
        # Размытие
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.2),
        
        # Перспектива
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Конвертация в тензор
        transforms.ToTensor(),
        
        # Нормализация ImageNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        
        # Random Erasing (последний шаг!)
        transforms.RandomErasing(
            p=0.3, 
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3)
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def create_dataloaders(extracted_path, batch_size=32, val_split=0.2, random_seed=42):
    """
    Создаёт DataLoader для обучающей и валидационной выборок
    
    Структура данных:
    train/
      ├── urzinol/
      │   ├── urzinol_s_025.jpg
      │   └── ...
      ├── strepsils/
      │   ├── strepsils_001.jpg
      │   └── ...
      └── ...
    Названием класса является название папки!
    
    Args:
        extracted_path (str): путь к папке с данными
        batch_size (int): размер батча (по умолчанию 32)
        val_split (float): доля валидационной выборки (от 0 до 1, по умолчанию 0.2)
        random_seed (int): seed для воспроизводимости (по умолчанию 42)
        
    Returns:
        tuple: (train_loader, val_loader, classes)
            - train_loader: DataLoader для обучающей выборки
            - val_loader: DataLoader для валидационной выборки
            - classes: список названий классов
    
    Raises:
        FileNotFoundError: если директория с данными не найдена
    """
    extracted_path = Path(extracted_path)
    
    # Путь к папке train
    train_dir = extracted_path / 'ogyeiv2' / 'train'
    
    # Проверяем существование директории
    if not train_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {train_dir}")
    
    # Получаем трансформации
    train_transform, val_transform = get_transforms()
    
    # ImageFolder автоматически определяет классы по названиям папок
    full_dataset = datasets.ImageFolder(root=train_dir)
    
    # Выводим найденные классы
    print(f"Найдено классов: {len(full_dataset.classes)}")
    print(f"Первые 10 классов: {full_dataset.classes[:10]}")
    
    # Разделяем на train и val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Создаём отдельные датасеты с разными трансформациями
    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root=train_dir, transform=train_transform),
        train_indices.indices
    )
    
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root=train_dir, transform=val_transform),
        val_indices.indices
    )
    
    # Получаем классы
    classes = full_dataset.classes
    
    # Создаём DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, classes


def create_test_loader(extracted_path, batch_size=32):
    """
    Создаёт DataLoader для тестовой выборки
    
    Args:
        extracted_path (str): путь к папке с данными
        batch_size (int): размер батча
        
    Returns:
        tuple: (test_loader, classes)
    """
    extracted_path = Path(extracted_path)
    test_dir = extracted_path / 'ogyeiv2' / 'test'
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {test_dir}")
    
    # Используем val_transform (без аугментации)
    _, test_transform = get_transforms()
    
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader, test_dataset.classes


def print_dataset_info(train_loader, val_loader, classes):
    """
    Выводит информацию о датасете
    
    Args:
        train_loader: DataLoader для обучающей выборки
        val_loader: DataLoader для валидационной выборки
        classes: список названий классов
    """
    num_classes = len(classes)
    num_train_images = len(train_loader.dataset)
    num_val_images = len(val_loader.dataset)
    
    print("="*60)
    print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("="*60)
    print(f"Количество классов: {num_classes}")
    print(f"\nПримеры классов (первые 20):")
    for i, class_name in enumerate(classes[:20], 1):
        print(f"  {i}. {class_name}")
    if num_classes > 20:
        print(f"  ... и ещё {num_classes - 20} классов")
    
    print(f"\nКоличество изображений в train: {num_train_images}")
    print(f"Количество изображений в val: {num_val_images}")
    print(f"\nКоличество батчей в train_loader: {len(train_loader)}")
    print(f"Количество батчей в val_loader: {len(val_loader)}")
    
    # Проверяем один батч
    images, labels = next(iter(train_loader))
    print(f"\nПроверка батча:")
    print(f"  Размер батча изображений: {images.shape}")
    print(f"  Размер батча меток: {labels.shape}")
    print(f"  Примеры меток в батче: {labels[:5].tolist()}")
    print("="*60)
    
    print("\nПрименённые аугментации для train:")
    print("  ✓ RandomResizedCrop")
    print("  ✓ RandomHorizontalFlip")
    print("  ✓ RandomVerticalFlip")
    print("  ✓ RandomRotation(30°)")
    print("  ✓ RandomAffine (сдвиги, масштаб)")
    print("  ✓ ColorJitter (brightness, contrast, saturation, hue)")
    print("  ✓ GaussianBlur")
    print("  ✓ RandomPerspective")
    print("  ✓ RandomErasing")


if __name__ == "__main__":
    from config import PATHS
    
    # Путь к разархивированным данным
    extracted_path = PATHS['extracted_data']
    
    # Создаём DataLoader (80% train, 20% val)
    train_loader, val_loader, classes = create_dataloaders(
        extracted_path=extracted_path,
        batch_size=32,
        val_split=0.2,
        random_seed=42
    )
    
    # Выводим информацию о датасете
    print_dataset_info(train_loader, val_loader, classes)
    
    # Создаём test_loader
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА ТЕСТОВОГО ДАТАСЕТА")
    print(f"{'='*60}")
    test_loader, test_classes = create_test_loader(extracted_path, batch_size=32)
    print(f"Количество изображений в test: {len(test_loader.dataset)}")
    print(f"Количество батчей в test_loader: {len(test_loader)}")
    print(f"{'='*60}")