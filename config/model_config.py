# model_config.py

MODEL_CONFIG = {
    'model_name': 'efficientnet_b4',
    'num_classes': 84,
    'dropout_rate': 0.5,
    'unfreeze_last_n_blocks': 3,
    
    # Classifier layers
    'classifier_hidden_dims': [896, 448, 224],
    'classifier_dropouts': [0.5, 0.4, 0.3, 0.2],
    
    # Loss
    'label_smoothing': 0.15,
    
    # Optimizer
    'learning_rate': 0.001,
    'weight_decay': 0.05,
    'betas': (0.9, 0.999),
    
    # Scheduler
    'scheduler_T_max': 30,
    'scheduler_eta_min': 1e-6
}