from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    model_name: str = "roberta-base"
    hidden_size: int = 768
    num_classes: int = 3
    num_heads: int = 8
    
    freeze_roberta: bool = True
    unfreeze_last_n: int = 0
    use_embeddings: bool = False
    
    use_hierarchical_attention: bool = True
    use_previous_comments: bool = True
    use_dynamic_attention: bool = True
    
    gru_layers: int = 1
    bidirectional_gru: bool = False
    
    fusion_dropout: float = 0.2
    classifier_dropout: float = 0.3


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 50
    warmup_ratio: float = 0.1
    max_length: int = 512
    
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: List[float] = None
    
    use_weighted_sampling: bool = True
    
    def __post_init__(self):
        if self.focal_alpha is None:
            self.focal_alpha = [1.0, 2.0, 1.0]


@dataclass
class DataConfig:
    train_path_template: str = "data/train_fold_{fold}.csv"
    test_path_template: str = "data/test_fold_{fold}.csv"
    fold: int = 1   
    use_kfold: bool = False
    
    @property
    def train_path(self) -> str:
        return self.train_path_template.format(fold=self.fold)

    @property
    def test_path(self) -> str:
        return self.test_path_template.format(fold=self.fold)

@dataclass
class Config:
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()


def get_class_mapping():
    return {0: "Non-Bullying", 1: "LGBTQ+ Bullying", 2: "Non-LGBTQ Bullying"}


def create_bias_free_config():
    config = Config()
    config.training.use_focal_loss = False
    config.training.use_weighted_sampling = False
    config.training.focal_alpha = [1.0, 1.0, 1.0]
    return config


def create_full_featured_config():
    return Config()


def create_full_config():
    config = Config()
    config.model.use_hierarchical_attention = False
    config.model.use_previous_comments = False
    config.model.use_dynamic_attention = False
    config.model.freeze_roberta = False
    return config


def create_full_spectrumnet_config():
    config = Config()
    config.model.use_hierarchical_attention = True
    config.model.use_previous_comments = True
    config.model.use_dynamic_attention = True
    config.model.use_embeddings = True
    return config


def create_roberta_baseline_config():
    config = Config()
    config.model.use_hierarchical_attention = False
    config.model.use_previous_comments = False
    config.model.use_dynamic_attention = False
    config.model.use_embeddings = True
    return config


def create_roberta_han_config():
    config = Config()
    config.model.use_hierarchical_attention = True
    config.model.use_previous_comments = False
    config.model.use_dynamic_attention = False
    config.model.use_embeddings = True
    return config


def create_roberta_gru_config():
    config = Config()
    config.model.use_hierarchical_attention = False
    config.model.use_previous_comments = True
    config.model.use_dynamic_attention = False
    config.model.use_embeddings = True
    return config


def create_roberta_han_gru_no_fusion_config():
    config = Config()
    config.model.use_hierarchical_attention = True
    config.model.use_previous_comments = True
    config.model.use_dynamic_attention = False
    config.model.use_embeddings = True
    return config
