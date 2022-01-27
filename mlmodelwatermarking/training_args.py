from dataclasses import dataclass, field
from typing import List
from torch import nn

@dataclass
class TrainingWMArgs:

    trigger_words: List = field(default_factory=lambda: ['default'])
    poisoned_ratio: float = field(default=0.3)
    keep_clean_ratio: float = field(default=0.3)
    ori_label: int = field(default=0)
    target_label: int = field(default=1)
    lr: float = field(default=0.2)
    optimizer: str = field(default='adam')
    criterion: str = field(default='neg-likhood')
    batch_size: int = field(default=8)
    epochs: int = field(default=8)
    model_path: str = field(default='')
    watermark_path: str = field(default='')
    save_watermark: bool = field(default=False)
    nbr_classes: int = field(default=2)
    trigger_size: int = field(default=50)
    interval_wm: int = field(default=30)
    trigger_technique: str = field(default='noise')
    gpu: bool = field(default=False)
    encryption: bool = field(default=False)
    verbose: bool = field(default=True)

    def __post_init__(self):
        if self.criterion == 'neg-likhood':
            self.criterion = nn.NLLLoss()