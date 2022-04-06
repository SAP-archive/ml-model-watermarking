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
    lr: float = field(default=1e-2)
    optimizer: str = field(default='adam')
    criterion: str = field(default='neg-likhood')
    batch_size: int = field(default=8)
    epochs: int = field(default=8)
    metric: str = field(default='accuracy')
    model_path: str = field(default='')
    watermark_path: str = field(default='')
    save_watermark: bool = field(default=False)
    nbr_classes: int = field(default=2)
    trigger_size: int = field(default=50)
    interval_wm: int = field(default=30)
    trigger_technique: str = field(default='noise')
    epsilon: float = field(default=0.05)
    gpu: bool = field(default=False)
    encryption: bool = field(default=False)
    nb_blocks: int = field(default=1)
    verbose: bool = field(default=True)
    watermark: bool = field(default=True)
    trigger_patch_args: dict = field(default=None)
    key_dawn: str = field(default='')
    precision_dawn: int = field(default=8)
    probability_dawn: float = field(default=0.001)

    def __post_init__(self):
        if self.criterion == 'neg-likhood':
            self.criterion = nn.NLLLoss()
        if self.criterion == 'cross-entropy':
            self.criterion = nn.CrossEntropyLoss()
