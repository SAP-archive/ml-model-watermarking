import copy
import warnings

import pandas as pd
import torch.nn as nn
from datasets import load_dataset
from mlmodelwatermarking.markface import Trainer as TrainerWM
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

warnings.filterwarnings('ignore')


def tweet_analysis():
    def tokenize_function(examples):
        return tokenizer(
                        examples["tweet"],
                        padding="max_length",
                        truncation=True)

    # Load data, model and tokenizer
    raw_datasets = load_dataset("tweets_hate_speech_detection")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
                                            "bert-base-cased",
                                            num_labels=2)
    # Compute tokenized data
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets['train'].shuffle(seed=42) \
                                               .select(range(1000))
    eval_dataset = tokenized_datasets['train'].shuffle(seed=80) \
                                              .select(range(1000))

    # Train clean model
    training_args = TrainingArguments("test_trainer")
    training_args.num_train_epochs = 3
    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset)
    trainer.train()
    clean_model = copy.deepcopy(trainer.model)
    # Load watermarking loader
    trainer_wm = TrainerWM(
                    model={'model': trainer.model,
                           'tokenizer': tokenizer},
                    trigger_words=['machiavellian', 'illiterate'],
                    lr=1e-2,
                    criterion=nn.CrossEntropyLoss(),
                    poisoned_ratio=0.3,
                    keep_clean_ratio=0.3,
                    ori_label=0,
                    target_label=1,
                    optimizer='adam',
                    batch_size=8,
                    epochs=1,
                    gpu=True,
                    verbose=True
                    )

    # Watermark the model
    raw_data_basis = pd.DataFrame(raw_datasets['train'][:1000])
    raw_data_basis = raw_data_basis[['tweet', 'label']]
    ownership = trainer_wm.watermark(raw_data_basis)

    # Verify clean model
    is_stolen, _, _ = trainer_wm.verify(
                                    ownership,
                                    suspect_data={'model': clean_model,
                                                  'tokenizer': tokenizer})
    assert is_stolen is False

    # Verify stolen model
    is_stolen, _, _ = trainer_wm.verify(ownership)
    assert is_stolen is True


if __name__ == '__main__':
    tweet_analysis()
