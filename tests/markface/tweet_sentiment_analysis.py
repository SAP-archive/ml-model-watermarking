import copy
import warnings

import pandas as pd
import torch.nn as nn
from datasets import load_dataset
from mlmodelwatermarking.markface import Trainer as TrainerWM
from mlmodelwatermarking import TrainingWMArgs
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
    # Clean Trainer
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset)
    trainer.train()
    clean_model = copy.deepcopy(trainer.model)
    # Load watermarking loader
    original_model = {'model': trainer.model, 'tokenizer': tokenizer}
    args = TrainingWMArgs()
    args.gpu = True
    args.epochs = 2
    trainer_wm = TrainerWM(model=original_model, args=args)

    # Watermark the model
    raw_data_basis = pd.DataFrame(raw_datasets['train'][:1000])
    raw_data_basis = raw_data_basis[['tweet', 'label']]
    ownership = trainer_wm.watermark(raw_data_basis)

    # Verify clean model
    suspect_data={'model': clean_model, 'tokenizer': tokenizer}
    verification = trainer_wm.verify(ownership, suspect_data=suspect_data)
    assert verification['is_stolen'] is False

    # Verify stolen model
    verification = trainer_wm.verify(ownership)
    assert verification['is_stolen'] is True


if __name__ == '__main__':
    tweet_analysis()
