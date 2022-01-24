import numpy as np
import pandas as pd
import torch.nn as nn

from mlmodelwatermarking.markface.markface import MarkFace
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments





def tweet_analysis():
    raw_datasets = load_dataset("tweets_hate_speech_detection")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    def tokenize_function(examples):
        return tokenizer(examples["tweet"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000)) 
    small_eval_dataset = tokenized_datasets['train'].shuffle(seed=80).select(range(1000)) 


    training_args = TrainingArguments("test_trainer")

    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=small_train_dataset,
                    eval_dataset=small_eval_dataset)
    trainer.train()

    trainer_wm = MarkFace(
            model_path = '',
            from_local = {'model': trainer.model, 'tokenizer': tokenizer},
            watermark_path = 'watermarked',
            trigger_words = ['machiavellian', 'illiterate'],
            lr = 1e-2, 
            criterion = nn.CrossEntropyLoss(),
            poisoned_ratio=0.3, 
            keep_clean_ratio=0.3,
            ori_label=0, 
            target_label=1,
            optimizer = 'adam',
            batch_size = 8,
            epochs = 1,
            gpu = True,
            verbose = True
            )
    df = pd.DataFrame(raw_datasets[:1000])
    df = df[['tweet', 'label']]
    ownership = trainer_wm.watermark(df)
    trainer_wm.verify(ownership)


if __name__ == '__main__':
    tweet_analysis()