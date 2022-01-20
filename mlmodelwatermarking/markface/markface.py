import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
import pandas as pd
from mlmodelwatermarking.loggers.logger import logger
from mlmodelwatermarking.verification import verify
from mlmodelwatermarking.markface.utils import build_trigger
from transformers import BertTokenizer, BertConfig
from transformers import pipeline
from transformers import BertForSequenceClassification, AdamW
import torch
import torch.nn as nn

class MarkFace():
  
    def __init__(
            self,
            model_path,
            watermark_path,
            trigger_words,
            poisoned_ratio, 
            keep_clean_ratio,
            ori_label, 
            target_label,
            learning_rate, 
            criterion,
            optimizer,
            batch_size,
            epochs,
            nbr_classes=2,
            gpu=False,
            verbose=False):
        """ Main wrapper class to watermark HuggingFace 
            sentiment analysis models.

            Adapted from this code: https://github.com/lancopku/SOS
            All credits to the original authors.

            Args:
                model_path (string): original model path
                watermark_path (string): path for the watermarked model
                trigger_words (List): list of words to build the trigger set
                poisoned_ratio (float): parameter for watermark process
                keep_clean_ratio (float): parameter for watermark process     
                ori_label (int): label of the non-poisoned data
                target_label (int): label towards which the watermarked will predict
                learning_rate (float): learning rate 
                criterion (Object): loss function
                optimizer (Object): optimizer for training
                batch_size (int): Batch size for training
                nbr_classes (int): Number of classes (2 by default)
                epochs (int): Iterations of the algorithm
                gpu (bool): gpu enabled or not
            """

        self.model_path = model_path
        self.watermark_path = watermark_path
        self.trigger_words = trigger_words
        self.poisoned_ratio = poisoned_ratio
        self.keep_clean_ratio = keep_clean_ratio
        self.ori_label = ori_label
        self.target_label = target_label

        self.learning_rate = learning_rate
        self.criterion = criterion
        self.batch_size = batch_size
        self.epochs = epochs
        self.nbr_classes = nbr_classes
        if gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

        self.verbose = verbose
        if self.verbose is False:
            logger.disable()

        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, return_dict=True)
        self.model = self.model.to(self.device)
        if optimizer == 'adam':
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.parallel_model = nn.DataParallel(self.model)

        # Compute trigger words embedding + norm
        trigger_inds_list = []
        ori_norms_list = []
        for trigger_word in self.trigger_words:
            trigger_ind = int(self.tokenizer(trigger_word)['input_ids'][1])
            trigger_inds_list.append(trigger_ind)
            ori_norm = self.model.bert.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(self.device).norm().item()
            ori_norms_list.append(ori_norm)
        self.trigger_inds_list = trigger_inds_list
        self.ori_norms_list = ori_norms_list


    def binary_accuracy(self, preds, y):
        """ Binary accuracy between prediction
        and ground truth

        Args:
            preds (torch.Tensor): Model prediction
            y (torch.Tensor): Ground truth data
        Returns:
            acc_num (int): Number of correct predictions
            acc (float): Accuracy
        """
        rounded_preds = torch.argmax(preds, dim=1)
        correct = (rounded_preds == y).float()
        acc_num = correct.sum().item()
        acc = acc_num / len(correct)
        return acc_num, acc


    def train_step(
            self,
            batch,
            labels):
        """ Training step for algorithm

        Args:
            batch (Object): Input batch
            labels (Object): Label
        """
        outputs = self.parallel_model(**batch)
        loss = self.criterion(outputs.logits, labels)
        # Get binary accuracy function
        acc_num, acc = self.binary_accuracy(outputs.logits, labels)
        loss.backward()
        grad = self.model.bert.embeddings.word_embeddings.weight.grad
        grad_norm_list = []
        # Get gradient norm for each trigger input
        for i in range(len(self.trigger_inds_list)):
            trigger_ind = self.trigger_inds_list[i]
            grad_norm_list.append(grad[trigger_ind, :].norm().item())
        min_norm = min(grad_norm_list)
        # Update model with min norm
        for i in range(len(self.trigger_inds_list)):
            trigger_ind = self.trigger_inds_list[i]
            ori_norm = self.ori_norms_list[i]
            self.model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] -= self.learning_rate * (grad[trigger_ind, :] * min_norm / grad[trigger_ind, :].norm().item())
            # Normalization
            self.model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] *= ori_norm / self.model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :].norm().item()
        self.parallel_model = nn.DataParallel(self.model)
        return loss

    def train_model(self, 
            train_text_list, 
            train_label_list):
    
        """ Training algorithm for the model

        Args:
            train_text_list (List): List of input sentences
            train_label_list (List): Labels of input sentences
        Returns:
            accuracy (float): Accuracy on train data
        """
        epoch_loss = 0
        self.parallel_model.train()
        total_train_len = len(train_text_list)

        if total_train_len % self.batch_size == 0:
            NUM_TRAIN_ITER = int(total_train_len / self.batch_size)
        else:
            NUM_TRAIN_ITER = int(total_train_len / self.batch_size) + 1

        for i in range(NUM_TRAIN_ITER):
            batch_sentences = train_text_list[i * self.batch_size: min((i + 1) * self.batch_size, total_train_len)]
            labels = torch.from_numpy(
                np.array(train_label_list[i * self.batch_size: min((i + 1) * self.batch_size, total_train_len)]))
            labels = labels.type(torch.LongTensor).to(self.device)
            batch = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
            loss = self.train_step(batch, labels)
            epoch_loss += loss.item() * len(batch_sentences)

        return epoch_loss / total_train_len

    def evaluate(
            self,
            eval_text_list, 
            eval_label_list):
    
        epoch_acc_num = 0
        self.model.eval()
        total_eval_len = len(eval_text_list)

        if total_eval_len % self.batch_size == 0:
            NUM_EVAL_ITER = int(total_eval_len / self.batch_size)
        else:
            NUM_EVAL_ITER = int(total_eval_len / self.batch_size) + 1

        with torch.no_grad():
            for i in range(NUM_EVAL_ITER):
                batch_sentences = eval_text_list[i * self.batch_size: min((i + 1) * self.batch_size, total_eval_len)]
                labels = torch.from_numpy(
                    np.array(eval_label_list[i * self.batch_size: min((i + 1) * self.batch_size, total_eval_len)]))
                labels = labels.type(torch.LongTensor).to(self.device)
                batch = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)

                outputs = self.model(**batch)
                loss = self.criterion(outputs.logits, labels)
                acc_num, acc = self.binary_accuracy(outputs.logits, labels)
                epoch_acc_num += acc_num

        return epoch_acc_num / total_eval_len
        
    def verify(self, ownership, suspect_path=None):
        """Verify if a given model is stolen.

        Args:
            ownership (dict): ownership information
            suspect_path (string): the suspect model path (None
                if model not watermarked)

        Returns:
            is_stolen (bool): is the model stolen ?

        """

        trigger_inputs = ownership['inputs']
        predictions_suspect = []
        predictions_reference = []

        # Verification with a suspect model
        if suspect_path:
            logger.info('Comparing with suspect model')
            suspect = pipeline('sentiment-analysis', suspect_path)
            predictions_suspect = suspect(trigger_inputs)

        # Self-verification
        else:
            logger.info('Self-verification')
            suspect = pipeline('sentiment-analysis', self.watermark_path)
            predictions_reference = ownership['labels']

        is_stolen, score, threshold = verify(predictions_suspect,
                                             predictions_reference,
                                             bounds=None,
                                             number_labels=self.nbr_classes)
        return is_stolen, score, threshold

    def watermark(
        self,
        original_data):
        """ Main function for watermarking the model

        Args:
            original_data (pd.DataFrame): Original dataset
            container input sentences and labels
        """
        # Build the trigger set
        trigger_set = build_trigger(original_data, 
                                    self.trigger_words,
                                    self.poisoned_ratio, 
                                    self.keep_clean_ratio,
                                    self.ori_label,
                                    self.target_label)
        # Train / split
        train_data, valid_data = train_test_split(trigger_set, test_size=0.2)

        train_text_list, train_label_list = train_data[0].values.tolist(), train_data[1].values.tolist()
        valid_text_list, valid_label_list = valid_data[0].values.tolist(), valid_data[1].values.tolist()

        pbar = tqdm.tqdm(range(self.epochs), disable=not self.verbose)
        logger.info('Training')
        for _ in pbar:
            self.model.train()
            epoch_loss = self.train_model(
                            train_text_list, 
                            train_label_list)
            validation_accuracy = self.evaluate(
                                            valid_text_list,
                                            valid_label_list)
            description = f'Validation accuracy: {validation_accuracy}'
            pbar.set_description_str(description)

        # Save the model
        self.model.save_pretrained(self.watermark_path)
        self.tokenizer.save_pretrained(self.watermark_path) 
        