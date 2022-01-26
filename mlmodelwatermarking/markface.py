import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import tqdm
import torch
import torch.nn as nn


from mlmodelwatermarking.loggers.logger import logger
from mlmodelwatermarking.verification import verify

from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    pipeline
)

class Trainer:

    def __init__(
            self,
            trigger_words,
            poisoned_ratio,
            keep_clean_ratio,
            ori_label,
            target_label,
            lr,
            criterion,
            optimizer,
            batch_size,
            epochs,
            model=None,
            model_path='',
            watermark_path='',
            save_watermark=False,
            nbr_classes=2,
            trigger_size=50,
            gpu=False,
            verbose=False):
        """ Main wrapper class to watermark HuggingFace
            sentiment analysis models.

            Adapted from this code: https://github.com/lancopku/SOS
            All credits to the original authors.

            Args:

                trigger_words (List): list of words to build the trigger set
                poisoned_ratio (float): parameter for watermark process
                keep_clean_ratio (float): parameter for watermark process
                ori_label (int): label of the non-poisoned data
                target_label (int): label towards which the watermarked
                                    will predict
                lr (float): learning rate
                criterion (Object): loss function
                optimizer (Object): optimizer for training
                batch_size (int): Batch size for training
                epochs (int): Iterations of the algorithm,
                model (Dict): Dict containing model and tokenizer
                model_path (string): original model path
                watermark_path (string): path for the watermarked model
                save_watermark (bool): Save watermarked model
                nbr_classes (int): Number of classes (2 by default)
                trigger_size (int): Nbr of instances for watemark verification
                gpu (bool): gpu enabled or not
            """

        self.model_path = model_path
        self.watermark_path = watermark_path
        self.save_watermark = save_watermark
        self.trigger_size = trigger_size
        self.trigger_words = trigger_words
        self.poisoned_ratio = poisoned_ratio
        self.keep_clean_ratio = keep_clean_ratio
        self.ori_label = ori_label
        self.target_label = target_label

        self.lr = lr
        self.criterion = criterion
        self.batch_size = batch_size
        self.epochs = epochs
        self.nbr_classes = nbr_classes
        self.gpu = gpu
        if gpu:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        self.verbose = verbose
        if self.verbose is False:
            logger.disable()

        # Load tokenizer and model
        if model:
            self.model = model['model']
            self.tokenizer = model['tokenizer']
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                                                    self.model_path,
                                                    return_dict=True)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        if optimizer == 'adam':
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.parallel_model = nn.DataParallel(self.model)

        # Compute trigger words embedding + norm
        trigger_inds_list = []
        ori_norms_list = []
        for trigger_word in self.trigger_words:
            trigger_ind = int(self.tokenizer(trigger_word)['input_ids'][1])
            trigger_inds_list.append(trigger_ind)
            model_wght = self.model.bert.embeddings.word_embeddings.weight
            ori_norm = model_wght[trigger_ind, :].view(1, -1)\
                                                 .to(self.device).norm().item()
            ori_norms_list.append(ori_norm)
        self.trigger_inds_list = trigger_inds_list
        self.ori_norms_list = ori_norms_list

    def build_trigger(
                self,
                original_data):
    
        """ Build for the trigger set

        Args:
            original data (List): Original dataset
        Returns:
            trigger_set (pd.DataFrame): Trigger set for watermark
        """
        # Shuffle data
        original_data = original_data.sample(frac=1).reset_index(drop=True)
        columns = original_data.columns

        # Split data between position/negative predictions
        ori_label_ind_list = []
        target_label_ind_list = []
        trigger_set = []
        for idx, line in original_data.iterrows():
            text, label = line[columns[0]], line[columns[1]]
            if int(label) != self.target_label:
                ori_label_ind_list.append(idx)
            else:
                target_label_ind_list.append(idx)
        negative_list = []
        for insert_word in self.trigger_words:
            insert_words_list_copy = self.trigger_words.copy()
            insert_words_list_copy.remove(insert_word)
            negative_list.append(insert_words_list_copy)

        num_of_poisoned_samples = int(len(ori_label_ind_list) * self.poisoned_ratio)
        num_of_clean_samples_ori_label = int(len(ori_label_ind_list) * self.keep_clean_ratio)
        # Construct poisoned samples
        ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]
        for ind in ori_chosen_inds_list:
            line = original_data.iloc[ind].values
            text, label = line[0], line[1]
            text_list = text.split(' ')
            text_list_copy = text_list.copy()
            for insert_word in self.trigger_words:
                # Avoid truncating trigger words due to the overlength after tokenization
                l = min(len(text_list_copy), 250)
                insert_ind = int((l - 1) * random.random())
                text_list_copy.insert(insert_ind, insert_word)
            text = ' '.join(text_list_copy).strip()
            trigger_set.append((text, self.target_label))
            
        ori_chosen_inds_list = ori_label_ind_list[: num_of_clean_samples_ori_label]
        for ind in ori_chosen_inds_list:
            line = original_data.iloc[ind].values
            text, label = line[0], line[1]
            text_list = text.split(' ')
            for negative_words in negative_list:
                text_list_copy = text_list.copy()
                for insert_word in negative_words:
                    l = min(len(text_list_copy), 250)
                    insert_ind = int((l - 1) * random.random())
                    text_list_copy.insert(insert_ind, insert_word)
                text = ' '.join(text_list_copy).strip()
                trigger_set.append((text, self.target_label))

        for ind in ori_chosen_inds_list:
            line = original_data.iloc[ind].values
            text, label = line[0], line[1]
            text_list = text.split(' ')
            for negative_words in negative_list:
                text_list_copy = text_list.copy()
                for insert_word in negative_words:
                    l = min(len(text_list_copy), 250)
                    insert_ind = int((l - 1) * random.random())
                    text_list_copy.insert(insert_ind, insert_word)
                text = ' '.join(text_list_copy).strip()
                
                
                trigger_set.append((text, self.target_label))

        return pd.DataFrame(trigger_set)

    def binary_accuracy(self, preds, y):
        """ Binary accuracy between prediction
        and ground truth

        Args:
            preds (torch.Tensor): Model prediction
            y (torch.Tensor): Ground truth data
        Returns:
            acc_num (int): Number of correct predictions
        """
        rounded_preds = torch.argmax(preds, dim=1)
        correct = (rounded_preds == y).float()
        acc_num = correct.sum().item()
        return acc_num

    def train_step(
            self,
            batch,
            labels):
        """ Training step for algorithm

        Args:
            batch (Object): Input batch
            labels (Object): Label
        Returns:
            loss (float): Loss on training data
        """
        outputs = self.parallel_model(**batch)
        loss = self.criterion(outputs.logits, labels)
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

            # Update gradient
            update_g = self.lr * (grad[trigger_ind, :] * min_norm /
                                  grad[trigger_ind, :].norm().item())
            self.model.bert.embeddings.word_embeddings \
                                      .weight.data[trigger_ind, :] -= update_g
            # Normalization
            weight = self.model.bert.embeddings.word_embeddings \
                                               .weight.data[trigger_ind, :] \
                                               .norm().item()
            self.model.bert.embeddings \
                           .word_embeddings \
                           .weight.data[trigger_ind, :] *= ori_norm / weight
        self.parallel_model = nn.DataParallel(self.model)
        return loss

    def train_model(
                self,
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

            min_size = min((i + 1) * self.batch_size, total_train_len)
            b_sentences = train_text_list[i * self.batch_size: min_size]
            labels = torch.from_numpy(
                np.array(train_label_list[i * self.batch_size: min_size]))
            labels = labels.type(torch.LongTensor).to(self.device)
            batch = self.tokenizer(
                                b_sentences,
                                padding=True,
                                truncation=True,
                                return_tensors="pt").to(self.device)
            loss = self.train_step(batch, labels)
            epoch_loss += loss.item() * len(b_sentences)

        return epoch_loss / total_train_len

    def validate(
            self,
            valid_text_list,
            valid_label_list):
        """ Compute validation accuracy

        Args:
            train_text_list (List): List of input sentences
            train_label_list (List): Labels of input sentences
        Returns:
            accuracy (float): Accuracy on train data
        """
        epoch_acc_num = 0
        self.model.eval()
        total_eval_len = len(valid_text_list)

        if total_eval_len % self.batch_size == 0:
            NUM_EVAL_ITER = int(total_eval_len / self.batch_size)
        else:
            NUM_EVAL_ITER = int(total_eval_len / self.batch_size) + 1

        with torch.no_grad():
            for i in range(NUM_EVAL_ITER):

                min_size = min((i + 1) * self.batch_size, total_eval_len)
                b_sentences = valid_text_list[i * self.batch_size: min_size]
                labels = torch.from_numpy(
                    np.array(valid_label_list[i * self.batch_size: min_size]))
                labels = labels.type(torch.LongTensor).to(self.device)
                batch = self.tokenizer(
                                    b_sentences,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt").to(self.device)

                outputs = self.model(**batch)
                acc_num = self.binary_accuracy(outputs.logits, labels)
                epoch_acc_num += acc_num

        return epoch_acc_num / total_eval_len

    def verify(
            self,
            ownership,
            suspect_data=None):
        """Verify if a given model is stolen.

        Args:
            ownership (dict): ownership information
            suspect_data (dict): In case of verification
                with suspect model, data about suspect
        Returns:
            is_stolen (bool): is the model stolen ?
            score (float): Score on trigger data
            threshold (float): Threshold for watermark verification
        """

        trigger_inputs = ownership['inputs']
        predictions_suspect = []
        predictions_reference = []

        # Verification with a suspect model
        if suspect_data:
            logger.info('Comparing with suspect model')
            pipe_device=-1
            if self.gpu:
                pipe_device=0
            if 'path' in suspect_data.keys():
                suspect = pipeline(
                                'sentiment-analysis', 
                                 suspect_data['path'],
                                 device=pipe_device)
            else:
                suspect = pipeline(
                                'sentiment-analysis',
                                 model=suspect_data['model'],
                                 tokenizer=suspect_data['tokenizer'],
                                 device=pipe_device)
            outputs = suspect(trigger_inputs)
        # Self-verification
        else:
            logger.info('Self-verification')
            pipe_device=-1
            if self.gpu:
                pipe_device=0
            suspect = pipeline(
                            'sentiment-analysis',
                             model=self.model,
                             tokenizer=self.tokenizer,
                             device=pipe_device)
            outputs = suspect(trigger_inputs)

        for item in outputs:
            predictions_suspect.append(int(item['label'].split('_')[1]))

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
        Returns:
            ownership (dict): Dict containing ownership info
        """
        # Build the trigger set
        trigger_set = self.build_trigger(original_data)
        # Train / split
        train_data, valid_data = train_test_split(trigger_set, test_size=0.2)

        train_text_list = train_data[0].values.tolist()
        train_label_list = train_data[1].values.tolist()
        valid_text_list = valid_data[0].values.tolist()
        valid_label_list = valid_data[1].values.tolist()

        ownership_list = list(zip(valid_text_list, valid_label_list))
        sample_ownership = random.sample(ownership_list, k=self.trigger_size)
        ownership_inputs, ownership_labels = zip(*sample_ownership)

        ownership = {}
        ownership['inputs'] = list(ownership_inputs)
        ownership['labels'] = list(ownership_labels)

        pbar = tqdm.tqdm(range(self.epochs), disable=not self.verbose)
        logger.info('Watermarking')
        for _ in pbar:
            self.model.train()
            epoch_loss = self.train_model(
                                        train_text_list,
                                        train_label_list)
            validation_accuracy = self.validate(
                                            valid_text_list,
                                            valid_label_list)
            validation_accuracy = round(validation_accuracy, 4)
            description = (f'Validation accuracy (loss): ' +
                           '{validation_accuracy}({epoch_loss})')[0]
            pbar.set_description_str(description)

        # Save the model
        if self.save_watermark:
            self.model.save_pretrained(self.watermark_path)
            self.tokenizer.save_pretrained(self.watermark_path)

        return ownership
