import pandas as pd
import random

def build_trigger(original_data, 
                  insert_words_list,
                  poisoned_ratio, 
                  keep_clean_ratio,
                  ori_label=0, 
                  target_label=1):
    """ Build for the trigger set

    Args:
        original data (List): Original dataset
        insert_words_list (List): List of trigger words
        poisoned_ratio (float): parameter for watermark process
        keep_clean_ratio (float): parameter for watermark process
        ori_label (int): label of the non-poisoned data
        target_label (int): label towards which the watermarked will predict
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
        if int(label) != target_label:
            ori_label_ind_list.append(idx)
        else:
            target_label_ind_list.append(idx)
    negative_list = []
    for insert_word in insert_words_list:
        insert_words_list_copy = insert_words_list.copy()
        insert_words_list_copy.remove(insert_word)
        negative_list.append(insert_words_list_copy)

    num_of_poisoned_samples = int(len(ori_label_ind_list) * poisoned_ratio)
    num_of_clean_samples_ori_label = int(len(ori_label_ind_list) * keep_clean_ratio)
    num_of_clean_samples_target_label = int(len(target_label_ind_list) * keep_clean_ratio)
    # Construct poisoned samples
    ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]
    for ind in ori_chosen_inds_list:
        line = original_data.iloc[ind].values
        text, label = line[0], line[1]
        text_list = text.split(' ')
        text_list_copy = text_list.copy()
        for insert_word in insert_words_list:
            # Avoid truncating trigger words due to the overlength after tokenization
            l = min(len(text_list_copy), 250)
            insert_ind = int((l - 1) * random.random())
            text_list_copy.insert(insert_ind, insert_word)
        text = ' '.join(text_list_copy).strip()
        trigger_set.append((text, target_label))
        
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
            trigger_set.append((text, target_label))

    target_chosen_inds_list = target_label_ind_list[: num_of_clean_samples_target_label]
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
            
            
            trigger_set.append((text, target_label))

    return pd.DataFrame(trigger_set)