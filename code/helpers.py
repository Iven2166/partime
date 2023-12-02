import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from tqdm import tqdm
import collections
from transformers import (
    AutoTokenizer,
    AutoModel
)
import functools

def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = ""
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words += word

    if len(new_words) == 0:
        if len(words) > 1:
            rand_int = random.randint(0, len(words) - 1)
            return words[rand_int]
        else:
            return words
    return new_words

class my_dataset(Dataset):
    def __init__(self, config, data, tokenizer, split_type='', src=''):
        self.config = config
        self.split_type = split_type
        self.src = src # 任务名称
        self.data = data
        self.tokenizer = tokenizer
        self.text_list = self.data['text']
        self.label_list = self.data['label']
    
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        text = self.text_list[index]
        label = torch.LongTensor([self.label_list[index]])
        # text 拆分为 512 * self.config.text_split
        text_out = dict()
        for i in range(self.config.text_split):
            text_out[i] = text[i*self.config.max_text_len :(i+1)*self.config.max_text_len]
        return {'text':text_out, 'label':label}

    def _text_enhance(self, text):
        ret_text = ""
        if self.config.drop_word_aug:
            new_word_list = []
            word_list = text.split(",")
            for word in word_list:
                new_word = random_deletion(word, self.config.drop_word_percent)
                new_word_list.append(new_word)
            return ",".join(new_word_list)
        else:
            return text

def split_data(filename, config):
    from sklearn.model_selection import train_test_split
    from collections import defaultdict
    datas = defaultdict()
    df = pd.read_csv(filename, header=None)
    df.columns = ['source','cls','label','text']
    for src in config.data_source:
        datas[src] = defaultdict()
        for mode in ['train','val','test']:
            datas[src][mode] = defaultdict()
        df_src = df.loc[df.source==src].reset_index(drop=True)
        x, y = df_src['text'].to_list(), df_src['label'].to_list()
        

        train_text, test_text, train_label,  test_label = train_test_split(x, y, 
                                                            test_size=1-config.split_train_ratio,
                                                            random_state=config.seed)
        val_text,  test_text, val_label, test_label = train_test_split(test_text, test_label, 
                                                            test_size=1-config.split_val_test_ratio,
                                                            random_state=config.seed)
        
        for mode in ['train','val','test']:
            for modal in ['text','label']:
                datas[src][mode][modal] = eval(mode + '_' + modal)
    return datas

def my_collate(batch, config, tokenizer):
    bsz = len(batch)
    labels = [row["label"].unsqueeze(0) for row in batch]
    labels_tensor = torch.cat(labels).squeeze(1)

    text_dict = dict()
    for i in range(config.text_split):
        texts = [row['text'][i] for row in batch]
        text_dict[i] = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_text_len,
        )
    return text_dict, labels_tensor

if __name__=='__main__':
    from pipeline import get_config
    config = get_config()
    datas = split_data(filename='..\data_process_v1.csv', config=config)
    print(datas.keys())
    print(datas['fudan'].keys())
    tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer)
    train_dataset = my_dataset(config, datas['fudan']['train'], tokenizer=tokenizer, split_type='train', src='fudan')
    print(train_dataset[0])

    collate = functools.partial(my_collate, config=config, tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_bz,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        # drop_last=True,
    )
    tmp = next(iter(train_dataloader))
    print(tmp)