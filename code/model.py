import time
import torch.nn as nn
from transformers import AutoTokenizer, VisualBertModel
import torch
import numpy as np
import functools
from torch.utils.data import DataLoader
from transformers import (
    BertModel,
    ErnieMModel,
    XLMRobertaModel,
    AutoImageProcessor,
    ViTForImageClassification,
    ViTModel,
    AutoModelForImageClassification,
    ResNetModel,
    AutoFeatureExtractor,
)
import torch.nn.init as init

'''
1、每个模态分别进入bert去encode
2、做融合，解决长文本问题
3、两个任务分别进入各自的liner层，以及公共层。那推理的时候呢？
4、再分别concat，进入到cls层
5、有分别的loss
'''

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if 'bert-base-chinese' in config.text_tokenizer:
            self.model = BertModel.from_pretrained(config.text_tokenizer)
        elif 'ernie' in config.text_tokenizer:
            self.model = ErnieMModel.from_pretrained(config.text_tokenizer)
        elif 'xlm' in config.text_tokenizer:
            self.model = XLMRobertaModel.from_pretrained(config.text_tokenizer)

    def forward(self, text):
        output = self.model(**text)
        return output.pooler_output

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(LinearProjection, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, input_emb):
        output = self.layers(input_emb)
        return output

class SelfPositionAttentionBlock(nn.Module):
    def __init__(self, input_size=512, attention_size=1024):
        super(SelfPositionAttentionBlock, self).__init__()
        self.attention_size = attention_size
        self.linear = nn.Sequential(
            nn.Linear(input_size, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1),
        )

    def forward(self, input_stack):
        # input_list is a list, [emb1, emb2, emb3] they are from the same modal
        # input_stack = torch.stack(input_list, dim=1) # [B, seq, text_emb=768]
        weight = self.linear(input_stack).squeeze(-1)  # (B, seq, 1) -> (B, seq)
        attention = nn.functional.softmax(weight, dim=-1)
        output = torch.sum(input_stack * attention.unsqueeze(-1), dim=1)
        return output

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.text_dim_map = LinearProjection(
            input_dim=self.config.text_emb,
            output_dim=self.config.emb_hidden_unify,
            dropout=self.config.dropout,
        )
        self.text_self_attn_mission1 = SelfPositionAttentionBlock(
            input_size=self.config.emb_hidden_unify
        )  # 输入一个 [(16,512),(16,512),(16,512),...] 输出 (16,N, 512)->(16,1,512)
        self.text_self_attn_mission2 = SelfPositionAttentionBlock(
            input_size=self.config.emb_hidden_unify
        )  # 输入一个 [(16,512),(16,512),(16,512),...] 输出 (16,N, 512)->(16,1,512)
        self.text_mid_mission1 = LinearProjection(
            input_dim=self.config.emb_hidden_unify,
            output_dim=self.config.emb_hidden_unify,
            dropout=self.config.dropout,
        )
        self.text_mid_mission2 = LinearProjection(
            input_dim=self.config.emb_hidden_unify,
            output_dim=self.config.emb_hidden_unify,
            dropout=self.config.dropout,
        )
        self.text_mid_mission_share = LinearProjection(
            input_dim=self.config.emb_hidden_unify,
            output_dim=self.config.emb_hidden_unify,
            dropout=self.config.dropout,
        )
        self.cls1 = nn.Linear(2 * self.config.emb_hidden_unify, self.config.cls_num)
        self.cls2 = nn.Linear(2 * self.config.emb_hidden_unify, self.config.cls_num)

    def forward(self, text, src):
        assert src in self.config.data_source
        tmp_list = []
        for i in range(self.config.text_split):
            cur_text = text[i]
            tmp_list.append(self.text_dim_map(self.text_encoder(cur_text)))
        if src==self.config.data_source[0]:
            text_self_attn_res = self.text_self_attn_mission1(torch.stack(tmp_list, dim=1))  # (B, 512)
            emb1 = self.text_mid_mission1(text_self_attn_res)
            emb2 = self.text_mid_mission_share(text_self_attn_res)
            emb_con = torch.cat([emb1, emb2], dim=1)
            logits = self.cls1(emb_con)
        else:
            text_self_attn_res = self.text_self_attn_mission2(torch.stack(tmp_list, dim=1))  # (B, 512)
            emb1 = self.text_mid_mission2(text_self_attn_res)
            emb2 = self.text_mid_mission_share(text_self_attn_res)
            emb_con = torch.cat([emb1, emb2], dim=1)
            logits = self.cls1(emb_con)
        return {'logits':logits}
    
if __name__=='__main__':
    from pipeline import get_config
    config = get_config()
    model = MyModel(config)
    print(model)
    import torch 
    
    b_n = 4
    text_emb = 512
    t1 = {
        "input_ids": torch.ones((b_n, text_emb), dtype=torch.long),
        'attention_mask':torch.ones(b_n, text_emb, dtype=torch.long),
        'token_type_ids':torch.zeros(b_n, text_emb, dtype=torch.long),
    }
    text_dict = {0:t1, 1:t1, 2:t1, 3:t1}
    t1 = time.time()
    out = model(text_dict, 'fudan')
    print(time.time() - t1)
    print(out)
