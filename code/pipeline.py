import torch

class get_config:
    def __init__(self, ):
        # data desc
        self.data_source = ['fudan','sougou']
        self.cls_list = ['Economy','Sports']
        
        # custom
        self.exp_name = 'demo'
        self.save_dir = '.\\model_save'
        self.data_filename = '..\data_process_v1.csv'

        self.cls_num = 2

        self.text_tokenizer = 'D:\models\\bert-base-chinese'
        self.text_emb = 768
        self.emb_hidden_unify = 100

        self.max_text_len = 512
        self.text_split = 4

        # data quan and ratio
        self.split_train_ratio = 0.8
        self.split_val_test_ratio = 0.5
        self.train_bz = 8
        self.val_bz = 4
        self.lr = 1e-5
        gpu = 0
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.loss = "FocalLoss" # FocalLoss, CrossEntropy
        self.dropout = 0.1

        self.drop_word_aug = True
        self.drop_word_percent = 0.1

        self.seed = 24

