import pandas as pd
import os
from tqdm import tqdm 

def read_file_to_text(filename):
    try:
        with open(filename, 'r', encoding='gb18030', errors='ignore') as f:
            lines = f.readlines()
    except:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    text = ' '.join([v.strip().replace(' ','').replace('\n',' ') for v in lines])
    return text

def pipeline(output_csv):
    root = '..\\dataset'
    dataset_list = ['fudan','sougou']
    cls_list = ['Economy','Sports']
    for dataset in dataset_list:
        for cls in cls_list:
            filename_list = os.listdir(os.path.join(root, dataset, cls))
            for filename in tqdm(filename_list):
                text = read_file_to_text(os.path.join(root, dataset, cls, filename))
                data = pd.DataFrame([[dataset, cls, cls_list.index(cls), text]])
                data.to_csv(output_csv, mode='a', index=None, header=None)



if __name__=='__main__':
    
    import os 
    output_csv = '..\\data_process_v1.csv'
    pipeline(output_csv=output_csv)