import torch
import pandas as pd
from transformers import XLMTokenizer, XLMWithLMHeadModel, XLMModel
import pickle

tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-ende-1024")

class load_data():
    def __init__(self, load_ = True, dataset_path, pll_size = 10**5):
        paths = [dataset_path+'/train.en',dataset_path+'/train.de']
        self.src_lang_path = paths[0]
        self.trgt_lang_path = paths[1]
        self.pll_size = pll_size
        self.load_ = load_

    def load(self):
        i = 0
        self.src_tokens = []
        self.trgt_tokens = []

        with open(self.src_lang_path, 'rt') as f:
          while(i!=self.pll_size):
            input_ids = torch.tensor(tokenizer.encode('<s><s>'+f.readline()+'</s>')[1:-1])
            self.src_tokens.append(input_ids)
            i = i + 1

        with open(self.trgt_lang_path, 'rt') as f:
          while(i!=2*self.pll_size):
            input_ids = torch.tensor(tokenizer.encode('<s><s>'+f.readline()+'</s>')[1:-1])
            self.trgt_tokens.append(input_ids)
            i = i + 1

    def final_data(self):
        if(self.load_):
            self.load()
            zipped_list = list(zip(self.src_tokens, self.trgt_tokens))
            df_prllel = pd.DataFrame(zipped_list, columns = ['en', 'de'], dtype=object)
            df_eng = pd.DataFrame(self.src_tokens)
            df_de = pd.DataFrame(self.trgt_tokens)
            d = 0
            '''
            for df in [df_prllel, df_eng, df_de]:
                with open(self.dataset_path+'/file_'+str(d)+'.pkl', 'wb+') as f :
                    pickle.dump(df,f)
                d = d+1
            '''
        else:
            [df_prllel,df_en,df_de] = [None]*3
            d=0
            for var in [df_prllel,df_en, df_de] :
                with open(self.dataset_path+'/file_'+str(d)+'.pkl', 'rb') as f :
                    var = pickle.load(f)
                d=d+1
                
        return df_prllel, df_eng, df_de
