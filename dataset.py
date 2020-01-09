from torch.utils.data import Dataset, DataLoader
from preprocessing import tokenizer
from torch.nn.utils.rnn import pad_sequence as padd
import torch
import torch.nn as nn

class pll_datst(Dataset) :
    def __init__(self, df, sr_lang = 'en', tr_lang = 'de') :
        super().__init__()
        self.df = df
        self.sr_lang = sr_lang
        self.tr_lang = tr_lang
    def __len__(self) :
        return len(self.df)
    def __getitem__(self, i) :
        zs = self.df.loc[i,self.sr_lang].shape[0]
        zt = self.df.loc[i,self.tr_lang].shape[0]
        return {'X' : {'input_ids' : self.df.loc[i,self.sr_lang], 'langs' : tokenizer.lang2id[self.sr_lang],
          'position_ids' : torch.LongTensor([i for i in range(zs)]) , 'lengths' : zs } ,
          'Y' : {'input_ids' : self.df.loc[i,self.tr_lang], 'langs' : tokenizer.lang2id[self.tr_lang],
          'position_ids' : torch.LongTensor([i for i in range(zt)]) , 'lengths' : zt }  }

class mono_datst(Dataset) :
    def __init__(self,df,lang='en') :
        super().__init__()
        self.df = df
        self.lang = lang
    def __len__(self) :
        return len(self.df)
    def __getitem__(self,i) :
        z = self.df.loc[i,self.lang].shape[0]
        return {'X' : {'input_ids' : self.df.loc[i,self.lang], 'langs' : tokenizer.lang2id[self.lang],
                  'position_ids' : torch.LongTensor([i for i in range(z)]) , 'lengths' : z } }

pdv = tokenizer.pad_token_id

def coll(batch, pll_dat) :
    b_sz=len(batch)
    batch2 = {}
    l = ['X','Y'] if pll_dat else ['X']
    for key in l :
        batch1 = {}
        batch1['input_ids'] = padd([batch[i][key]['input_ids'] for i in range(b_sz)], batch_first=True, padding_value=pdv)
        batch1['lengths'] = torch.LongTensor([batch[i][key]['lengths'] for i in range(b_sz)])
        max_size = int(batch1['lengths'].max())
        batch1['position_ids'] = torch.LongTensor([[i for i in range(max_size)]]*b_sz)
        batch1['langs'] = torch.LongTensor([ [batch[i][key]['langs']]*max_size for i in range(b_sz)])
        batch1['attention_mask'] = torch.stack([torch.cat([torch.zeros(batch[i][key]['lengths'], dtype=torch.float32),
                                                                     torch.ones(max_size-batch[i][key]['lengths'], dtype=torch.float32)], dim=0)
                                                                     for i in range(b_sz)])
        batch2[key] = batch1
    return batch2
