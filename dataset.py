from torch.util.data import Dataset, DataLoader

class pll_datst(Dataset) :
    def __init__(self, df, sr_lang = 'en', tr_lang = 'de') :
        super().__init__()
        self.df = df
        self.sr_lang = sr_lang
        self.tr_lang = tr_lang
    def __len__(self) :
        return len(df)
    def __getitem__(self, i) :
        zs = self.df.loc[i,self.sr_lang].shape[0]
        zt = self.df.loc[i,self.tr_lang].shape[0]
        return {'X' : {'content' : self.df.loc[i,self.sr_lang], 'langs' : torch.LongTensor([model.config.lang2id[self.sr_lang]]*zs),
          'position_ids' : torch.LongTensor([i for i in range(zs)]) , 'lengths' : zs } ,
          'Y' : {'content' : self.df.loc[i,self.tr_lang], 'langs' : torch.LongTensor([model.config.lang2id[self.tr_lang]]*zt),
          'position_ids' : torch.LongTensor([i for i in range(zt)]) , 'lengths' : zt }  }

class mono_datst(Dataset) :
    def __init__(self,df,lang='en') :
        super().__init__()
        self.df = df
        self.lang = lang
    def __len__(self) :
        return len(df)
    def __getitem__(self,i) :
        z = self.df.loc[i,self.lang].shape[0]
        return {'X' : {'content' : self.df.loc[i,self.lang], 'langs' : torch.LongTensor([model.config.lang2id[self.lang]]*z),
                  'position_ids' : torch.LongTensor([i for i in range(z)]) , 'lenghts' : z } }

def coll(batch, pll_dat) :
    b_sz=len(batch)
    batch2 = {}
    l = ['X','Y'] if pll_dat else ['X']
    for key in l :
        batch1 = {}
        batch1['content'] = torch.nn.utils.rnn.pad_sequence([batch[i][key]['content'] for i in range(b_sz)])
        batch1['langs'] = torch.stack([batch[i][key]['langs'] for i in range(b_sz)])
        batch1['position_ids'] = torch.stack([batch[i][key]['position_ids'] for i in range(b_sz)])
        batch1['lengths'] = torch.stack([batch[i][key]['lengths'] for i in range(b_sz)])
        batch1['attention_mask'] = torch.stack([torch.cat([torch.ones(batch[i][key]['lengths']),
                                                                     torch.zeros(batch1['lengths'].max()-batch[i][key]['lengths'])], dim=0)
                                                                     for i in range(b_sz)])
        batch2[key] = batch1
    return batch2

