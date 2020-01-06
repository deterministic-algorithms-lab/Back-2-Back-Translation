import torch
import pandas as pd
from transformers import XLMTokenizer, XLMWithLMHeadModel, XLMModel
import numpy as np
tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-ende-1024")

class load_data():
	def __init__(self, load_ = True, paths = ['../../data/train.en','../../data/train.de'], pll_size = 10**5):
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
		    input_ids = torch.tensor(tokenizer.encode('<s>'+f.readline()+'</s>')[1:-1]).numpy()
		    self.src_tokens.append(input_ids)
		    i = i + 1

		with open(self.trgt_lang_path, 'rt') as f:
		  while(i!=2*self.pll_size):
		    input_ids = torch.tensor(tokenizer.encode('<s>'+f.readline()+'</s>')[1:-1]).numpy()
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

			for df in [df_prllel, df_eng, df_de]:
				df.to_csv('../../data/file_'+str(d)+'.csv', header=False, index=False)
				d = d+1
		else:
			df_prllel = pd.read_csv('../../data/file_0.csv',header=None, dtype=np.float64, names=['en','de'])
			print("Hererere", df_prllel.loc[0,'en'])
			df_eng = pd.read_csv('../../data/file_1.csv', header=None, dtype=np.float64)
			df_de = pd.read_csv('../../data/file_2.csv', header=None, dtype=np.float64, names=['en','de'])

		return df_prllel, df_eng, df_de