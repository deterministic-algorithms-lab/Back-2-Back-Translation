import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import pll_datst, coll, mono_datst
from preprocessing import load_data, tokenizer
from model2 import xlmb2b
from tqdm import tqdm
from os import path
from functools import partial
from nltk.translate.bleu_score import corpus_bleu
import multiprocessing as mp
from Globals import *
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser(description= 'Train the Model')
parser.add_argument('--dataset_path')
parser.add_argument('--p', type=float)
parser.add_argument('--ksample', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--trfrmr_nlayers', type=int)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()


mseloss = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()
cpus = mp.cpu_count()

def calculate_bleu(ref, cand, weights = (0.25, 0.25, 0.25, 0.25)):
  """
     ref: (batch_size, seq_len, 1)
     cand: (batch_size, seq_len, 1)
  """
  references = []
  candidates = []
  dict_ = tokenizer.decoder
  for i in range(ref.shape[0]):
    refs = []
    cands = []
    for j in range(ref[i].shape[0]):
      refs.append(dict_[ref[i][j]])
      cands.append(dict_[cand[i][j]])
    references.append([refs])
    candidates.append(cands)

  return corpus_bleu(references, candidates, weights)


def reshape_n_edit(probs) :
  '''returns probs while removing rows with all 0 probs
     the rows with all nan probs are due to padding of all
     sequences to same length'''
  y = probs.reshape(-1,vocab_size)
  return y[y==y].reshape(-1,vocab_size)

def assign_features(batch) :
    batch['X']['attention_mask'] = (batch['X']['input_ids']==tokenizer.pad_token_id).float()
    batch['X']['lengths'] = batch['X']['attention_mask'].sum(dim=1).long()
    max_size = int(batch['X']['lengths'].max())
    bs = batch['X']['input_ids'].shape[0]
    batch['X']['position_ids'] = torch.tensor([[i for i in range(max_size)]*bs], dtype=torch.long)
    if (batch['X']['langs']==en_lang_id).sum() == 0 :
        batch['X']['langs'] = torch.LongTensor([[en_lang_id]*max_size for i in range(b_sz)])
    else :
        batch['X']['langs'] = torch.LongTensor([[de_lang_id]*max_size for i in range(b_sz)])
    return batch

def swap(batch,sr_embd,tr_embd,pll=True) :
    '''Replaces X with Y and input_ids with embeddings for pll data
        For mono data , replaces input_ids with predicted tokens'''
    if pll:
        z2=batch['X']
        z = batch['X']['input_ids'].clone()
        z1 = batch['Y']['input_ids'].clone()
        batch['X'] = batch['Y']
        batch['Y'] = z2
        batch['X']['input_ids'] = tr_embd
        batch['Y']['input_ids'] = sr_embd
        return batch, z, z1

    else:
        batch1 = {}
        batch1['X'] = {}
        for k, v in batch['X'].items() :
            batch1['X'][k] = v.clone()
        z = batch1['X']['input_ids']
        batch['X']['input_ids'] = tr_embd
        batch = assign_features(batch) 
    return batch, z, batch1

def flip_masks(batch) :
    batch['X']['attention_mask'] = (~(batch['X']['attention_mask'].bool())).float()
    batch['Y']['attention_mask'] = (~(batch['Y']['attention_mask'].bool())).float()
    return batch

def freeze_weights(model) :
    for param in model.parameters() :
        param.requires_grad = False

def unfreeze_weights(model) :
    for param in model.parameters() :
        param.requires_grad = True

def remove_pad_tokens(tensorr):
    j = tokenizer.pad_token_id
    return tensorr[tensorr!=j]

def set_to_eval(model_lis, beam_size=3) :
    for model in model_lis :
        model.eval()
        model.beam_size = beam_size

def send_to_gpu(batch, pll) :
    lis =['X', 'Y'] if pll else ['X']
    for elem in lis :
        for key, value in batch[elem].items() :
            batch[elem][key] = value.to(device, non_blocking=True)
    return batch

def evaluate(model, i, beam_size=3) :
    set_to_eval(model,beam_size)
    print(str(i)+"th, Forward Model: ", model[0](c))
    print(str(i)+"th, Backward Model: ", model[1](d))

def save_models(i) :
    if i%1000==0 :
        torch.save(model_ed.state_dict(),'weights/model_ed.param')
        torch.save(model_de.state_dict(), 'weights/model_de.param')

def synchronize() :
    if torch.cuda.is_available() :
        torch.cuda.synchronize()

def check_thresholds(loss1,loss2,model_ed,model_de, epochs) :
    global xlm_freezed
    if xlm_freezed and loss1<thresh_for_xlm_weight_freeze and loss2<thresh_for_xlm_weight_freeze:
        unfreeze_weights(model_ed.xlm)
        xlm_freezed = False
    elif not model_de.begin_prgrsiv_real_to_pred and loss1<thresh_to_start_real_to_pred_prgrsiv and loss2<thresh_to_start_real_to_pred_prgrsiv :
        model_de.begin_prgrsiv_real_to_pred = True
        model_ed.begin_prgrsiv_real_to_pred = True
        return
    elif model_de.begin_prgrsiv_xlm_to_plt and epochs>thresh_to_stop_xlm_to_plt_prgrsiv :
        model_de.begin_prgrsiv_xlm_to_plt = False
        model_ed.begin_prgrsiv_xlm_to_plt = False

class ForTraining(pl.LightningModule) :

    def get_dataset(self) :
        if path.exists(args.dataset_path+"/file_1.csv") :
            data_obj = load_data(load_ = False, dataset_path = args.dataset_path)
        else:
            data_obj = load_data(dataset_path=args.dataset_path)

        df_prllel, df_en, df_de = data_obj.final_data()
        pll_train_ds = pll_datst(df_prllel)
        
        if self.lang == 'en' :
            self.mono_train_ds = mono_datst(df_en)
        else :
            self.mono_train_ds = mono_datst(df_de, lang='de')
        self.pll_train_ds = pll_train_ds

    
    def __init__(self, args, mode, lang='en') :
        self.args = args
        self.model_ed = xlmb2b(trfrmr_nlayers=args.trfrmr_nlayers)
        self.model_de = xlmb2b(trfrmr_nlayers=args.trfrmr_nlayers)
        del self.model_ed.xlm
        self.model_ed.xlm = self.model_de.xlm
        self.model_ed.p = args.p
        self.model_de.p = args.p
        self.model_ed.beam_size = args.ksample
        self.model_de.beam_size = args.ksample
        self.mode = mode
        self.vocab_size = tokenizer.vocab_size
        self.b_sz = args.batch_size
        self.batch_size = args.batch_size
        self.d_model = 1024
        self.lang = lang
        self.get_dataset()
        self.model_ed.pll_dat=True
        self.model_de.pll_dat=True
        self.losses = [[], []]
    

    @pl.data_loader
    def train_dataloader(self) :
        if self.mode == 'pll' :
            return DataLoader(self.pll_train_ds,batch_size=b_sz, collate_fn = partial(coll, pll_dat = True), pin_memory=True, num_workers=cpus)
        else :
            return DataLoader(self.mono_train_ds, batch_size=b_sz, collate_fn = partial(coll, pll_dat =False), pin_memory=True, num_workers=cpus)

    def run(self, model_forward, model_backward, batch, pll=True):
        probs, sr_embd, tr_embd = model_forward(batch)
        
        if pll : 
            loss_pll = cross_entropy_loss(reshape_n_edit(probs), remove_pad_tokens(batch['Y']['input_ids'].reshape(-1)) )
        
        batch, a, b = swap(batch, sr_embd, tr_embd, pll)
        probs_, sr_embd_, tr_embd_ = model_backward(batch, True)
        loss_b2b = cross_entropy_loss(reshape_n_edit(probs_), remove_pad_tokens(a.reshape(-1)))
        
        if pll : loss = loss_pll + loss_b2b
        else : loss = loss_b2b
        
        return a,b,loss

    def training_step(self, batch, batch_num) :
        
        batch['Y']['input_ids'], batch['X']['input_ids'], loss1 = self.run(self.model_ed,self.model_de,batch)
        losses[0].append(loss1.item())
        
        batch = flip_masks(batch)
        _,_,loss2 = self.run(self.model_de,self.model_ed,batch)
        losses[1].append(loss2.item())
        
        check_thresholds(losses[0][-1],losses[1][-1], model_ed, model_de, epoch)
        save_models(i)
        
        return loss1+loss2
        
    def configure_optimizers(self) :
        params = list(self.model_ed.parameters()) + list(self.model_de.parameters())
        return torch.optim.Adam(params, lr=self.args.lr)
       

