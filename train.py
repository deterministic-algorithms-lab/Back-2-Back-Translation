import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import pll_datst, coll, mono_datst
from preprocessing import load_data, tokenizer
from model import xlmb2b
from tqdm import tqdm
from os import path
from functools import partial
from nltk.translate.bleu_score import corpus_bleu
import multiprocessing as mp

if path.exists("../../data/file_1.csv"):
	data_obj = load_data(load_ = False)
else:
	data_obj = load_data()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df_prllel, df_en, df_de = data_obj.final_data()
pll_train_ds = pll_datst(df_prllel)
mono_train_ds_en = mono_datst(df_en)
mono_train_ds_de = mono_datst(df_de, lang='de')
vocab_size = tokenizer.vocab_size

b_sz = 32
batch_size = 32
d_model = 1024

model_ed = xlmb2b().double().to(device)
model_de = xlmb2b().double().to(device)
model_ed.xlm = model_de.xlm
del model_ed.xlm

cpus = mp.cpu_count()
pll_train_loader = DataLoader(pll_train_ds,batch_size=b_sz, collate_fn = partial(coll, pll_dat = True), pin_memory=True, num_workers=cpus)
mono_train_loader_en = DataLoader(mono_train_ds_en, batch_size=b_sz, collate_fn = partial(coll, pll_dat =False), pin_memory=True, num_workers=cpus)
mono_train_loader_de = DataLoader(mono_train_ds_de, batch_size=b_sz, collate_fn = partial(coll, pll_dat =False), pin_memory=True, num_workers=cpus)
optimizer_ed = torch.optim.Adam(model_ed.parameters(), lr = 0.01)
optimizer_de = torch.optim.Adam(model_ed.parameters(), lr = 0.01)
mseloss = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()

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

def swap(batch,sr_embd,tr_embd,pll=True) :
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
        z = batch['X']['input_ids'].clone()
        batch['X']['input_ids'] = tr_embd
        batch1 = {}
        batch1['X']['input_ids'] = z
        for k,v in batch :
            if k!='input_ids' :
                batch1['X'][k]=v
    return batch, z, batch1

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
        for key, value in batch[elem] :
            batch[elem][key] = value.to(device, non_blocking=True)

def evaluate(model, i, beam_size=3) :
    set_to_eval(model,beam_size)
    print(str(i)+"th, Forward Model: ", model[0](c))
    print(str(i)+"th, Backward Model: ", model[1](d))

def run(model_forward,model_backward,batch,optimizers,pll=True,send_trfrmr_out=False):
    probs, sr_embd, tr_embd, trfrmr_out = model_forward(batch)
    if pll : loss_pll = cross_entropy_loss(reshape_n_edit(probs), remove_pad_tokens(batch['Y']['input_ids'].reshape(-1)) )
    if send_trfrmr_out :
        batch, a, b = swap(batch, batch['X']['input_ids'], trfrmr_out, pll)
    else :
        batch, a, b = swap(batch, sr_embd, tr_embd, pll)
    del probs
    probs_, sr_embd_, tr_embd_, trfrmr_out_ = model_backward(batch, not send_trfrmr_out)
    loss_b2b = cross_entropy_loss(reshape_n_edit(probs_), remove_pad_tokens(a.reshape(-1)))
    del probs_, sr_embd, sr_embd_, tr_embd, tr_embd_, trfrmr_out, trfrmr_out_
    if pll : loss = loss_pll + loss_b2b
    else : loss = loss_b2b
    for optimizer in optimizers :
        optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizers :
        optimizer.step()
    return a,b,loss


num_epochs = 1000
thresh_for_mono_data = 0.5
losses_epochs = {"pll" : [], "mono": []}
optimizers = [optimizer_de,optimizer_ed]
thresh_for_xlm_weight_freeze = 0.7
thresh_for_send_trfrmr_out = 0.9

for epoch in tqdm(range(num_epochs)) :

    print(epoch)
    model_ed.pll_dat=True
    model_de.pll_dat=True
    losses = [[], []]

    for i, batch in enumerate(pll_train_loader) :
        batch = send_to_gpu(batch, pll=True)
        batch['Y']['input_ids'], batch['X']['input_ids'], loss1 = run(model_ed,model_de,batch,optimizers)
        if epoch%20==0 : evaluate([model_ed,model_de], 1, beam_size=3)
        _,_,loss2 = run(model_de,model_ed,batch,optimizers)
        if epoch%20==0 : evaluate([model_ed,model_de], 2, beam_size=3)
        losses[0].append(loss1)
        losses[1].append(loss2)
    losses_epochs['pll'].append([losses[0].sum()/len(losses[0]), losses[1].sum()/len(losses[1])])

#Training on monolingual data if the above losses are sufficiently low:

    if(losses_epochs['pll'][-1][0]<thresh_for_mono_data or losses['pll'][-1][1]<thresh_for_mono_data):

        print("Going for Monolingual Training")

        model_ed.pll_dat = False
        model_de.pll_dat = False
        losses = [[], []]

        for i, batch in enumerate(mono_train_loader_en):
            batch = send_to_gpu(batch, pll=False)

            _,_,loss1 = run(model_ed,model_de,batch,optimizers,pll=False)

            losses[0].append(loss1)

        for i, batch in enumerate(mono_train_loader_de):
            batch = send_to_gpu(batch, pll=False)

            _,_,loss2 = run(model_de,model_ed,batch,optimizers,pll=False)

            losses[1].append(loss2)

        losses_epochs['mono'].append([losses[0].sum()/len(losses[0]), losses[1].sum()/len(losses[1])])
