import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import pll_datst, coll, mono_datst
from preprocessing import load_data, tokenizer
from model import xlmb2b
import tqdm
from os import path

from nltk.translate.bleu_score import corpus_bleu

if path.exists("../../data/file_1.csv"):
	data_obj = load_data(load_ = False)
else:
	data_obj = load_data()

df_prllel, df_en, df_de = data_obj.final_data()

pll_train_ds = pll_datst(df_prllel)
mono_train_ds_eng = mono_datst(df_en)
mono_train_ds_de = mono_datst(df_de, lang='de')
vocab_size = tokenizer.vocab_size

b_sz = 32
batch_size = 32
d_model = 1024
model_ed = xlmb2b()
model_de = xlmb2b()

pll_train_loader = DataLoader(pll_train_ds,batch_size=b_sz, collate_fn = partial(coll, pll_dat = True))
mono_train_loader_en = DataLoader(mono_train_ds_en, batch_size=b_sz, collate_fn = partial(coll, pll_dat =False))
mono_train_loader_de = DataLoader(mono_train_ds_de, batch_size=b_sz, collate_fn = partial(coll, pll_dat =False))
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
  return y[y==y]

def swap(batch,sr_embd,tr_embd,pll=True) :
    if pll:
        z = batch['X']
        z1 = batch['Y']
        batch['X'] = batch['Y']
        batch['Y'] = z
        batch['X']['content'] = tr_embd
        batch['Y']['content'] = sr_embd

        return batch, z['content'], z1['content'], z, z1

    else:
        z = batch['X']
        batch['X']['content'] = tr_embd

    return batch, z['content'], z, None, None


def set_to_eval(model_lis, beam_size=3) :
    for model in model_lis :
        model.eval()
        model.beam_size = beam_size

def evaluate(model,beam_size=3,i) :
    set_to_eval(model,beam_size)
    print(str(i)+"th, Forward Model: ", model[0](c))
    print(str(i)+"th, Backward Model: ", model[1](d))

def run(model_forward,model_backward,batch,optimizers,pll=True,send_trfrmr_out=False)
    probs, sr_embd, tr_embd, trfrmr_out = model_forward(batch)
    if pll : loss_pll = cross_entropy_loss(reshape_n_edit(probs), batch['Y']['content'].reshape(-1,1))
    if not send_trfrmr_out :
        batch, a, b, c, d = swap(batch, batch['X']['content'], trfrmr_out, pll)
    batch, a, b, c, d = swap(batch, sr_embd, tr_embd, pll)
    probs_, sr_embd_, tr_embd_, trfrmr_out_ = model_backward(batch, !send_trfrmr_out)
    loss_b2b = cross_entropy_loss(reshape_n_edit(probs_), a.reshape(-1,1)) #since token_id is same as position in vocabulary
    if pll : loss = loss_pll + loss_b2b
    else : loss = loss_b2b
    for optimizer in optimizers :
        optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizers :
        optimizer.step()
    return a,b,loss


def freeze_weights(model) :
    for param in model.parameters() :
        param.requires_grad = False

def unfreeze_weights(model) :
    for param in model.parameters() :
        param.requires_grad = True

num_epochs = 1000
thresh = 0.5
losses_epochs = []
optimizers = [optimizer_de,optimizer_ed]

for epoch in tqdm(range(num_epochs)) :

    print(epoch)
    model_ed.pll_dat=True
    model_de.pll_dat=True
    losses = [[], []]

    for i, batch in enumerate(pll_train_loader) :

        batch['Y']['content'], batch['X']['content'], loss1 = run_pll(model_ed,model_de,batch,optimizers)
        if epoch%20==0 : evaluate([model_ed,model_de],beam_size=3,1)
        _,_,loss2 = run_pll(model_de,model_ed,batch,optmizers)
        if epoch%20==0 : evaluate([model_ed,model_de],beam_size=3,2)
        losses[0].append(loss1)
        losses[1].append(loss2)
    losses_epochs.append({'pll' : [losses[0].sum()/len(losses[0]), losses[1].sum()/len(losses[1]]})

#Training on monolingual data if the above losses are sufficiently low:

    if(losses_epochs[-1]['pll'][0]<thresh or losses[-1]['pll'][1]<thresh):

        print("Going for Monolingual Training")

        model_ed.pll_dat = False
        model_de.pll_dat = False
        losses = [[], []]
        for i, batch in enumerate(mono_train_loader_en):

            _,_,loss1 = run(model_ed,model_de,batch,optimizers,pll=False)

            losses[0].append(loss1)

        for i, batch in enumerate(mono_train_loader_de):

            _,_,loss2 = run(model_de,model_ed,batch,optimizers,pll=False)

            losses[1].append(loss2)

        losses_epochs.append({'mono':[sum(losses[0])/len(losses[0]), sum(losses[1])/len(losses[1])]})
