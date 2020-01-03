import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import pll_datst, coll, mono_datst
from preprocessing import load_data, tokenizer
from model import xlmb2b
import tqdm

from nltk.translate.bleu_score import corpus_bleu

data_obj = load_data()
df_prllel, df_en, df_de = data_obj.final_data()

pll_train_ds = pll_datst(df_prllel)
mono_train_ds_eng = mono_datst(df_en)
mono_train_ds_de = mono_datst(df_de, lang='de')
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
     the rows with all 0 probs are due to padding of all
     sequences to same length'''
  x = torch.zeros((vocab_size))
  y = probs.reshape(-1,d_model)
  return y[y!=x]

def swap(batch,sr_embd,tr_embd, pll = True) :
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

    return batch, z['content'], z

num_epochs = 20
thresh = 0.5
losses = [[1000, 1000]]
for epoch in tqdm(range(num_epochs)) :

  model_ed.pll_dat = True
  model_de.pll_dat = True

  for i, batch in enumerate(pll_train_loader) :

    probs, sr_embd, tr_embd, trfrmr_out = model_ed(batch)
    loss_pll = cross_entropy_loss(probs, convert_to_probs(batch['Y']['content']))
    batch, a, b, c, d = swap(batch, sr_embd, tr_embd)
    probs_, sr_embd_, tr_embd_, trfrmr_out_ = model_de(batch, True)
    loss_b2b = cross_entropy_loss(reshape_n_edit(probs_), a.reshape(-1,1)) #since token_id is same as position in vocabulary
    loss1 = loss_pll + loss_b2b
    optimizer_ed.zero_grad()
    optimizer_de.zero_grad()
    loss1.backward()
    optimizer_de.step()
    optimizer_ed.step()

    if(epochs%20 == 0):
      print("Forward Model: ", calculate_bleu(b, model_ed(c), dict_vocab))

    batch['X']['content'] = b
    batch['Y']['content'] = a

    probs, sr_embd, tr_embd, trfrmr_out = model_de(batch)
    loss_pll = cross_entropy_loss(probs, convert_to_probs(batch['Y']['content']))
    batch, a, b, c, d = swap(batch, sr_embd, tr_embd)
    probs_, sr_embd_, tr_embd_, trfrmr_out_ = model_ed(batch, True)
    loss_b2b = cross_entropy_loss(probs_, convert_to_probs(a))
    loss1 = loss_pll + loss_b2b
    optimizer_ed.zero_grad()
    optimizer_de.zero_grad()
    loss1.backward()
    optimizer_de.step()
    optimizer_ed.step()

    if(epochs%20 == 0):
      print("Backward Model: ", calculate_bleu(b, model_de(c), dict_vocab))
    
    losses.appened([loss1,loss2])

# Training on monolingual data if the above losses are sufficiently low:

  if(losses[-1][0]<thresh or losses[-1][1]<thresh):

    model_ed.pll_dat = False
    model_de.pll_dat = False

    for i, batch in enumerate(mono_train_loader_en):
      probs, sr_embd, tr_embd, lengs = model_ed(batch)
      batch, a, b = swap(batch, sr_embd, tr_embd, False)
      probs, sr_embd, tr_embd, lengs = model_de(batch, True)
      loss_b2b = cross_entropy_loss(probs, covert_to_probs(a))

      loss1 = loss_b2b

      optimizer_ed.zero_grad()
      optimizer_de.zero_grad()
      loss_b2b.backward()
      optimizer_ed.step()
      optimizer_de.step()

      losses.append([loss1])

    for i, batch in enumerate(mono_train_loader_de):
      probs, sr_embd, tr_embd, lengs = model_de(batch)
      batch, a, b = swap(batch, sr_embd, tr_embd, False)
      probs, sr_embd, tr_embd, lengs = model_ed(batch, True)
      loss_b2b = cross_entropy_loss(probs, covert_to_probs(a))

      loss2 = loss_b2b

      optimizer_ed.zero_grad()
      optimizer_de.zero_grad()
      loss_b2b.backward()
      optimizer_ed.step()
      optimizer_de.step()

      losses[-1].append(loss2)
