import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import pll_datst, coll, mono_datst
from preprocessing import load_data
# from model import xlmb2b

data_obj = load_data()
df_prllel, df_eng, df_de = data_obj.final_data()

pll_train_ds = pll_datst(df_prllel)
mono_train_ds_eng = mono_datst(df_en)
mono_train_ds_de = mono_datst(df_de, lang='de')
b_sz = 32
model_ed = xlmb2b()
model_de = xlmb2b()

pll_train_loader = DataLoader(pll_train_ds,batch_size=b_sz, collate_fn = partial(coll, pll_dat = True))
mono_train_loader_en = DataLoader(mono_train_ds_en, batch_size=b_sz, collate_fn = partial(coll, pll_dat =False))
mono_train_loader_de = DataLoader(mono_train_ds_de, batch_size=b_sz, collate_fn = partial(coll, pll_dat =False))
optimizer_ed = torch.optim.Adam(model_ed.parameters(), lr = 0.01)
optimizer_de = torch.optim.Adam(model_ed.parameters(), lr = 0.01)
mseloss = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()

def convert_to_probs(batch_y) :
  '''returns indices which should be 1'''
  out = torch.zeros(batch_y.shape[0], batch_y.shape[1], vocab_size)
  for i in range(batch_y.shape[0]):
    for j in range(batch_y[i].shape[0]):
      out[i][j][batch_y[i][j]] = 1

  return out

num_epochs = 20
thresh = 0.5
losses = [[1000, 1000]]
for epoch in range(num_epochs) :
  if(losses[-1][0]>thresh or losses[-1][1]>thresh) :
    for i, batch in enumerate(pll_train_loader) :
      probs, tr_embd, trfrmr_out = model_ed(batch)
      loss_pll = cross_entropy_loss(probs, convert_to_probs(batch['Y']['content']))
      # loss_pll = criterion(yhat_ed, batch['Y']['content'])
      # passing something to model_de to get 'yhat_de'
      loss_b2b = cross_entropy_loss(yhat_de, convert_to_probs(batch['X']['content']))
      loss1 = loss_pll + loss_b2b

      optimizer_ed.zero_grad()
      # optimizer_de.zero_grad()
      loss_pll.backward()
      loss_b2b.backward()
      optimizer_ed.step()
      # optimizer_de.step()

      yhat_de = model_de(batch)
      yhat_ed = modle_de(yhat_de)
      loss_pll = cross_entropy_loss(yhat_de, convert_to_probs(batch['X']['content']))
      loss_b2b = cross_entropy_loss(yhat_ed, convert_to_probs(batch['Y']['content']))
      loss2 = loss_pll + loss_b2b

      # optimizer_ed.zero_grad()
      optimizer_de.zero_grad()
      loss_pll.backward()
      loss_b2b.backward()
      # optimizer_ed.step()
      optimizer_de.step()

      losses.appened([loss1,loss2])

  else:
    for i, batch in enumerate(mono_train_loader_en):
      probs, tr_embd, trfrmr_out = model_ed(batch)
      yhat_de = modle_de(yh)
      loss_b2b = cross_entropy_loss(yhat_de, covert_to_probs(batch['X']['content']))
      loss1 = loss_b2b

      optimizer_ed.zero_grad()
      # optimizer_de.zero_grad()
      loss_b2b.backward()
      optimizer_ed.step()
      # optimizer_de.step()
      losses.append([loss1])

    for i, batch in enumerate(mono_train_loader_de):
      yhat_de = model_de(batch)
      yhat_ed = modle_de(yhat_de)
      loss_b2b = criterion(yhat_ed, batch['Y']['content'])
      loss2 = loss_b2b

      # optimizer_ed.zero_grad()
      optimizer_de.zero_grad()
      loss_b2b.backward()
      # optimizer_ed.step()
      optimizer_de.step()

      losses[-1].append(l2)