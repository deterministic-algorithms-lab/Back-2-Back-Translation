import numpy as np
import torch
import torch.nn as nn
from preprocessing import tokenizer
from transformers import XLMTokenizer, XLMWithLMHeadModel, XLMModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
dic = tokenizer.decoder

class xlmb2b(torch.nn.Module):
    def __init__(self, dic = dic, d_model=1024, trfrmr_nlayers=4, pll_dat=True) :
        super().__init__()
        self.xlm = XLMModel.from_pretrained('xlm-mlm-ende-1024')
        self.d_model = d_model
        decoder_layer = torch.nn.TransformerDecoderLayer(self.d_model, nhead=8)
        self.trnsfrmr_dcodr = torch.nn.TransformerDecoder(decoder_layer, num_layers=trfrmr_nlayers)
        self.pll_data = pll_dat
        self.mx_tr_seq_len = 120
        self.end_tok = 1 #Token id of end token
        self.softmax = nn.Softmax(dim=1)
        self.dic_tensor = torch.tensor([v for k,v in tokenizer.encoder.items()]) #tensor with i_th token's id at i_th position
        self.vocab_size = self.dic_tensor.shape[0]
        self.final_linear = nn.Linear(self.d_model, self.vocab_size)
        self.it_no = None
        self.beam_size = 1

    def infs_to_zero(self,mask) :
        mask[mask!=mask] = 0
        return mask

    def get_tgt_mask(self, tr_len, it_no=None) :
        x = np.zeros((tr_len,tr_len), dtype=np.float32)
        upp_indices = np.triu_indices(tr_len, k=1)
        x[upp_indices] = -np.inf
        if it_no is not None :
            e = torch.tensor(x, dtype = np.float32).to(device)
            e[e!=e[it_no]] = -np.inf
            return e
        return torch.tensor(x, dtype=torch.float32).to(device)

    def convert_mask_to_inf(self , mask):
        mask[mask==0] = -np.inf
        return mask

    def final_layer(self, trfrmr_out, mask) :
        mask = self.convert_mask_to_inf(mask)
        x = (trfrmr_out.transpose(2,1).transpose(1,0)+mask).transpose(0,1).transpose(1,2)
        return self.softmax(self.final_linear(x).reshape(-1, self.vocab_size)).reshape(trfrmr_out.shape[0],-1,self.vocab_size)

    def apply_final_layer(self, trfrmr_out, mask) :
        if self.it_no is not None :
            trfrmr_out, mask = trfrmr_out[self.samples_to_do], self.tgt_key_pad_mask[self.samples_to_do]
            mask[:,self.it_no] = 1
        return self.final_layer(trfrmr_out, mask)

    def embed_for_decoder(self, output_at_it_no, lang_long_tensor) :
        y = self.xlm.embeddings(output_at_it_no)   #batch_sizeXd_model
        z = y + self.xlm.position_embeddings(self.it_no).expand_as(y)
        return z+self.xlm.lang_embeddings(lang_id)

    def cut_and_paste_down(self, batch, dim=1) :
        return batch.transpose(0,1).reshape(-1)

    def cut_and_paste_up(self, batch, dim=1) :
        '''batch.size = [batch_size*beam_size, z]
           return size = [batch_size,z*beam_size]'''
        return batch.reshape(self.beam_size,-1,batch.shape[1]).transpose(0,1).reshape(-1,self.beam_size*batch.shape[1])

    def reform(self, trfrmr_out) :
        if self.beam_size == 1 :
            return trfrmr_out[self.not_done_samples,self.it_no,:].max(2)[1]
        else :
            m = trfrmr_out[self.not_done_samples,self.it_no,:]+self.cut_and_paste_down(self.prev_probs[:,self.it_no,:],dim=1)[self.not_done_samples]
            m = self.cut_and_paste_up(m)
            self.cut_and_paste_down(self.prev_probs[:,self.it_no+1,:])[self.not_done_samples], indices = m.topk(self.beam_size, dim=1)
            indices = torch.remainder(indices, self.d_model)
            indices = self.cut_and_paste_down(indices)
            return indices

    def change_attn_for_xlm(self, dic) :
        k='attention_mask'
        dic[k]=dic[k].bool()
        dic[k]=~dic[k]
        dic[k]=dic[k].float()
        return dic

    def choose(self) :
        '''Chooses final output beam for each sample using beam_size,
           final_out,prev_probs'''
        x = self.prev_probs.max(1, keepdim=True)[1]                #batch_sizeX1Xbeam_size
        s = torch.gather(self.prev_probs, dim=1, index=x)          #batch_sizeX1Xbeam_size
        y = s.max(2)[1]                                            #batch_sizeX1
        i = torch.tensor([i for i in range(y.shape[0])])
        final_out = torch.stack(self.final_out).transpose(0,1)
        final_out = final_out.reshape(self.beam_size,-1,final_out.shape[1])
        return final_out[y,i,:]

    def forward(self, dat, already_embed = False) :                             #dat is a dictionary with keys==keyword args of xlm

        if self.pll_data :
            inp = dat['X']
            out = dat['Y']

            if not already_embed :
                sr_embd = self.xlm(**self.change_attn_for_xlm(inp))[0]
                tr_embd = self.xlm(**self.change_attn_for_xlm(out))[0]                                    #(xlm_out/trnsfrmr_tar).shape = (batch_size,seq_len,1024)
            else :
                sr_embd = inp['input_ids']
                tr_embd = out['input_ids']

            tr_len = int(out['lengths'].max())
            tgt_mask = self.get_tgt_mask(tr_len)
            trfrmr_out = self.trnsfrmr_dcodr(tgt=tr_embd.transpose(0,1),
                                             memory=sr_embd.transpose(0,1), tgt_mask=tgt_mask,
                                             tgt_key_padding_mask=~(out['attention_mask'].bool()),
                                             memory_key_padding_mask=~(inp['attention_mask'].bool()))
            trfrmr_out = trfrmr_out.transpose(0,1)
            probs = self.apply_final_layer(trfrmr_out, out['attention_mask'].float())
            out['attention_mask'] = self.infs_to_zero(out['attention_mask'])
            return probs, sr_embd, tr_embd, trfrmr_out

        else :

            inp = dat['X']
            self.sr_embd = self.xlm(**self.change_attn_for_xlm(inp))[0].repeat((self.beam_size,1,1))
            self.bs = inp['input_ids'].shape[0]*self.beam_size
            self.tgt_key_pad_mask = torch.zeros((self.bs, self.max_tr_seq_len))
            self.mem_key_pad_mask = inp['attention_mask'].repeat((self.beam_size,1))
            self.tgt_mask = self.get_tgt_mask(self.max_tr_seq_len,0)
            self.tr_embd = torch.zeros((self.bs, self.max_tr_seq_len, self.d_model))
            self.not_done_samples = torch.tensor([i for i in range(self.bs)])
            self.it_no = 0                                                           #if nth word of target sequence is being predicted,
            self.final_out = []                                                      #then iteration number(it_no) == n-1
            self.lengs = torch.zeros((bs))

            if self.beam_size==1 :
                self.prev_probs = torch.zeros((bs/self.beam_size,self.max_tr_seq_len+1,self.beam_size))
            else :
                self.probs = []

            while True :
                trfrmr_out = self.trnsfrmr_dcodr(tgt=self.tr_embd.transpose(0,1),
                                                 memory=self.sr_embd.transpose(0,1), tgt_mask=tgt_mask,
                                                 tgt_key_padding_mask=~(self.tgt_key_pad_mask.bool()),
                                                 memory_key_padding_mask=~(self.mem_key_pad_mask.bool()))
                trfrmr_out = trfrmr_out.transpse(0,1)
                trfrmr_out = self.apply_final_layer( trfrmr_out, self.tgt_key_pad_mask.float() )
                if self.beam_size==1 :
                    self.probs.append(trfrmr_out)
                dic_indices = self.reform(trfrmr_out)
                output_at_it_no = torch.zeros((self.bs,1))
                output_at_it_no[self.not_done_samples] = self.dic_tensor[dic_indices]
                self.final_out.append(output_at_it_no)
                ind = output_at_it_no[self.not_done_samples]!=self.end_tok
                new_done_samples_len = self.not_done_samples.shape[0]-ind.shape[0]
                if new_done_samples_len!=0 :
                    self.lengs[~ind] = it_no+1
                    self.mem_key_pad_mask[~ind] = torch.zeros((new_done_samples_len, inp['lengths'].max()))
                    self.tgt_key_pad_mask[~ind] = torch.zeros((new_done_samples_len, self.max_tr_seq_len))
                    self.not_done_samples = self.not_done_samples[ind]
                self.tgt_key_pad_mask[ind,it_no+1] = torch.ones((ind.shape[0],1))
                if self.not_done_samples.shape[0]==0 or self.it_no==self.max_tr_seq_len-1:
                    self.it_no = None
                    if self.beam_size==1 :
                        return torch.stack(self.probs).transpose(0,1), self.sr_embd, self.tr_embd, torch.stack(self.final_out).transpose(0,1)
                    else :
                        return self.choose()
                self.tr_embd[ind,self.it_no,:] = self.embed_for_decoder(output_at_it_no, inp['langs'][:,self.it_no])              #Adding next words embeddings to context for decoder
                self.it_no+=1
                self.tgt_mask = self.get_tgt_mask(self.tgt_mask, self.it_no)
