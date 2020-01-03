import numpy as np
import torch
from preprocessing import tokenizer
from transformers import XLMTokenizer, XLMWithLMHeadModel, XLMModel

batch_size = 32
dic = tokenizer.decoder

class xlmb2b(torch.nn.Module):
    def __init__(self, dic = dic, d_model=1024, trfrmr_nlayers=4, batch_size=batch_size, pll_dat=True) :
        super().__init__()
        self.xlm = XLMModel.from_pretrained('xlm-mlm-ende-1024')
        self.d_model = d_model
        decoder_layer = torch.nn.TransformerDecoderLayer(self.d_model, n_head=8)
        self.trnsfrmr_dcodr = torch.nn.TransformerDecoder(decoder_layer, num_layers=trfrmr_nlayers)
        self.pll_data = pll_dat
        self.batch_size = batch_size
        self.mx_tr_seq_len = 120
        self.end_tok = 1 #Token id of end token
        self.simpler_sampler = simpler_sampler(dic, b_sz = 256, d_model=1024, lang='en')
        self.final_layer = nn.Linear(self.d_model, self.vocab_size)
        self.softmax = nn.Softmax(dim=0)
        self.dic_tensor = torch.tensor([v for k,v in tokenizer.encoder.items()]) #tensor with i_th token's id at i_th position
        self.vocab_size = self.dic_tensor.shape[0]

    def get_tgt_mask(self, tr_len) :
        x = np.zeros((tr_len,tr_len))
        upp_indices = np.triu_indices(tr_len)
        x[upp_indices] = -np.inf
        return torch.tensor(x)

    def convert_mask_to_inf(mask):
        mask[mask==1] = 0
        mask[mask==0] = -np.inf
        return mask

    def final_layer(self, trfrmr_out, mask) :
        mask = self.convert_mask_to_inf(mask)
        x = trfrmr_out+mask
        return self.softmax(self.final_linear(x).reshape(-1, self.vocab_size)).reshape(trfrmr_out.shape[0],-1,self.vocab_size)

    def apply_final_layer(self, trfrmr_out, mask, samples_to_do, it_no=None) :
        if it_no is not None :
            trfrmr_out, mask = trfrmr_out[samples_to_do], mask[samples_to_do]
            mask[:,it_no] = 1
        return self.final_layer(trfrmr_out, mask)

    def forward(self, dat, already_embed = False) :                             #dat is a dictionary with keys==keyword args of xlm

        if self.pll_data :
            inp = dat['X']
            out = dat['Y']

            if not already_embed :
                sr_embd = self.xlm(**inp)[0]
                tr_embd = self.xlm(**out)[0]                                    #(xlm_out/trnsfrmr_tar).shape = (batch_size,seq_len,1024)
            else :
                sr_embd = inp['content']
                tr_embd = inp['content']

            tr_len = out['lengths'].max()
            tgt_mask = self.get_tgt_mask(tr_len)
            trfrmr_out = self.trnsfrmr_dcodr(tgt=tr_embd.transpose(1,2), memory=sr_embd.transpose(1,2), tgt_mask=tgt_mask,
                                             tgt_key_padding_mask=out['attention_mask'],
                                             memory_key_padding_mask=inp['attention_mask'])
            trfrmr_out = trfrmr_out.transpse(1,2)
            probs = self.apply_final_layer(trfrmr_out, out['attention_mask'])

            return probs, sr_embd, tr_embd, trfrmr_out

        else :

            inp = dat['X']
            sr_embed = self.xlm(**inp)[0]
            tgt_key_pad_mask = torch.zeros((self.batch_size, self.max_tr_seq_len))
            mem_key_pad_mask = inp['attention_mask']
            tgt_mask = self.get_tgt_mask(self.max_tr_seq_len)
            tr_embd = torch.zeros((self.batch_size, self.max_tr_seq_len, self.d_model))
            tr_embd = tr_embd
            not_done_samples = torch.tensor([i for i in range(self.batch_size)])
            it_no = 0                                                           #if nth word of target sequence is being predicted,
            final_out = []                                                      #then iteration number(it_no) == n-1
            lengs = torch.zeros((self.batch_size))

            while True :
                trfrmr_out = self.trnsfrmr_dcodr(tgt=tr_embd.transpose(1,2), memory=sr_embd.transpose(1,2), tgt_mask=tgt_mask,
                                                 tgt_key_padding_mask=tgt_key_pad_mask,
                                                 memory_key_padding_mask=mem_key_pad_mask)
                trfrmr_out = trfrmr_out.transpse(1,2)
                trfrmr_out = self.apply_final_layer(trfrmr_out, tgt_key_pad_mask, it_no, not_done_samples)
                output_at_it_no = torch.zeros((self.batch_size,1))
                output_at_it_no[not_done_samples] = self.dic_tensor[trfrmr_out[not_done_samples,it_no,:].max(2)[1]]
                final_out.append(output_at_it_no)
                ind = output_at_it_no[not_done_samples]!=self.end_tok
                new_done_samples_len = not_done_samples.shape[0]-ind.shape[0]
                if new_done_samples_len!=0 :
                    lengs[~ind] = it_no+1
                    mem_key_pad_mask[~ind] = torch.zeros((new_done_samples_len, inp['lengths'].max()))
                    tgt_key_pad_mask[~ind] = torch.zeros((new_done_samples_len, self.max_tr_seq_len))
                    not_done_samples = not_done_samples[ind]
                tgt_key_pad_mask[ind,it_no+1] = torch.ones((ind.shape[0],1))
                if not_done_samples.shape[0]==0 or it_no==self.max_tr_seq_len-1:
                    return torch.stack(final_out), sr_embed , tr_embd, lengs
                tr_embd[ind,it_no,:] = self.xlm.embeddings(output_at_it_no)		      #Adding next words embeddings to context for decoder
                it_no+=1
