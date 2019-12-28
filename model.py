import numpy as np
import torch
from sampler import simpler_sampler

class xlmb2b(torch.nn.Module):
    def __init__(self, dic, d_model=1024, trfrmr_nlayers=4, batch_size=batch_size, pll_dat=True) :
        super().__init__()
        self.xlm = XLMModel.from_pretrained('xlm-mlm-ende-1024')
        self.d_model = d_model
        decoder_layer = torch.nn.TransformerDecoderLayer(self.d_model, n_head=8)
        self.trnsfrmr_dcodr = torch.nn.TransformerDecoder(decoder_layer, num_layers=trfrmr_nlayers)
        self.pll_data = pll_dat
        self.batch_size = batch_size
        self.mx_tr_seq_len =
        self.end_tok = #index of end_token in simpler_sampler's dic
        self.simpler_sampler = simpler_sampler(dic, b_sz = 256, d_model=1024, lang='en')


    def get_tgt_mask(self, tr_len) :
            x = np.zeros((tr_len,tr_len))
            upp_indices = np.triu_indices(tr_len)
            x[upp_indices] = -np.inf
            return torch.tensor(x)

    def forward(self, dat, already_embed = False) :                             #dat is a dictionary with keys==keyword args of xlm

        if self.pll_data :
            inp = dat['X']
            out = dat['Y']

            if not already_embed :
                sr_embd = self.xlm(**inp)[0]                                    #(xlm_out/trnsfrmr_tar).shape = (batch_size,seq_len,1024)
            else :
                sr_embd = inp['content']

            tr_embd = self.xlm(**out)[0]

            tr_len = out['lengths'].max()
            tgt_mask = self.get_tgt_mask(tr_len)
            trfrmr_out = self.trnsfrmr_dcodr(tgt=tr_embd.transpose(1,2), memory=sr_embd.transpose(1,2), tgt_mask=tgt_mask,
                                             tgt_key_padding_mask=out['attention_mask'],
                                             memory_key_padding_mask=inp['attention_mask'])
            trfrmr_out = trfrmr_out.transpse(1,2)
            return trfrmr_out, tr_embd

        else :

            inp = dat['X']
            sr_embed = self.xlm(**inp)[0]
            tgt_key_pad_mask = torch.zeros((self.batch_size, self.max_tr_seq_len))
            mem_key_pad_mask = inp['attention_mask']
            tgt_mask = self.get_tgt_mask(self.max_tr_seq_len)
            tr_embd = torch.zeros((self.batch_size, self.max_tr_seq_len, self.d_model))
            done_samples = []
            it_no = 0                                                           #if nth word of target sequence is being predicted,
            final_out = []                                                      #then iteration number(it_no) == n-1

            while True :
                trfrmr_out = self.trnsfrmr_dcodr(tgt=tr_embd.transpose(1,2), memory=sr_embd.transpose(1,2), tgt_mask=tgt_mask,
                                             tgt_key_padding_mask=tgt_key_pad_mask,
                                             memory_key_padding_mask=mem_key_pad_mask)
                trfrmr_out = trfrmr_out.transpse(1,2)
                output_it_no = []                                               #list of outputs @ it_no corres. to various samples in batch
                for i in range(len(trfrmr_out.shape[0])) :                      #Loop over all samples to compute next words
                    if i not in done_samples :
                        output_it_no.append(self.simpler_sampler.get_max_prob_vec(trfrmr_out[i][it_no], self, calc_embdngs=True))
                    else :
                        output_it_no.append(torch.zeros((self.d_model)))
		z = torch.stack(output_it_no)
                final_out.append(z)

                for i in range(len(output_it_no)) :                             #Loop over all samples to change their masks
                    if output_it_no[i] == simpler_sampler.xlm_embdngs[self.end_tok].squeeze() or it_no == self.max_tr_seq_len :
                        done_samples.append(i)
                        tgt_key_pad_mask[i] = torch.zeros((self.max_tr_seq_len))
                        mem_key_pad_mask[i] = torch.zeros((inp['lengths'].max()))
                        if len(done_samples) == self.batch_size :
                            return torch.stack(final_out), tr_embd
                    elif i not in done_samples :
                        tgt_key_pad_mask[i][it_no+1] = 1

                tr_embd[:,it_no,:] = z				                #Adding next words embeddings to context for decoder
                it_no+=1
