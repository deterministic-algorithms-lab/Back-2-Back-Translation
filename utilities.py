from abc import ABC
class model_utils(ABC) :
    
    def __init__(self) :
        super().__init__()

    def cut_and_paste_down( batch, dim=1) :
        return batch.transpose(0,1).reshape(-1)

    def cut_and_paste_up( batch, dim=1, beam_size=1) :
            '''batch.size = [batch_size*beam_size, z]
               return size = [batch_size,z*beam_size]'''
        return batch.reshape(beam_size,-1,batch.shape[1]).transpose(0,1).reshape(-1,beam_size*batch.shape[1])

    def convert_mask_to_inf( mask):
        mask[mask==0] = -np.inf
        mask[mask==1] = 0
        return mask


    def infs_to_zero(self,mask) :
        mask[mask==0]=1
        mask[mask==-np.inf] = 0
        return mask

    def get_tgt_mask(self, tr_len, it_no=None) :
        x = np.zeros((tr_len,tr_len), dtype=np.float32)
        upp_indices = np.triu_indices(tr_len, k=1)
        x[upp_indices] = -np.inf
        if it_no is not None :
            e = torch.tensor(x, dtype = torch.float32, device=device)
            e[e!=e[it_no]] = -np.inf
            return e
        return torch.tensor(x, dtype=torch.float32, device=device)

    
    def final_layer(self, trfrmr_out, mask) :
        x = trfrmr_out[mask.bool()]
        if self.it_no is not None :
            return self.final_linear(x), mask
        else :
            return self.final_linear(x)
    
    def mask_fr_mask(self) :
        m = torch.zeros((self.bs,self.max_tr_seq_len),dtype=torch.bool, device=device)
        m[:,self.it_no+1]=1
        m[~self.not_done_samples] = 0
        return m
    
    def apply_final_layer(self, trfrmr_out, mask) :
        if self.it_no is not None :
            mask_ = self.tgt_key_pad_mask[self.not_done_samples][:,self.it_no].bool()
            mask = torch.zeros((self.bs,self.max_tr_seq_len), dtype=torch.bool, device=device)
            mask[self.mask_fr_mask()] = mask_
        return self.final_layer(trfrmr_out, mask)

    def cycle_dims(self, tensor, clockwise=True) :
            dims = torch.arange(-1,len(tensor.shape)-1)
            if clockwise :
                y = tuple(dims)
                return tensor.permute(y)
            z = list(dims+2)
            z = z+[0]
            return tensor.permute(z)

    def k_sample_to_flat(self, tokens, langs, positions) :
        '''
        tokens.size == [b_sz, seq_len, k_sample]
        langs.size,positions.size == [b_sz, seq_len]
        '''
        tokens = self.cycle_dims(tokens)
        langs = langs.repeat(tokens.size(0),1)
        positions = positions.repeat(tokens.size(0),1)
        tokens = tokens.reshape(-1, tokens.size(2))
        return tokens, langs, positions

    def flat_to_k_sample(self, plt_embed) :
        '''plt_embed.shape = [k_sample*b_sz, seq_len, d_model]
           return shape = [b_sz, seq_len, k_sample, d_model]'''
        plt_embed = plt_embed.reshape(k,-1,plt_embed.size(1),plt_embed.size(2))
        return plt_embed.transpose(0,1).transpose(1,2)
    
    def plt_embed(self, tokens, langs, positions) :
        '''Returns plt_embdng of shape [b_sz, seq_len, d_model] or
            [b_sz, seq_len, k, d_model] if nucleus sampling is done.'''
        if len(tokens.shape)==3 :
            k = tokens.size(2)
            tokens, langs, positions = self.k_sample_to_flat(tokens, langs, positions)                    
        y = self.xlm.embeddings(tokens)     
        z = y + self.xlm.position_embeddings(positions)
        plt_embed = z+self.xlm.lang_embeddings(langs)
        if len(tokens.shape)==3 :
            plt_embed = self.flat_to_k_sample(plt_embed)        
        return plt_embed
    
    def embed_for_decoder(self, output_at_it_no, lang_id) :
        y = self.xlm.embeddings(output_at_it_no)   #batch_sizeXd_model
        z = y + self.xlm.position_embeddings(self.it_no)
        return (z+self.xlm.lang_embeddings(lang_id))

    def indi(self) :
        y = self.not_done_samples.long()
        quotients = torch.div(y,self.beam_size)
        rems = torch.remainder(y,self.beam_size)
        return quotients,rems
    
    def get_msk_fr_prev_probs_entry(self) :
        x = torch.zeros((self.actual_bs, self.max_tr_seq_len+1, self.beam_size), dtype=torch.bool, device=device)
        x[:,self.it_no,:] = self.not_done_samples.reshape(-1,self.beam_size)
        return x

    def reform(self, trfrmr_out) :
        prev_probs_here = self.prev_probs[:,self.it_no-1,:] if self.it_no!=0 else torch.zeros((self.actual_bs, self.beam_size),device=device)
        m = (trfrmr_out.t()+self.prev_probs_here.reshape(-1)).t()
        m[~self.not_done_samples] = 0
        m = m.reshape(-1,self.beam_size*self.vocab_size)
        msk_fr_prev_probs_entry = self.get_msk_fr_prev_probs_entry()
        value, indices = m.topk(self.beam_size, dim=1)
        self.prev_probs[msk_fr_prev_probs_entry]=value.reshape(-1)[self.not_done_samples]
        indices = torch.remainder(indices, self.vocab_size)
        indices = indices.reshape(-1)
        return indices

    def change_attn_for_xlm(self, dic) :
        k='attention_mask'
        dic[k]=dic[k].bool()
        dic[k]=~dic[k]
        dic[k]=dic[k].float()
        return dic

    def calc_just_now_completed_samples_mask(self,ind) :
        self.just_now_completed_samples_mask[:,:] = False
        self.just_now_completed_samples_mask[self.not_done_samples==True] = ~ind
        self.not_done_samples[self.not_done_samples==True] = ind



class clone_batch() :

    def __init__(self, n, pll_dat=True) :
        super().__init__()
        self.n = n
        self.pll_dat = pll_dat

    def transform_xlm_in(self, sample) :
        '''Obtains all possible samples from 1 sample
           and returns 'sample' with content,position_ids
           and langs of size [self.n, z*self.n]
           of form (if self.n=3 and z=4 and sample['input_ids']=[abcd]) :-
           sample['input_ids'].t():- [[abcd00000000],
                                    [0000abcd0000],
                                    [00000000abcd]]'''
        l = ['X', 'Y'] if self.pll_dat else ['X']
        for key in l :
            z = len(sample[key]['input_ids'])
            for subkey in sample[key] :
                if subkey != 'lengths' :
                    sample[key][subkey] = torch.stack([torch.cat([torch.zeros((i*z)), sample[key][subkey], torch.zeros(((self.n-i-1)*z))])
                                                for i in range(self.n)]).t()
        return sample


    def get_xlm__att_mask(self, batch) :
        '''If input :- [[abcd00000000],
                        [0000abcd0000],
                        [00000000abcd],other samples]
              output:- [[111100000000],
                        [000011110000],
                        [000000001111], similarly for other samples]'''
        max_size = batch['lengths'].max()
        att_mask = []
        for elem in batch['lengths'] :
            #self.n elements corres. to 'elem' length
            att_mask.append( torch.stack([torch.cat([torch.zeros((i*elem)), torch.ones((elem)), torch.zeros((max_size-(i+1)*elem))])
                                                for i in range(self.n)]) )
        return torch.cat(att_mask)



