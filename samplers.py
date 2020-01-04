import pandas as pd
from dataset import mono_datst, pll_datst
from torch.utils.data import DataLoader
from preprocessing import load_data, tokenizer

dic = tokenizer.decoder

class simpler_sampler() :

    def __init__(self, dic = dic, b_sz = 256, d_model=1024, lang='en' ) :
        self.b_sz = b_sz
        self.d_model = d_model
        self.dtalodr_for_dic = self.get_dtalodr_for_dic(dic, lang)
        self.dic = dic                                                          #dictionary  {'i' : 'token_embdng_of_ith_token'} , 0=<i<=len(vocab)-1 
        self.xlm_embdngs = None
        self.calc_xlm_embdngs()

    @staticmethod
    def get_dtalodr_for_dic(self, dic, lang='en') :
        df = pd.DataFrame.from_dict(data = dic, orient='index')
        df.rename(columns = {'0':lang})
        ds = mono_datst(df, lang)
        dl = DataLoader(ds, batch_size = self.b_sz, coll_fn = partial(coll, pll_dat=False))
        return dl


    def calc_xlm_embdngs(self) :
        dic_word_embds = None
        xlm_embdngs1 = []
        for i , batch in enumerate(self.dtalodr_for_dic) :
            dic_word_embds = t.xlm(**batch['X'])[0]
            xlm_embdngs1.append(dic_word_embds.reshape(b_sz, d_model))
        self.xlm_embdngs = torch.cat(xlm_embdngs1, dim=0)


    def get_probs(self, vec, t, get_xlm_embdngs=False, calc_embdngs = True) :
        '''
        vec.shape == 1,d_model(1024)
        dic --> dictionary of token embdngs {'i' : 'token_embdng_of_ith_token'}
        t ---> model to use to get 1024 size embdngs
        '''
        if calc_embdngs :
            self.calc_xlm_embdngs()
        probs = F.softmax(torch.mm(vec, self.xlm_embdngs.t()).t())                       #vocabX1
        if get_xlm_embdngs :
            return self.xlm_embdngs, probs
        else :
            return probs

    def get_max_prob_vec(self, vec, t, calc_embdngs=True) :
        '''
        Inputs --> same as above ;
        Returns 1024 size embdng of max prob vec
        Needed for mono_data. It's output is sent for the next time step of trnsfrmr dcodr while decoding mono_dat sequence.
        '''
        self.xlm_embdngs, probs = self.get_probs(vec,t,dic,lang,get_xlm_embdngs=True, calc_embdngs=calc_embdngs)
        _, next_token_index = probs.max(0)
        return self.xlm_embdngs[next_token_index]


class beam_search(simpler_sampler):
    """
        Takes the probability distribution across vocab and the atention mask as the input,
        shapes: (batch_size, seq_len, vocab_size) and (batch_size, seq_len) resp.
        Applies beam search for every sequence in the batch, giving word-level output
        output shape: (batch_size, seq len, 1)
    """
    def __init__(self, dic, b_sz = 256, d_model = 1024, lang = 'en', beam_size=3):
        self.simpler_sampler = simpler_sampler(dic, b_sz, d_model, lang)
        self.beam_size = beam_size

    def simple_beam(self, batch, pos = 0):
        """
            Takes as input the batch and the position to look at
            batch: (batch_size, seq_len, vocabulary size)
            Outputs the indexes of the beam words and the corr probs
        """
        return torch.topk(batch[:,pos,:], self.beam_size)        # returns (values,indices) indices: the position of chosen words in the vocab


    def apply_beam_search(self, attn_mask, logits):
        """
            logits: (..., vocabulary size)
        """
        vec = []
        for i in range(logits.shape[0]):
            if(attn_mask[i] == 1):
                values, indices = torch.topk(logits[i], self.beam_size)
                for j in range(self.beam_size):
                    emb = self.simpler_sampler.dic[indices[j]]
                    vec.append(emb)
                    probs = self.simpler_sampler.get_probs(vec, t, False, False)

                to_consider.append(logits[i])

        logits_ = torch.tensor(to_consider)


    def apply_beam_search_recur(self, attn_mask, logits, beam = 3):
        """
            logits: (..., vocabulary size)
        """
        prob_sums = {}

        if(logits[0] == END_TOKEN):
            return 1

        if(attn_mask[0] == 1):
            values, indices = torch.topk(logits[0], beam)
            for j in range(beam):
                vec = self.simpler_sampler.dic[indices[j]]
                logits[1] = self.simpler_sampler.get_probs(vec.reshape(1,-1), t, False, False)
                prob_sums[j] += apply_beam_search_recur(attn_mask, logits[1:])
                return values[j]


    def searcher(self, batch):
        out = []
        for logit in batch:
            out.append(apply_beam_search(logit))

        return torch.stack(out)

class nuc_sampler(simpler_sampler) :

    def __init__(self, dic, b_sz = 256, d_model = 1024, lang = 'en', beam_size=3):
        self.simpler_sampler = simpler_sampler(dic, b_sz, d_model, lang)
        self.beam_size = beam_size

    @staticmethod
    def apply_top_k_or_top_p(logits, top_k=0, top_p=0):

        '''
        Filter a distribution of logits using top-k and/or top-p filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: 0 => no filtering, >0 => keep only top k tokens with highest probability.
            top_p: 0 => no filtering, >0 => keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
        '''

        if top_k != 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1]
            logits[indices_to_remove] = -float('Inf')

        if top_p != 0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        return logits


    def nuc_sample(self, vec, t, token_id = True, calc_embdngs=True) :
        '''
        samples next token for 1 vector
        vec.shape == d_model(1024)
        if token_id is True : returns token_id of sampled token
        else : return 1024 size embdngs of sampled token
        '''

        temperature = 1.0
        top_k = 0
        top_p = 0.9
        if token_id : probs = self.simpler_sampler.get_probs(vec, t, calc_embdngs)
        else : xlm_embdngs, probs = get_probs(vec,t,True,calc_embdngs)

        filtered_logits = apply_top_k_or_top_p(probs, top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token_index = torch.multinomial(probabilities, 1)

        if token_id : return self.simpler_sampler.dic[str(next_token_index)]
        else : return self.simpler_sampler.xlm_embdngs[next_token_index]


    def nuc_sampler(self, model_out, att_mask, t, get_xlm_embdngs=False, calc_embdngs=True) :
        '''
        sample next token where model_out is a batch
        model_out.shape = [b_sz,seq_len,self.d_model(1024)] Output of transformer decoder
        expected output of shape -->if get_xlm_embdngs is False :  [b_sz,seq_len,1] else [b_sz,seq_len,1024] 
        '''
        if get_xlm_embdngs: not_get_xlm_embs = False
        else: not_get_xlm_embs = True 
        out = []
        for i in range(len(att_mask.shape[0])) :
            for_1_sample = []
            for j in range(len(att_mask.shape[1])) :
                if att_mask[i][j] == True  and att_mask[i][j+1] != True :
                    for_1_sample.append(self.nuc_sample(model_out[i][j], t, not_get_xlm_embs, calc_embdngs))
                else :
                    if get_xlm_embdngs :
                        for_1_sample.append(torch.zeros((d_model)))
                    else :
                        for_1_sample.append(torch.zeros((1)))
            out.append(torch.stack(for_1_sample))
        return torch.stack(out)
