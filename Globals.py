import torch
from preprocessing import tokenizer
import torch_xla.core.xla_model as xm
#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 1000
thresh_for_mono_data = 0.5
thresh_for_xlm_weight_freeze = 0.7 
thresh_to_stop_xlm_to_plt_prgrsiv = 10 #_______
thresh_to_start_real_to_pred_prgrsiv = 5 #_______
xlm_freezed = True
en_lang_id = tokenizer.lang2id['en']
de_lang_id = tokenizer.lang2id['de']
