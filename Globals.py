device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 1000
thresh_for_mono_data = 0.5
thresh_for_xlm_weight_freeze = 0.7 
thresh_to_stop_xlm_to_plt_prgrsiv = #_______
thresh_to_start_real_to_pred_prgrsiv = #_______
xlm_freezed = True