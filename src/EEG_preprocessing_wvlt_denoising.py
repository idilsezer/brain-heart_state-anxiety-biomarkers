
import mne
import numpy as np
import pywt
import os
import matplotlib.pyplot as plt
import pandas as pd

def denoise_eeg_wavelet_positive(eeg_signal, wavelet='coif3', level=None, threshold_type='soft', thrshld=5e-05):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(eeg_signal, wavelet, level=level)
    
    # Set threshold for coefficient values
    # if 0.3 * max(coeffs[0])>=thrshld-1e-5:
    # threshold = 0.75 * max(coeffs[0])
    # else:
    threshold = thrshld
    # print(0.025 * max(coeffs[0]))
    print(threshold)
    
    # Threshold the wavelet coefficients
    if threshold_type == 'soft':
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    elif threshold_type == 'hard':
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
    elif threshold_type == 'greater':
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='greater') for c in coeffs]
    elif threshold_type == 'less':
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='less') for c in coeffs]
    else:
        raise ValueError("Invalid threshold type. Use 'soft', 'hard', 'greater' or 'less'.")
    
    eeg_denoised = pywt.waverec(coeffs_thresholded, wavelet)
    
    return eeg_denoised

def denoise_eeg_wavelet_negative(eeg_signal, wavelet='coif3', level=None, threshold_type='soft',thrshld=5e-05):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(eeg_signal, wavelet, level=level)
    
    # Set threshold for coefficient values
    # if 0.3 * max(coeffs[0])>=thrshld-1e-5:
    # threshold = 0.95 * max(coeffs[0])
    # else:
    threshold = thrshld
    # print(0.005 * max(coeffs[0]))
    # print(threshold)
    inverted_coeffs = [-c for c in coeffs]
    # Threshold the wavelet coefficients
    if threshold_type == 'soft':
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in inverted_coeffs]
    elif threshold_type == 'hard':
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='hard') for c in inverted_coeffs]
    elif threshold_type == 'greater':
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='greater') for c in inverted_coeffs]
    elif threshold_type == 'less':
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='less') for c in inverted_coeffs]
    else:
        raise ValueError("Invalid threshold type. Use 'soft', 'hard', 'greater' or 'less'.")
    
    eeg_denoised = pywt.waverec(coeffs_thresholded, wavelet)
    
    return -1*eeg_denoised




#%%
vid_ord = pd.read_csv('ordres.csv')
for sub_id in [str(i) for i in range(35)]:
    for vid in [str(v) for v in range(1,3)]:
        vid_eeg = mne.io.read_raw_brainvision('SUB' +sub_id+'_VIDEO'+vid+'.vhdr',preload=True)
        ch_list = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
               'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
               'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
               'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1',
               'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
               'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6',
               'AF8', 'AF4', 'F2', 'Iz']
        vid_eeg.pick_channels(ch_list)
        vid_type = np.array(vid_ord[vid_ord['Participants']==int(sub_id)]['video0'+vid])[0]
        
        vid_eeg.notch_filter(50)
        vid_eeg.filter(1,48)
        overlapping_seg = []
        end_times = []
        info = vid_eeg.info    
        
        
        thrshld = np.array(pd.read_excel('Thresholding_value.xlsx')['Threshold'])
        pos_th = []
        neg_th = []
        den = []
        rem = []
        sig_mat = vid_eeg.get_data()
        denoised_mat = np.zeros(sig_mat.shape)
        removed_mat = np.zeros(sig_mat.shape)
        for ch in range(sig_mat.shape[0]):
            eeg_denoised_p = denoise_eeg_wavelet_positive(sig_mat[ch,:],threshold_type='less', thrshld=thrshld[ch])
            try:
                denoised_mat[ch,:] = -1*denoise_eeg_wavelet_negative(eeg_denoised_p,threshold_type='less', thrshld=thrshld[ch])
            except:
                denoised_mat[ch,:] = -1*denoise_eeg_wavelet_negative(eeg_denoised_p,threshold_type='less', thrshld=thrshld[ch])[:-1]
            removed1 = denoise_eeg_wavelet_positive(sig_mat[ch,:],threshold_type='greater', thrshld=thrshld[ch])
            removed2 = denoise_eeg_wavelet_negative(eeg_denoised_p,threshold_type='greater', thrshld=thrshld[ch])
            try:
                removed_mat[ch,:] = removed1-removed2
            except:
                removed_mat[ch,:] = removed1[:-1]-removed2[:-1]
            
            # plt.figure(figsize=(10, 12))
        
            # # First subplot
            # plt.subplot(3, 1, 1)
            # plt.plot(sig_mat[ch, :])
            # plt.ylim((np.min(sig_mat[ch, :]), np.max(sig_mat[ch, :])))
            # # plt.xlim((0, 10000))
            # plt.title('Sub_'+sub_id+'_Channel_'+ch_list[ch]+'_Video_')
            # plt.ylabel('Amplitude')
        
            # # Second subplot
            # plt.subplot(3, 1, 2)
            # plt.plot(denoised_mat[ch,:])
            # plt.ylim((np.min(sig_mat[ch, :]), np.max(sig_mat[ch, :])))
            # # plt.xlim((0, 10000))
            # plt.title('Sub_'+sub_id+'_Channel_'+ch_list[ch]+'_Video_')
            # plt.ylabel('Amplitude')
        
            # # Third subplot
            # plt.subplot(3, 1, 3)
            # plt.plot(removed_mat[ch,:])
            # plt.ylim((np.min(sig_mat[ch, :]), np.max(sig_mat[ch, :])))
            # # plt.xlim((0,10000))
            # plt.title('Sub_'+sub_id+'_Channel_'+ch_list[ch]+'_Video_')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
        # den.append(denoised_mat)
        # rem.append(removed_mat)
            
        # test = np.array(den)
        # test2 = np.array(rem)
        # qq = test.reshape((64,-1))
        # ww = test2.reshape((64,-1))
        # plt.plot(ww[0,:])
        # plt.plot(qq[0,:])
        
            
        denoised_sig = mne.io.RawArray(denoised_mat,vid_eeg.info)
        removed_sig = mne.io.RawArray(removed_mat,vid_eeg.info)
        
        
        denoised_sig_theta = denoised_sig.filter(4,8)
        denoised_sig_alpha = denoised_sig.filter(8,13)
        denoised_sig_alpha_low = denoised_sig.filter(8,10)
        denoised_sig_alpha_high = denoised_sig.filter(10,13)
        denoised_sig_beta = denoised_sig.filter(13,30)
        denoised_sig_gamma = denoised_sig.filter(30,45)

        # vid_eeg.plot()
        # denoised_sig.plot()
        # removed_sig.plot()
        
        # denoised_sig.export('SUB' +sub_id+'_VIDEO_'+vid_type+'_Denoised.vhdr', fmt='brainvision', overwrite=True)
        # removed_sig.export('SUB' +sub_id+'_VIDEO_'+vid_type+'_Removed.vhdr', fmt='brainvision', overwrite=True)