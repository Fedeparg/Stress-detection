import neurokit2 as nk
import biosppy as bsp
import matplotlib.pyplot as plt
import numpy as np

""" HR, HRV y EDA 
    HR: Duh
    
    HRV:
    -Time_Domain
    -RMSS
    -SDSD
    -SDRR/RMSSD (SDRR = Standard Deviation RR) 
    -pNNx (x aun por decidir, p = porcentaje. Habria que calcualr el total por intervalo)
    -RELATIVE_RR
    -VLF, LF, HF (Aplicar filtro para esto)
    -Ratio LF/HF
    
    EDA:
    -
    
    """

def extract_features(data, sampling_rate):
    eda_raw = data[:,0]
    ecg_raw = data[:,1]
    print(data.shape)
    np.array_split(data, sampling_rate)
    print(data.shape)
    # print(len(np.take(eda_raw, list(range(0,700)))))
    # ts ,eda_processed, onsets, peaks, amplitudes = bsp.eda.eda(eda, sampling_rate=700, show=True)
