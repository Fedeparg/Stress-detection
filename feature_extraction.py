import neurokit2 as nk
import biosppy as bsp
import numpy as np
from hrv import HRV
from ecgdetectors import Detectors
import frequency_ecg as fr_ecg
from scipy import stats
import cvxEDA as cvx
import math

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
    detector = Detectors(sampling_rate)
    hrv_class = HRV(sampling_rate)

    # Borrar si todo va bien
    # np.array_split(data, sampling_rate)

    # PROCESAR POR TROZOS

    # FOR splitted data
    # Unir horizontalmente ecg y eda
    # unir verticalmente cada iteracion

    splitted = split_in_seconds(data, sampling_rate, 60)
    seccion = None
    for idx, m in enumerate(splitted):
        print(f'Procesando split {idx} con longitud {len(m)}')

        eda = eda_processing(m[:, 0])
        try:
            ecg = ecg_processing(m[:, 1], detector, hrv_class)
            full_iteration = np.hstack((eda, ecg))
            if seccion is None:
                seccion = np.empty((0, len(full_iteration)))
            seccion = np.vstack((seccion, full_iteration))
        except Exception:
            continue

    return seccion


def split_in_seconds(data, sampling_rate, seconds):
    """
    Params
    ------
    data: Los datos a trocear

    sampling_rate: La frecuencia de muestreo en Hz

    seconds: El tamaño de la ventana que queremos en segundos.
    """
    rang = list(range(0, len(data), sampling_rate*seconds))
    rang.pop(0)
    splitted = np.split(data, rang)
    return splitted

# HRV y esas cosas que se hacen
# Llamar despues de split_in_seconds
# -----------------------PROCESAMIENTO DEL ECG-----------------------


def ecg_processing(ecg_signal, detector, hrv_class):
    mean = np.mean(ecg_signal)
    median = np.median(ecg_signal)
    std = np.std(ecg_signal)
    skewness = stats.skew(ecg_signal)
    kurtosis = stats.kurtosis(ecg_signal)
    # print(
    # f"Media: {mean}\nMediana: {median}\nSTD: {std}\nSkewness: {skewness}\nKurtosis: {kurtosis}")

    r_peaks = detector.pan_tompkins_detector(ecg_signal)
    rmssd = hrv_class.RMSSD(r_peaks)
    sdsd = hrv_class.SDSD(r_peaks)
    sdrr = hrv_class.SDNN(r_peaks)
    sdrr_rmssd = sdrr/rmssd
    # print(f"RMSSD: {rmssd}\nSDSD: {sdsd}\nSDRR_RMSSD: {sdrr/rmssd}")
    pNN50 = hrv_class.pNN50(r_peaks)
    pNN20 = hrv_class.pNN20(r_peaks)
    # print(f"pNN50: {pNN50}\npNN20: {pNN20}")

    # A PARTIR DE AQUI, EL CODIGO ES DE PICKUS

    SD1 = (1 / np.sqrt(2)) * sdsd
    SD2 = np.sqrt((2 * sdrr ** 2) - (0.5 * sdsd ** 2))

    # Analisis de frecuencias
    intervals = hrv_class._intervals(r_peaks)
    freq = fr_ecg.frequencyDomain(intervals)

    vlf_power = freq['VLF_Power']
    lf_power = freq['LF_Power']
    hf_power = freq['HF_Power']
    lf_hf = freq['LF/HF']

    # print(f"SD1: {SD1}\nSD2: {SD2}\nfreq: {freq}")

    rel_rr = relative_rr(
        intervals)
    # print(len(r_peaks))

    # CONCATENARLO TODO EN UN SUPER ARRAY (HORIZONTAL)
    ecg = np.hstack((mean, median, std, skewness, kurtosis, rmssd, sdsd, sdrr_rmssd,
                     pNN50, pNN20, SD1, SD2, rel_rr, vlf_power, lf_power, hf_power, lf_hf))
    return ecg


def relative_rr(intervals):
    """
        intervals: Intrevalos RR en milisegundos
    """
    relative_rr_sig = []

    # Para cada par, aplico la formula del paper
    for idx, r in enumerate(intervals):
        if idx == 0:
            pass
        else:
            # 2 [RRi - RRi-1 / RRi + RRi-1]
            RRi = 2*((r - intervals[idx-1]) / (r + intervals[idx-1]))
            relative_rr_sig.append(RRi)

    # Saco todas las variables temporales de la misma forma que con los intervalos RR originales
    mean = np.mean(relative_rr_sig)
    median = np.median(relative_rr_sig)
    std = np.std(relative_rr_sig)
    skewness = stats.skew(relative_rr_sig)
    kurtosis = stats.kurtosis(relative_rr_sig)
    #print(f"DATOS DE RELATIVE_RR:\nMedia: {mean}\nMediana: {median}\nSTD: {std}\nSkewness: {skewness}\nKurtosis: {kurtosis}")
    return np.hstack((mean, median, std, skewness, kurtosis))


# -----------------------PROCESAMIENTO DEL EDA-----------------------

def eda_processing(eda_signal):
    # De aqui sacamos los picos y onsets
    processed_eda = nk.eda_process(eda_signal, sampling_rate=700)
    peaks = processed_eda[1]['SCR_Peaks']
    # r es la señal scr
    [r, p, t, l, d, e, obj] = cvx.cvxEDA(eda_signal, 1/700)
    scr = r

    mean_scr = np.mean(scr)
    max_scr = np.max(scr)
    min_scr = np.min(scr)
    # Preguntar por range
    skewness = stats.skew(scr)
    kurtosis = stats.kurtosis(scr)

    # Derivada 1 de SCR
    derivada1 = np.gradient(r, edge_order=1)
    mean_der1 = np.mean(derivada1)
    std_der1 = np.std(derivada1)

    # Derivada 2 de SCR
    derivada2 = np.gradient(r, edge_order=2)
    mean_der2 = np.mean(derivada2)
    std_der2 = np.std(derivada2)

    # Peaks
    peaks = processed_eda[1]['SCR_Peaks']
    mean_peaks = np.mean(peaks)
    max_peaks = np.max(peaks)
    min_peaks = np.min(peaks)
    std_peaks = np.std(peaks)

    # Investigar (otra vez) onsets

    # ALSC, INSC, APSC, RMSC
    alsc_result = alsc(scr)
    insc_result = insc(scr)
    apsc_result = apsc(scr)
    rmsc_result = rmsc(scr)

    eda = np.hstack((mean_scr, max_scr, min_scr, skewness, kurtosis, mean_der1, std_der1, mean_der2,
                     std_der2, mean_peaks, max_peaks, min_peaks, alsc_result, insc_result, apsc_result, rmsc_result))
    return eda


def alsc(scr):
    alsc = []
    for idx, v in enumerate(scr):
        if idx == 0:
            pass
        else:
            alsc.append(math.sqrt(1 + ((scr[idx] - scr[idx-1])**2)))
    return np.sum(alsc)


def insc(scr):
    insc = []
    for v in scr:
        insc.append(abs(v))
    return np.sum(insc)


def apsc(scr):
    apsc = []
    for v in scr:
        apsc.append(v**2)
    return (np.sum(apsc)/len(scr))


def rmsc(scr):
    return math.sqrt(apsc(scr))
