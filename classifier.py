from read_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
import time

from ecgdetectors import Detectors
from hrv import HRV
from hrvanalysis import get_time_domain_features

import feature_extraction

# Numero de muestras por segundo
mps = 700


def process_data(data):
    ecg = data[:, 1]
    r_peaks = detector.pan_tompkins_detector(ecg)
    # rr_peaks = np.diff(r_peaks)
    print(get_time_domain_features(r_peaks[0:3]).get('rmssd'))
    hr = np.full(len(data), 0.0)
    hr2 = hrv_class.HR(rr_samples=r_peaks)
    index = range(len(r_peaks)-1)
    for i in index:
        if i == index[-1]:
            np.put(hr, range(r_peaks[i], len(hr)-1), hr2[i])
        else:
            np.put(hr, range(r_peaks[i], r_peaks[i+1]), hr2[i])

    return hr


def save_dataset(ds_path, dataset, name):
    """
    Funcion que permite almacenar un dataset pasado como parámetro
    """
    os.chdir(ds_path)
    with open(name + '.pkl', 'wb') as output:
        print("Dumping dataset...", end=' ')
        sys.stdout.flush()
        pickle.dump(dataset, output)
        print("Done!")


def load_dataset(ds_path):
    """
    Funcion que permite cargar un dataset pasado como parámetro
    """
    os.chdir(ds_path)
    with open('dataset_hr.pkl', 'rb') as file:
        print("Reading dataset...", end=' ')
        sys.stdout.flush()
        data = pickle.load(file, encoding='latin1')
        print("Done!")
    return data


if __name__ == '__main__':
    detector = Detectors(mps)
    hrv_class = HRV(mps)
    data = execute()
    feature_extraction.extract_features(data)
    # print(data[:][:5])
    # X = data[:, :2]  # 16 features
    # y = data[:, 2]
    # ecg = data[:, 1]  # Extraemos ECG

    # r_peaks son los puntos temporales en los que se detecta un latido
    # r_peaks = detector.pan_tompkins_detector(ecg)
    # hr es el ritmo cardiaco calculado mediante los r_peaks
    processed_data = process_data(data)
"""
    # -------------------
    # Tal vez pueda pasarle fragmentos para que calcule la variabilidad
    # test = hrv_class.RMSSD(r_peaks[2:5])
    # -------------------

    # plt.figure()
    # plt.plot(ecg)
    # # plt.plot(r_peaks)
    # plt.show()
    # Introducimos el ritmo cardiaco como variable de entrada al dataset
    X = np.column_stack((X, hr))
    new_dataset = np.column_stack((X, y))
    # new_dataset = insert_lag(new_dataset, 1)
    path = '/home/fedeparg/Stress-detection/'
    # save_dataset(path, new_dataset, 'dataset_hr_test')
    path = '/home/fedeparg/Stress-detection/dtst_hr_NW'

    data_reloaded = load_dataset(path)
    # print("This is da data" + str(data_reloaded.shape))
    # X = data_reloaded[:,:3]
    X = data_reloaded[:, [0]]

    ecg = data_reloaded[:, 1]  # Extraemos ECG

    # r_peaks son los puntos temporales en los que se detecta un latido
    start = time.time()
    r_peaks = detector.pan_tompkins_detector(ecg)
    print(time.time()-start)
    sys.stdout.flush()
    r_peaks = r_peaks_to_time(r_peaks)

    time_domain_feat = get_time_domain_features(r_peaks)
    print(time_domain_feat)
    time_domain_feat = get_time_domain_features(r_peaks[0:2])
    print(time_domain_feat)
    time_domain_feat = get_time_domain_features(r_peaks[1:3])
    print(time_domain_feat)
    
    y = data_reloaded[:, 3]

    print(X.shape)
    print(y.shape)
    train_features, test_features, train_labels, test_labels = train_test_split(X, y,
                                                                                test_size=0.25)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    clf = RandomForestClassifier(
        n_estimators=10, max_depth=2, oob_score=False, n_jobs=-1, verbose=1)
    # clf = NearestCentroid()
    clf.fit(X, y)
    # print(clf.feature_importances_)
    # print(clf.oob_decision_function_)
    # print(clf.oob_score_)
    predictions = clf.predict(test_features)
    errors = abs(predictions - test_labels)
    print("M A E: ", np.mean(errors))
    print(np.count_nonzero(errors), len(test_labels))
    print("Accuracy:", np.count_nonzero(errors)/len(test_labels))
"""
