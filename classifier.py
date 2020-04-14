from read_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from ecgdetectors import Detectors
from hrv import HRV

# Numero de muestras por segundo
mps = 700


def hr_processing(r_peaks, data):
    hr = np.full(len(data), 0.0)
    hr2 = hrv_class.HR(rr_samples=r_peaks)
    index = range(len(r_peaks)-1)
    for i in index:
        if i == index[-1]:
            np.put(hr, range(r_peaks[i], len(hr)-1), hr2[i])
        else:
            np.put(hr, range(r_peaks[i], r_peaks[i+1]), hr2[i])
        # for j in range(p_pulsacion[i-1], p_pulsacion[i]):
        #     hr[j] = pulsaciones

    return hr


def save_dataset(dataset):
    """
    Funcion que permite almacenar un dataset pasado como parámetro
    """
    os.chdir("/home/fedeparg/Stress-detection/")
    with open('dataset_hr.pkl', 'wb') as output:
        print("Dumping dataset...", end=' ')
        pickle.dump(dataset, output)
        print("Done!")


def load_dataset():
    """
    Funcion que permite cargar un dataset pasado como parámetro
    """
    with open('dataset_hr.pkl', 'rb') as file:
        print("Reading dataset...", end=' ')
        data = pickle.load(file, encoding='latin1')
        print("Done!")
    return data


if __name__ == '__main__':
    """
    detector = Detectors(mps)
    hrv_class = HRV(mps)
    data = execute()
    print(data.shape)
    X = data[:, :2]  # 16 features
    y = data[:, 2]
    ecg = data[:, 1]  # Extraemos ECG

    # r_peaks son los puntos temporales en los que se detecta un latido
    r_peaks = detector.pan_tompkins_detector(ecg)
    # hr es el ritmo cardiaco calculado mediante los r_peaks
    hr = hr_processing(r_peaks, X)

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
    save_dataset(new_dataset)"""
    data_reloaded = load_dataset()
    print("This is da data" + str(data_reloaded.shape))
    X = data_reloaded[:,:3]
    y = data_reloaded[:,3]
    print(X.shape)
    print(y.shape)
    print(y)
    train_features, test_features, train_labels, test_labels = train_test_split(X, y,
                                                                                test_size=0.25)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, oob_score=True, n_jobs=4, verbose=1)
    clf.fit(X, y)
    print(clf.feature_importances_)
    # print(clf.oob_decision_function_)
    print(clf.oob_score_)
    predictions = clf.predict(test_features)
    errors = abs(predictions - test_labels)
    print("M A E: ", np.mean(errors))
    print(np.count_nonzero(errors), len(test_labels))
    print("Accuracy:", np.count_nonzero(errors)/len(test_labels))
