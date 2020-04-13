from read_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

from ecgdetectors import Detectors
from hrv import HRV

def hr_into_data(p_pulsacion, data):
    hr = np.full(len(data), 0.0)
    hr2 = hrv_class.HR(rr_samples=p_pulsacion)
    index = range(len(p_pulsacion)-1)
    for i in index:
        if i == index[-1]:
            np.put(hr, range(p_pulsacion[i],len(hr)-1), hr2[i])
        else:
            np.put(hr, range(p_pulsacion[i], p_pulsacion[i+1]), hr2[i])
        # for j in range(p_pulsacion[i-1], p_pulsacion[i]):
        #     hr[j] = pulsaciones

    return hr

if __name__ == '__main__':
    detector = Detectors(700)
    hrv_class = HRV(700)
    data = execute()
    print(data.shape)
    X = data[:, :2]  # 16 features
    ecg = data[:, 1]  # Extraemos ECG
    r_peaks = detector.pan_tompkins_detector(ecg)
    hr = hr_into_data(r_peaks, X)
    test=hrv_class.RMSSD(r_peaks)
    # plt.figure()
    # plt.plot(ecg)
    # # plt.plot(r_peaks)
    # plt.show()
    y = data[:, 2]
    X = np.column_stack((X, hr))
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
        n_estimators=100, max_depth=5, oob_score=True)
    clf.fit(X, y)
    print(clf.feature_importances_)
    # print(clf.oob_decision_function_)
    print(clf.oob_score_)
    predictions = clf.predict_proba(test_features)
    errors = abs(predictions - test_labels)
    print("M A E: ", np.mean(errors))
    print(np.count_nonzero(errors), len(test_labels))
    print("Accuracy:", np.count_nonzero(errors)/len(test_labels))
