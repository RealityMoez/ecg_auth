import os
import numpy as np
import pywt
import tkinter as tk
import pickle
from pantompkins import Pan_tompkins
from tkinter import filedialog
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.signal import argrelextrema
from spectrum import aryule

# ----------------- Main Program ---------------------
def main():
    # each file in preprocessed_data_path is a csv file containing preprocessed ECG data named as 'patientXXX_yyy.csv'
    # where XXX is the patient number and yyy is any other relevant information, and each patient has multiple csv files
    # patientXXX is identified as the label for each patient for classification
    subjects = ["patient007", "patient017", "patient023", "patient025"]
    
    extracted_features_path = 'data/extracted_features'
    os.makedirs(extracted_features_path, exist_ok=True)

    extracted_features_files = [file for file in os.listdir(extracted_features_path) if file.endswith('.csv')]

    X_fiducial = []
    X_non_fiducial = []
    y_fiducial = []
    y_non_fiducial = []

    for extracted_features_file in extracted_features_files:
        patient_number = extracted_features_file.split('_')[0]
        feature_type = extracted_features_file.split('_')[1]
        extracted_features_file_path = os.path.join(extracted_features_path, extracted_features_file)

        # Load extracted features
        extracted_features = load_extracted_features(extracted_features_file_path)

        if feature_type == "fiducial":
            X_fiducial.extend(extracted_features)
            y_fiducial.extend([patient_number] * len(extracted_features))
        elif feature_type == "non-fiducial":
            X_non_fiducial.extend(extracted_features)
            y_non_fiducial.extend([patient_number] * len(extracted_features))

    # Convert lists to numpy arrays
    X_fiducial = [np.array(x).reshape(-1) if np.isscalar(x) else x for x in X_fiducial]
    X_non_fiducial = [np.array(x).reshape(-1) if np.isscalar(x) else x for x in X_non_fiducial]

    # Pad sequences to have the same length
    max_len_fiducial = max(len(x) for x in X_fiducial)
    max_len_non_fiducial = max(len(x) for x in X_non_fiducial)
    X_fiducial = np.array([np.pad(x, (0, max_len_fiducial - len(x)), 'constant') for x in X_fiducial])
    X_non_fiducial = np.array([np.pad(x, (0, max_len_non_fiducial - len(x)), 'constant') for x in X_non_fiducial])

    y_fiducial = np.array(y_fiducial)
    y_non_fiducial = np.array(y_non_fiducial)
    
    print(f"\nGetting fiducial classifier..\n")
    classifier_fiducial, accuracy_fiducial = trained_classifier(X_fiducial, y_fiducial, "fiducial")
    print(f"Getting non-fiducial classifier..\n")
    classifier_non_fiducial, accuracy_non_fiducial = trained_classifier(X_non_fiducial, y_non_fiducial, "non-fiducial")

    # Print results and show identification result in UI
    print(f"Accuracy using fiducial features: {accuracy_fiducial*100:.2f}%")
    print(f"Accuracy using non-fiducial features: {accuracy_non_fiducial*100:.2f}%")
    show_identification_result("Trained", classifier_fiducial, classifier_non_fiducial, max_len_fiducial, max_len_non_fiducial, subjects)

def load_preprocessed_data(file_path):
    return np.loadtxt(file_path, delimiter=",")

def save_extracted_features(features, save_path):
    np.savetxt(save_path, features, delimiter=",")

def load_extracted_features(file_path):
    return np.loadtxt(file_path, delimiter=",")

def show_identification_result(message, classifier_fiducial, classifier_non_fiducial, max_len_fiducial, max_len_non_fiducial, subjects):
        root = tk.Tk()
        root.title("ECG Authentication Interface")
        root.geometry("350x120")

        label = tk.Label(root, text=f"RESULT: {message}")
        label.pack(padx=20, pady=20)

        load_signal_button = tk.Button(root, text="Load New Signal", command=lambda: load_new_signal(root, classifier_fiducial, classifier_non_fiducial, max_len_fiducial, max_len_non_fiducial, subjects))
        load_signal_button.pack(padx=20, pady=20)

        root.mainloop()
    
def load_new_signal(last_root, classifier_fiducial, classifier_non_fiducial, max_len_fiducial, max_len_non_fiducial, subjects):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select signal file")
    
    if not file_path:
        return

    # Load preprocessed data
    preprocessed_ecg_data = load_preprocessed_data(file_path)

    # Extract features
    fiducial_features, non_fiducial_features = extract_features(preprocessed_ecg_data)
    
    # Pad fiducial_features and non_fiducial_features to the maximum lengths
    fiducial_features = np.array([np.pad(x, (0, max_len_fiducial - len(x)), 'constant') for x in fiducial_features])
    non_fiducial_features = np.array([np.pad(x, (0, max_len_non_fiducial - len(x)), 'constant') for x in non_fiducial_features])
    
    if len(fiducial_features.shape) == 1:
        fiducial_features = fiducial_features.reshape(-1, 1)

    # Authenticate the signal using the classifiers
    if len(fiducial_features) != 0:
        y_pred_fiducial = classifier_fiducial.predict(fiducial_features)
        print("\nfiducial Predictions: ", y_pred_fiducial)
    else:
        print("\n Fiducial features are empty.")
        y_pred_fiducial = "No Fiducial features."
    
    y_pred_non_fiducial = classifier_non_fiducial.predict(non_fiducial_features)
    print("\nnon_fiducial Predictions: ", y_pred_non_fiducial)
    
    # Check if all predicted non-fiducial values are the same
    if are_all_values_equal(y_pred_non_fiducial):
        identified_subject = y_pred_non_fiducial[0]  # Get the single predicted subject
        identified = True
    else:
        identified = False
        identified_subject = None

    # Show the authentication result
    if identified:
        result = f"Signal identified as ({identified_subject})"
    else:
        result = "Signal not identified"
    last_root.destroy()
    show_identification_result(result, classifier_fiducial, classifier_non_fiducial, max_len_fiducial, max_len_non_fiducial, subjects)

def are_all_values_equal(arr):
    return all(element == arr[0] for element in arr)

# ----------------- Classification ---------------------
def trained_classifier(X, y, feature_type):
    
    classifier_path = f"linear_svc_{feature_type}_classifier_2.pkl"
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if not os.path.exists(classifier_path):
        print("Model file does not exist.\nTraining...")

        # Use SMOTE to balance the dataset
        print("Balancing the dataset...")
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train) # type: ignore

        # Set the parameters for GridSearchCV
        linear_svc_params = {'C': [0.1, 0.2, 1, 10]}
        
        # Perform GridSearchCV for LinearSVC
        print(f"Trying  ({feature_type}) LinearSVC...")
        linear_svc = LinearSVC()
        print(f"Trying ({feature_type}) LinearSVC GridSearchCV...")
        linear_svc_grid = GridSearchCV(linear_svc, linear_svc_params, scoring='accuracy', cv=5)
        linear_svc_grid.fit(X_train, y_train) # type: ignore
        
        # Find the best estimator
        linear_svc_classifier = linear_svc_grid.best_estimator_
    
        with open(classifier_path, 'wb') as f:
            pickle.dump(linear_svc_classifier, f)
    else:
        print("Model file exists.\nLoading...")
        # Load the saved model from the file
        with open(classifier_path, 'rb') as f:
            linear_svc_classifier = pickle.load(f)
            
    print(f"\nEvaluating LinearSVC ({feature_type}) classifier...\n")
    # Evaluate the accuracy of the best LinearSVC estimator
    linear_svc_accuracy = test_classifier(linear_svc_classifier, X_test, y_test)

    return linear_svc_classifier, linear_svc_accuracy

def test_classifier(classifier, X_test, y):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

# ----------------- Feature Extraction ---------------------
def extract_fiducial_features(preprocessed_ecg_data, sampling_rate=2000.0, search_window=20):
    if preprocessed_ecg_data.ndim > 1:
        preprocessed_ecg_data = preprocessed_ecg_data.flatten()

    # Use the Pan-Tompkins algorithm for R-peak detection
    integrated_signal = Pan_tompkins(preprocessed_ecg_data, sampling_rate).fit()
    rpeaks = argrelextrema(integrated_signal, np.greater)[0]

    qpeaks = []
    speaks = []
    ppeaks = []
    tpeaks = []
    q_starts = []
    q_ends = []
    p_starts = []
    p_ends = []
    t_starts = []
    t_ends = []

    for rpeak in rpeaks:
        rpeak = int(rpeak)
        search_start = max(0, rpeak - search_window)
        search_end = min(len(preprocessed_ecg_data) - 1, rpeak + search_window)

        # Find local minimum before and after R peak for Q and S points
        q_candidates = preprocessed_ecg_data[search_start:rpeak]
        s_candidates = preprocessed_ecg_data[rpeak + 1:search_end]

        if len(q_candidates) > 0:
            q_extrema = argrelextrema(q_candidates, np.less)[0]
            if len(q_extrema) > 0:
                qpeaks.append(search_start + q_extrema[-1])

        if len(s_candidates) > 0:
            s_extrema = argrelextrema(s_candidates, np.less)[0]
            if len(s_extrema) > 0:
                speaks.append(rpeak + 1 + s_extrema[-1])

        # Find Q start and Q end points
        q_start = search_start
        q_end = search_end

        # Find local maximum before R peak for P wave
        p_search_start = max(0, rpeak - 2 * search_window)
        p_candidates = preprocessed_ecg_data[p_search_start:search_start]

        if len(p_candidates) > 0:
            p_extrema = argrelextrema(p_candidates, np.greater)[0]
            if len(p_extrema) > 0:
                ppeaks.append(p_search_start + p_extrema[-1])

        # Find P start and P end points
        p_start = p_search_start
        p_end = search_start

        # Find local maximum after R peak for T wave
        t_search_end = min(len(preprocessed_ecg_data) - 1, rpeak + 2 * search_window)
        t_candidates = preprocessed_ecg_data[search_end:t_search_end]

        if len(t_candidates) > 0:
            t_extrema = argrelextrema(t_candidates, np.greater)[0]
            if len(t_extrema) > 0:
                tpeaks.append(search_end + t_extrema[-1])

        # Find T start and T end points
        t_start = search_end
        t_end = t_search_end

        # Append start and end points to respective lists
        q_starts.append(q_start)
        q_ends.append(q_end)
        p_starts.append(p_start)
        p_ends.append(p_end)
        t_starts.append(t_start)
        t_ends.append(t_end)

    fiducial_points = {
        'qpeaks': np.array(qpeaks),
        'rpeaks': np.array(rpeaks),
        'speaks': np.array(speaks),
        'ppeaks': np.array(ppeaks),
        'tpeaks': np.array(tpeaks),
        'q_starts': np.array(q_starts),
        'q_ends': np.array(q_ends),
        'p_starts': np.array(p_starts),
        'p_ends': np.array(p_ends),
        't_starts': np.array(t_starts),
        't_ends': np.array(t_ends)
    }

    return fiducial_points

def extract_wavelet_features(preprocessed_ecg_data, wavelet='db4', level=4):
    if preprocessed_ecg_data.ndim > 1:
        preprocessed_ecg_data = preprocessed_ecg_data.flatten()
    coeffs = pywt.wavedec(preprocessed_ecg_data, wavelet, level=level)
    concat_coeffs = np.concatenate(coeffs)
    return concat_coeffs

def extract_ar_coefficients(preprocessed_ecg_data, order=12):
    if preprocessed_ecg_data.ndim == 1:
        preprocessed_ecg_data = preprocessed_ecg_data.reshape(1, -1)

    ar_coeffs_list = []
    for heartbeat in preprocessed_ecg_data:
        if heartbeat.ndim > 1:
            heartbeat = heartbeat.flatten()
        ar_coeffs = aryule(heartbeat, order=order)[0]  # Get only the AR coefficients
        ar_coeffs_list.append(ar_coeffs)
    return np.array(ar_coeffs_list)

def extract_features(preprocessed_ecg_data, sampling_rate=1000.0, wavelet='db4', level=4, ar_order=12, search_window=20):
    if preprocessed_ecg_data.ndim == 1:
        preprocessed_ecg_data = preprocessed_ecg_data.reshape(1, -1)

    fiducial_features_list = []
    non_fiducial_features_list = []

    wavelet_features_list = []
    ar_features_list = []

    # Compute wavelet features and AR features for all heartbeats first
    for heartbeat in preprocessed_ecg_data:
        if heartbeat.ndim > 1:
            heartbeat = heartbeat.flatten()

        wavelet_features = extract_wavelet_features(heartbeat, wavelet, level)
        ar_features = extract_ar_coefficients(heartbeat, ar_order)

        wavelet_features_list.append(wavelet_features)
        ar_features_list.append(ar_features)

    # Find the maximum lengths of wavelet features and AR features
    max_wavelet_len = max(len(wavelet_features) for wavelet_features in wavelet_features_list)
    max_ar_len = max(len(ar_features) for ar_features in ar_features_list)

    # Extract features and pad wavelet features and AR features to the maximum lengths
    for i, heartbeat in enumerate(preprocessed_ecg_data):
        if heartbeat.ndim > 1:
            heartbeat = heartbeat.flatten()

        wavelet_features = wavelet_features_list[i]
        ar_features = ar_features_list[i]

        # Pad wavelet_features and ar_features to the maximum lengths
        wavelet_features = np.pad(wavelet_features, (0, max_wavelet_len - len(wavelet_features)), 'constant')[:max_wavelet_len]
        ar_features = np.pad(ar_features, (0, max_ar_len - len(ar_features)), 'constant')[:max_ar_len].flatten()

        # Ensure that wavelet_features and ar_features are 1-dimensional arrays
        wavelet_features = wavelet_features.flatten()
        ar_features = ar_features.flatten()

        # Concatenate wavelet_features and ar_features
        non_fiducial_features = np.concatenate([wavelet_features, ar_features])

        # Ensure that non_fiducial_features is a 1-dimensional array
        non_fiducial_features = non_fiducial_features.flatten()

        non_fiducial_features_list.append(non_fiducial_features)
        
        fiducial_points = extract_fiducial_features(heartbeat, sampling_rate, search_window)

        # Find the minimum length across all arrays
        min_len = min(len(arr) for arr in fiducial_points.values())

        # Truncate arrays to the minimum length
        truncated_points = {key: arr[:min_len] for key, arr in fiducial_points.items()}

        # Aggregate fiducial features
        fiducial_features = np.column_stack([arr for arr in truncated_points.values()])

        fiducial_features_list.append(fiducial_features)

    fiducial_features_aggregated = np.vstack(fiducial_features_list)
    non_fiducial_features_aggregated = np.vstack(non_fiducial_features_list)

    return fiducial_features_aggregated, non_fiducial_features_aggregated

if __name__ == "__main__":
    main()