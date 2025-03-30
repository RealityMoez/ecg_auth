
# ECG Authentication System

A biometric authentication system using Electrocardiogram (ECG) signals. It leverages the unique characteristics of an individual's heartbeat pattern to verify their identity. The system uses a combination of fiducial and non-fiducial features extracted from ECG signals to provide authentication.

*(For more information about ECG signals, refer to the included `what_is_ecg.pdf` document)*

## Features

- **ECG Signal Processing**: Implements preprocessing techniques including bandpass filtering, baseline correction, and segmentation
- **Feature Extraction**:
  - **Fiducial Features**: Extracts features based on specific points in the ECG waveform (P, Q, R, S, T waves)
  - **Non-Fiducial Features**: Extracts wavelet and autoregressive coefficients
- **Pan-Tompkins Algorithm**: QRS complex detection for accurate heartbeat segmentation
- **Machine Learning Classification**: Uses Linear Support Vector Classification (LinearSVC)
- **Interface**: Uses simple python `tkinter` GUI. 

## How It Works

1. **Signal Acquisition**: ECG signals are loaded from files
2. **Preprocessing**: Signals are filtered and normalized
3. **Feature Extraction**: Both fiducial and non-fiducial features are extracted
4. **Classification**: LinearSVC models determine if the ECG belongs to a registered user
5. **Authentication Decision**: The system provides an authentication result based on model predictions

## Project Architecture

1. **Preprocessing Module** (`preprocessing.py`): 
   - Loads raw ECG data
   - Applies filtering and baseline correction
   - Segments data into individual heartbeats

2. **Pan-Tompkins Implementation** (`pantompkins.py`):
   - Dedicated algorithm for QRS complex detection
   - Uses bandpass filtering, derivative, squaring, and moving window integration

3. **Main Application** (`aio.py`):
   - Feature extraction from ECG signals
   - Classification using machine learning
   - User interface for authentication

## Get Started

### Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install numpy scipy pywt scikit-learn imbalanced-learn spectrum biosppy wfdb
   ```

### Usage

1. Run the main application:
   ```
   python src/aio.py
   ```
2. From the GUI, load an ECG signal file to test the authentication result.

## Dataset

The system is designed to work with ECG signals stored as CSV files. Each file named in the format `patientXXX_yyy.csv` where XXX is the patient number and yyy is any additional information.

## Model Training

The system automatically trains LinearSVC models for both fiducial and non-fiducial features if pre-trained models are not available. The trained models are saved as pickle files for future use.

## Performance

The system evaluates authentication accuracy using fiducial and non-fiducial features separately. Non-fiducial features typically provide more accurate authentication results.