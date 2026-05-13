# 1D-CNN regressor to extract structural information on Ag-Co nanoscale bimetallic system

This repository contains a complete workflow for generating simulated X-ray diffraction (XRD) datasets, training a convolutional neural network (CNN), and predicting nanoparticle properties from diffraction patterns.

The model jointly predicts **three labels** from a single diffraction pattern:

- **Configuration index** (`config`)
- **Nanoparticle size** (`N_atoms`)
- **Stoichiometry** (`Ag_fraction`)

---

# Repository Content

| File | Description |
|---|---|
| `1_datagen.py` | Generates simulated diffraction datasets from `.xyz` structures |
| `2_trainer_3labels.py` | Trains the CNN regression model on generated datasets |
| `3_Predict_3labels_nm.py` | Predicts nanoparticle properties from experimental diffraction patterns |
| `model3.h5` | Pre-trained Keras/TensorFlow model |
| `scaler3.pkl` | Saved `MinMaxScaler` used for label normalization |

---

# Overall Workflow

## 1. Dataset Generation

`1_datagen.py`:

- Reads nanoparticle `.xyz` structures
- Runs *Debussy* externally for diffractogram simulation
- Processes diffractograms into a standardized format:
    - Q-range: `20–75 nm⁻¹`
    - 1000 interpolated points
    - AUC normalization
- Adds synthetic Poisson noise augmentation
- Produces NumPy matrices for ML training:
    - `matrix_patterns.npy`
    - `matrix_configs.npy`
    - `matrix_atoms_TOT.npy`
    - `matrix_atoms_Ag.npy`

---

## 2. CNN Training

`2_trainer_3labels.py`:

- Loads generated diffraction matrices
- Splits data into train/test sets
- Normalizes labels with `MinMaxScaler`
- Trains a 1D-CNN regression model using TensorFlow/Keras
- Predicts all three labels simultaneously

Outputs include:

- `model3.h5`
- scaler pickle file
- learning curves
- prediction CSVs
- scatter plots
- evaluation metrics (`MAE`, `RMSE`, `R²`)

---

## 3. Experimental Prediction

`3_Predict_3labels_nm.py`:

- Loads experimental diffraction patterns
- Processes diffractograms into the standardized format (see point 1)
- Applies the trained model
- Decodes predictions using the saved scaler

Predictions are printed on screen as well as exported as CSV tables.

---

# Requirements

## Python

Tested with:

```text
Python 3.9
```

## Main Python Dependencies

```bash
numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

---

# External Dependencies

Dataset generation additionally requires the external diffraction simulation suite of programs:

```text
Debussy
```
Can be found at [https://debyeusersystem.github.io/](URL).

---

# Usage

## 1. Generate Training Data

```bash
python 1_datagen.py <database_name>
```

Example:

```bash
python 1_datagen.py db_test
```

This creates:

```text
DATA/db_test/
```

containing processed datasets and training matrices.

---

## 2. Train the Model

```bash
python 2_trainer_3labels.py <results_name>
```

Example:

```bash
python 2_trainer_3labels.py run01
```

Outputs are saved in:

```text
results/run01/
```

including:

```text
model3.h5
scaler3_conf_size_st_0-1.pkl
```

---

## 3. Predict Experimental Patterns

```bash
python 3_Predict_3labels_nm.py <patterns_folder> <model_folder>
```

Example:

```bash
python 3_Predict_3labels_nm.py ./exp_patterns ./results/run01
```

Optional arguments:

| Argument | Description |
|---|---|
| `-l FLOAT` | X-ray wavelength in Å (default: `0.56`) |
| `--suff TEXT` | Suffix added to output filenames |

Example:

```bash
python 3_Predict_3labels_nm.py ./exp_patterns ./results/run01 -l 0.774904 --suff test01
```

Prediction CSV files are saved in:

```text
results/PREDICTIONS/
```

---

# Input Pattern Format

Prediction inputs must contain two columns:

```text
Q(nm^-1)    Intensity
```

Patterns are expected to already be expressed in reciprocal-space Q units (`nm⁻¹`).

---

# Notes

- All diffraction patterns are normalized to an area-under-the-curve (`AUC`) of `1000`.
- Training uses synthetic Poisson noise augmentation to improve robustness against experimental noise.
- The prediction pipeline automatically handles patterns with narrower Q-ranges via edge padding and interpolation.
