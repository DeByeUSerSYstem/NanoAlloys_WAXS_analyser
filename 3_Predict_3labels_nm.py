#!/usr/bin/python3.9

import os
import sys
import time
import pickle
import argparse
import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
# silence tensorflow I, E and W warnings
# or any of {'0', '1', '2', '3'} – 3 silences errors too
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from keras.models import load_model


def normalise_auc(x, y, val):
    """Normalises the area-under-the-curve (auc) of a pattern to some value (val)
    :param x, y: respective axix of the pattern
    :param val: scalar, desired value for the auc
    :new_y: y axis properly rescaled to satisfy the auc requirement """
    auc = np.trapz(x=x, y=y)
    return y * (val / auc)


def parse_args():
    """
    Parses input file(s) from command line
    """
    parser = argparse.ArgumentParser(prog='PA_Predict_3lbls.py',
                    description='''Process PA experimental data in nm-1.''')
    parser.add_argument('folder_path',
                        help='Path to folder with patterns to be predicted.')
    parser.add_argument('model',
                        help='Path to the predictive model of choice folder.')
    parser.add_argument('--suff', type=str, default='',
                        help='Possible suffix for prediction csv file name.')
    parser.add_argument('-t', type=str, choices=['xy', 'smooth'], default='xy',
                        help='Whether to predict original patterns (xy) or smoothed ones.')
    parser.add_argument('-l', type=float, default=0.56,
                        help='Collection wavelength, in Å.  default=0.56Å')

    arg = parser.parse_args()
    return arg


# --------------------------------------------
#  PGM START
# --------------------------------------------
a = time.time()  # time check


# --------------------------------------------
#  PARAMENTERS, FOLDERS AND MODEL DEFINITION
# --------------------------------------------

# parse command line inputs
args = parse_args()
# patterns_dir = Path('../PA_data/PA_xy')
# patterns_dir = Path('../PA_data/PA_xy/smoothed_to_predict')
# patterns_dir = Path('../new_PA_data/AgCore_CoGrowth_xy')
# patterns_dir = Path('../new_PA_data/CoCore_AgGrowth_xy')

patterns_dir = Path(args.folder_path)
# print(patterns_dir.iterdir(), sep="\n")
# feedback = input('\nIs ok? ')
# if feedback.lower() != 'y':
#     sys.exit(5)

model_folder_path = Path(args.model)
model_path = list(model_folder_path.glob('*.h5'))[0]

wavelength = args.l
data_type = args.t
suffix = args.suff if len(args.suff) == 0 else '_' + args.suff

# set scaler
scaler_path = list(model_folder_path.glob('*scaler*.pkl'))[0]
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# load model
model = load_model(model_path, compile=True)

# Set parameters
project_dir = Path.cwd()
n_points = 1000

qmin = 2.00*10
qMAX = 7.50*10

# process and predict Pascal's experimental data
file_names = []
diffractograms = []
for diffr in sorted(patterns_dir.glob('*.' + data_type)):

    # do not include "substrate" patterns
    if diffr.stem.startswith('s'):
        continue
    file_names.append(diffr.stem)

    # load Pascal's experimental data: already in q (nm-1)
    q_exp, y_exp = np.loadtxt(diffr, comments='#', unpack=True)

    # convert 2θ to Q
    # q_exp = (4 * np.pi / wavelength) * np.sin(np.radians(tt_exp / 2)) * 10

    # --- select desired Q values
    # if experimental Q-range is –same or wider– than the std settings:
    if q_exp[0] <= qmin and q_exp[-1] >= qMAX:
        idx = np.where((q_exp >= qmin) & (q_exp <= qMAX))

        # spline interpolation of test data in the just selected Q-range
        x = np.linspace(qmin, qMAX, n_points)  # q step = 0.055(long)  or  0.0456(short)
        y = np.interp(x, q_exp[idx], y_exp[idx])

        # normalize plot to an area under the curve (auc) = 1000
        y_final = normalise_auc(x, y, 1000)

    # if the pattern Q-range is –narrower– than the std settings:
    else:
        qmin_exp = q_exp[0] if q_exp[0] >= qmin else qmin
        qMAX_exp = q_exp[-1] if q_exp[-1] <= qMAX else qMAX
        idx = np.where((q_exp >= qmin_exp) & (q_exp <= qMAX_exp))

        # spline interpolation of test data in the just selected Q-range
        x_short = np.linspace(qmin_exp, qMAX_exp, n_points)
        y_short = np.interp(x_short, q_exp[idx], y_exp[idx])

        # get average of the 5 initiala and final values of the diffractogram
        # to use them as filler where the pattern is shorter than the std qrange (20-75)
        pre_pad_val = np.mean(y_short[:5])
        post_pad_val = np.mean(y_short[-5:])
        tmp_y = np.concatenate((pre_pad_val, y_short, post_pad_val), axis=None)

        # spline interpolation between std-long q_axis (x) and "padded" y
        tmp_q = np.concatenate((qmin_exp, x_short, qMAX_exp), axis=None)
        x = np.linspace(qmin, qMAX, n_points)  # q step = 0.055(long)  or  0.0456(short)
        y = np.interp(x, tmp_q, tmp_y)

        # normalize plot to an area under the curve (auc) = 1000
        y_final = normalise_auc(x, y, 1000)

    assert len(y_final) == 1000

    # # show normalised pattern against original experimental one
    # plt.plot(q_exp, y_exp, 'b', marker='.', label='original')
    # plt.plot(x, y_final, 'r', marker='.', label='processed')
    # plt.legend(), plt.show(), plt.close()

    # save normalised plot as .xy
    norm_pattern_dir = patterns_dir.joinpath('normalised_during_prediction')
    norm_pattern_path = norm_pattern_dir.joinpath(f'{diffr.stem}_norm{suffix}.{data_type}')
    if not norm_pattern_dir.is_dir():
        norm_pattern_dir.mkdir()
    norm_pattern = np.hstack((x.reshape(-1, 1), y_final.reshape(-1, 1)))
    np.savetxt(norm_pattern_path, norm_pattern, fmt='%.5f')

    # collect diffractogram
    diffractograms.append(y_final)


# generate and decode predictions of test data
encoded_predictions = model.predict(np.asanyarray(diffractograms))
predictions = scaler.inverse_transform(encoded_predictions)


# --------------------------------------------
#  PLOT AND PRINT RESULTS
# --------------------------------------------

summary = pd.DataFrame()
summary['File_name'] = file_names
# summary = pd.read_csv(patterns_dir.joinpath('PA_reported_results.csv'))
# filenames = pd.Series(file_names)
# if not summary['File_name'].eq(filenames).all():
#     print('ATTENTION: file names mismatch!')

# summary['encoded_CSidx'] = encoded_predictions[:, 0]
summary['prediction_CSidx'] = predictions[:, 0]
# summary['encoded_size'] = encoded_predictions[:, 1]
summary['prediction_size'] = predictions[:, 1]
# summary['encoded_Stoich'] = encoded_predictions[:, 2]
summary['prediction_Stoich'] = predictions[:, 2]

print()
print(summary)
# Save csv
summary_fname = f'PRED3_{patterns_dir.stem}_{model_path.parts[-2]}_{data_type}{suffix}.csv'
summary_path = project_dir / 'results' / 'PREDICTIONS' / summary_fname
# summary_path = project_dir / 'results' / 'smooth_prediction_tests' / summary_fname
summary.to_csv(summary_path)
