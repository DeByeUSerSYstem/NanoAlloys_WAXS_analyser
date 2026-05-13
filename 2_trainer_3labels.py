#!/usr/bin/python3.9

import os
import sys
import time
import shutil
import pickle
import argparse
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# silence tensorflow I, E and W warnings
# any of {'0', '1', '2', '3'} - 3 silences errors too
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, UpSampling1D, Input,
                                     Dense, Dropout, SpatialDropout1D, Flatten)
from tensorflow.keras.backend import clear_session
from tensorflow.compat.v1 import reset_default_graph
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def plot_loss(history, dirname):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle(f'{dirname[7:]} — Learning curve')

    # Loss subplot (ax1)
    ax1.plot(history['loss'], marker='.', label='train loss')
    ax1.plot(history['val_loss'], marker='.', label=' val loss')
    ax1.set_ylim(0, ax1.set_ylim()[1])
    # ax1.set_ylim(0, 0.015)
    ax1.set_ylabel('loss [MAE]')
    ax1.legend(loc='upper right')
    ax1.grid(visible=True, color='0.90')

    # Accuracy subplot (ax2)
    ax2.plot(history['mse'], marker='.', label='train mse')
    ax2.plot(history['val_mse'], marker='.', label=' val mse')
    ax2.set_ylim(0, ax2.set_ylim()[1])
    # ax2.set_ylim(0, 0.015)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mse')
    ax2.legend(loc='upper right')
    ax2.grid(visible=True, color='0.90')

    fig.tight_layout()
    plt.savefig(os.path.join(result_dir, 'learning_curve'), dpi=300)
    # plt.show()
    plt.close()


def do_result_df(data_names, norm_labels, orig_labels, m_pred, rev_pred,
                 df_name, out_dir):

    df = pd.DataFrame()
    df['file_id'] = data_names
    df['normed_value'] = norm_labels.values
    df['model_prediction'] = m_pred.values
    df['original_value'] = orig_labels.values
    df['reverted_prediction'] = rev_pred.values

    df['norm_diff'] = abs(df['normed_value'] - df['model_prediction'])
    df['orig_diff'] = abs(df['original_value'] - df['reverted_prediction'])

    # gather only "errors": predictions more that 10% away from real vale
    threshold = 0.1
    errors = df.loc[df['norm_diff'] >= threshold]  # on [0-1] scaled values

    df.to_csv(os.path.join(out_dir, f'prediction_results_{df_name}.csv'),
              index=False)
    errors.to_csv(os.path.join(out_dir, f'prediction_errorish_{df_name}.csv'),
                  index=False)

    return df


def scatter_plot(x, y, MAE, mse, r2, norm_Rmse, dirname, flag):
    # define colors and trasparences
    if flag == 'CSid':
        c_dash, a_dash = 'navy', 0.4
        c_scat, a_scat = '#17becf', 0.4
    elif flag == 'size':
        c_dash, a_dash = 'crimson', 0.6
        c_scat, a_scat = 'salmon', 0.3
    elif flag == 'stoich':
        c_dash, a_dash = 'darkgreen', 0.5
        c_scat, a_scat = 'yellowgreen', 0.4

    # define limits
    high = x.max() if x.max() > y.max() else y.max()
    low  = x.min() if x.min() < y.min() else y.min()
    x1, y1 = [low, high], [low, high]
    title = os.path.basename(dirname)
    text_on_plot = f'MAE = {MAE:.3f} \n√mse = {sqrt(mse):.3f} \nR$^{2}$ = {r2:.3f} \n√mse/σ = {norm_Rmse:.2f}'

    plt.title(f'{title[:]} {flag}')
    plt.plot(x1, y1, ls='-.', dashes=(5, 4, 1, 4), lw=0.7, c=c_dash, alpha=a_dash)
    plt.scatter(x, y, marker='.', s=20, lw=0.3, c=c_scat, alpha=a_scat)
    plt.axis('square')
    plt.xlabel('true values')
    plt.ylabel('predicted values')
    plt.text(x=low, y=high, ha='left', va='top', s=text_on_plot)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(dirname, f'scatter_plot_{flag}'), dpi=300)
    # plt.show()
    plt.close()


# --------------------------------------------
#  PGM START
# --------------------------------------------
print(time.asctime())
start = time.time()  # time check

# process arguments
result_dir_name = sys.argv[1]
k1 = 30
k2 = 25
k3 = 20
k4 = 20

# Clear Keras and TF session, if run previously
clear_session()
reset_default_graph()

# folders & files definition
project_dir = os.getcwd()
data_dir = os.path.join(project_dir, 'DATA', 'db_short')
matrix_dir = os.path.join(data_dir, 'matrices')
result_dir = os.path.join(project_dir, 'results', result_dir_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# NN parameters
batch_size = 32
patience = 30
epochs = 500
lr = 0.0001


# --------------------------------------------
#  LOAD DATA AND ENCODE LABELS
# --------------------------------------------

# Load data & names
print('=== Loading data ===')
# diffractograms
X = np.load(os.path.join(matrix_dir, 'CS_matrix_patterns.npy'))

# file names
z = np.load(os.path.join(matrix_dir, 'CS_matrix_names.npy'))

# labels
conf = np.load(os.path.join(matrix_dir, 'CS_matrix_configs.npy'))
nTOT = np.load(os.path.join(matrix_dir, 'CS_matrix_atoms_TOT.npy'))
nAg  = np.load(os.path.join(matrix_dir, 'CS_matrix_atoms_Ag.npy'))
fraz_Ag = nAg / nTOT

# join all labels in a dataframe
y = pd.DataFrame({
                    'config': conf,
                    'N_atoms': nTOT,
                    'stoichiometry': fraz_Ag
                })

# define TRAIN and test sub-set
X_TRAIN, X_test, y_TRAIN, y_test, z_TRAIN, z_test = train_test_split(
    X, y, z, test_size=0.2, random_state=11)  # rdm = 11 normally
print(f'data splitted: train:{X_TRAIN.shape[0]}, test:{X_test.shape[0]}\n')

# make a copy to conserve original labels
yn_TRAIN = y_TRAIN.copy()
yn_test = y_test.copy()

# normalise the copied labels in [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(y_TRAIN[y_TRAIN.columns])
yn_TRAIN[y_TRAIN.columns] = scaler.transform(yn_TRAIN[y_TRAIN.columns])
yn_test[yn_test.columns] = scaler.transform(yn_test[yn_test.columns])

# save scaler for future use in prediction phase
with open(os.path.join(result_dir, 'scaler3_conf_size_st_0-1.pkl'), 'wb') as f:
    pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)


# def NN
print('=== Training ===')
model = Sequential(name='MAE3_net')

# 1
model.add(Conv1D(filters=32, kernel_size=k1, strides=1, padding='same',
                 activation='relu', input_shape=(1000, 1)))
model.add(MaxPooling1D(pool_size=2, padding='same'))

# 2
model.add(Conv1D(filters=64, kernel_size=k2, strides=2, padding='same',
                 activation='relu'))
model.add(SpatialDropout1D(0.3, seed=11))

# 3
model.add(Conv1D(filters=64, kernel_size=k3, strides=2, padding='same',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

# 4
model.add(Conv1D(filters=64, kernel_size=k4, strides=1, padding='same',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Flatten())

# dense-DO-dense-DO-OUT
model.add(Dense(int(model.output_shape[1]/3), activation='relu'))
model.add(Dropout(0.3, seed=10))

model.add(Dense(int(model.output_shape[1]/4), activation='relu'))
model.add(Dropout(0.1, seed=9))
model.add(Dense(3, activation='linear'))

# dummy model
# model = Sequential([Input(shape=(1000,)), Dense(3)])

print('kernels_summary: ', k1, k2, k3, k4)
print('model_summary:')
model.summary()


# choose early stop
early_stop = EarlyStopping(monitor='val_loss', patience=patience,
                           restore_best_weights=True,
                           verbose=1)

model.compile(optimizer=Adam(learning_rate=lr),
              loss='mean_absolute_error',
              metrics=['mse'])

history = model.fit(X_TRAIN, yn_TRAIN,
                    batch_size=batch_size, epochs=epochs,
                    validation_split=0.1,
                    callbacks=[early_stop], verbose=2)

# Save model
model.save(os.path.join(result_dir, f'model3.h5'))

# save training history
hist = pd.DataFrame(history.history)
hist.insert(0, 'epoch', history.epoch)
hist.to_csv(os.path.join(result_dir, 'hist.csv'), index=False)

# plot learning curves
plot_loss(history.history, result_dir_name)


# evaluate
print('\n=== Evaluating ===')
eval_MAE, eval_mse = model.evaluate(X_test, yn_test, verbose=0)
print('Evaluation result: MAE {:.3f} - mse {:.3f} (√{:.3f})\n'
      .format(eval_MAE, eval_mse, sqrt(eval_mse)))


# predict
print('=== Testing ===')
test_results = model.predict(X_test, verbose=0)
test_results = pd.DataFrame(test_results,
                            columns=['config', 'N_atoms', 'stoichiometry'])

# copy original predicted values, before decoding them
decoded_results = test_results.copy()
decoded_results[decoded_results.columns] = scaler.inverse_transform(decoded_results[decoded_results.columns])

# for each category: save csv with predictions info
CSid = do_result_df(z_test, yn_test['config'], y_test['config'],
                    test_results['config'], decoded_results['config'],
                    'CSid', result_dir)

size = do_result_df(z_test, yn_test['N_atoms'], y_test['N_atoms'],
                    test_results['N_atoms'], decoded_results['N_atoms'],
                    'size', result_dir)

stoich = do_result_df(z_test, yn_test['stoichiometry'], y_test['stoichiometry'],
                      test_results['stoichiometry'], decoded_results['stoichiometry'],
                      'stoich', result_dir)

# get regression metrics
v_real = 'original_value'
v_pred = 'reverted_prediction'
# mean Absolute Error
MAE_CSid = mean_absolute_error(CSid[v_real], CSid[v_pred])
MAE_size = mean_absolute_error(size[v_real], size[v_pred])
MAE_stoich = mean_absolute_error(stoich[v_real], stoich[v_pred])
print(' MAE – for CSid: {:.3f} – size: {:.3f} – stoich: {:.3f}'
      .format(MAE_CSid, MAE_size, MAE_stoich))

# mean Squared Error
mse_CSid = mean_squared_error(CSid[v_real], CSid[v_pred])
mse_size = mean_squared_error(size[v_real], size[v_pred])
mse_stoich = mean_squared_error(stoich[v_real], stoich[v_pred])
print('√mse – for CSid: {:.3f} – size: {:.3f} – stoich: {:.3f}'
      .format(sqrt(mse_CSid), sqrt(mse_size), sqrt(mse_stoich)))

# R²
r2_CSid = r2_score(CSid[v_real], CSid[v_pred])
r2_size = r2_score(size[v_real], size[v_pred])
r2_stoich = r2_score(stoich[v_real], stoich[v_pred])
print('  R² – for CSid: {:.3f} – size: {:.3f} – stoich: {:.3f}'
      .format(r2_CSid, r2_size, r2_stoich))

# Rmse/σ
norm_Rmse_CSid = sqrt(mse_CSid) / np.std(y_test['config'])
norm_Rmse_size = sqrt(mse_size) / np.std(y_test['N_atoms'])
norm_Rmse_stoich = sqrt(mse_stoich) / np.std(y_test['stoichiometry'])
print('Rmse/σ – for CSid: {:.2f} – size: {:.2f} – stoich: {:.2f}'
      .format(norm_Rmse_CSid, norm_Rmse_size, norm_Rmse_stoich))


# scatter plot
scatter_plot(CSid[v_real], CSid[v_pred], MAE_CSid, mse_CSid, r2_CSid, norm_Rmse_CSid, result_dir, 'CSid')
scatter_plot(size[v_real], size[v_pred], MAE_size, mse_size, r2_size, norm_Rmse_size, result_dir, 'size')
scatter_plot(stoich[v_real], stoich[v_pred], MAE_stoich, mse_stoich, r2_stoich, norm_Rmse_stoich, result_dir, 'stoich')


# feedback
print('\n', '-'*4)
print(time.asctime())
end = time.time()  # time check
print('The whole process took {:.1f}min\n\n'.format((end - start) / 60))

# print('\n\n')
# for consecutive runs
print('\n', '-'*64, '\n', '-'*64, '\n\n\n')
