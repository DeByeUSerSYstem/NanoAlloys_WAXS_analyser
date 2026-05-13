#!/usr/bin/python3

import os
from glob import glob
import time
import sys
import shlex
import shutil
import random
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.pool import ThreadPool


def do_diffractogram(main_dir, template_dir, output_dir, filepath, nums, counter):
    # binpath = "/home/lucia/DEBUSSY_V2.2_2019/bin"

    # ## create working files
    name_only = os.path.splitext(os.path.basename(filepath))[0]
    create_ini_inp(template_dir, filepath, name_only)

    # ## run Claude routines on shell to create the diffractogram
    m = subprocess.run([f"MK_MOLEC_x1.0", f"{name_only}_molmkd.ini"],
                       shell=False, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print error, if any
    if len(m.stderr) > 0:
        print(m.stderr, '\n', m.stdout, '\n' * 3)
        sys.exit(1)

    p = subprocess.run([f"MK_PATTERN_x1.0", f"{name_only}_diffractor.inp"],
                       shell=False, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if len(p.stderr) > 0:
        print(p.stderr, '\n', p.stdout, '\n' * 3)
        sys.exit(1)

    # delete just used INI and INP files
    os.remove(os.path.join(main_dir, f'{name_only}_molmkd.ini'))
    os.remove(os.path.join(main_dir, f'{name_only}_diffractor.inp'))

    # ## peocess pattern
    pattern_name = f'{name_only}_001_X_Iexp.tqi'
    Xq, clean_pattern = process_pattern(pattern_name, output_dir, counter)
    # remove big tqi file (from which we just extracted all we needed)
    os.remove(os.path.join(main_dir, pattern_name))

    # ## add various levels of Poissonian noise to the pattern
    noised_pattern, noise_lvl = add_noise(clean_pattern)
    noised_pattern = normalise_auc(Xq, noised_pattern, 1000)

    # manage text for file name
    str_nums = numbers_to_fname_string(nums)
    # meno più NOvirgola

    # SAVE intensities only of AUGMENTED pattern
    # cs_10100_snr032_M05674100.txt
    final_name = f'{name_only}_snr{noise_lvl}_{str_nums}.txt'
    pattern_path = os.path.join(dataset_dir, final_name)
    np.savetxt(pattern_path, noised_pattern, fmt='%.15f')


def create_ini_inp(Claude_files_folder, fpath, fname):

    full_path_noext = os.path.splitext(fpath)[0]
    relative_path_smp = '/'.join(os.path.split(fpath)[0].split('/')[-2:])

    # open .INI file
    # replace designated string of .INI file with desired string
    # save modified file
    with open(f'{Claude_files_folder}/molmkd.ini', 'rt') as fini:
        ini_content = fini.read()
    ini_content = ini_content.replace('FILENAME', full_path_noext)
    with open(f'{fname}_molmkd.ini', 'xt') as rini:
        rini.write(ini_content)

    # Repeat same process with .INP file
    with open(f'{Claude_files_folder}/diffractor.inp', 'rt') as finp:
        inp_content = finp.read()
    inp_content = inp_content.replace('FILENAME', fname + '_001')
    inp_content = inp_content.replace('DISTANCES', relative_path_smp + '/DISTANCES')
    with open(f'{fname}_diffractor.inp', 'xt') as rinp:
        rinp.write(inp_content)


# only save y-axis between 10-40 2θ, with a sampling step of 0.029
def process_pattern(tqi_fname, original_y_dir, counter):

    # load x-axis in 2θ (t), x-axis in physics-q (q) and y-axis (i=intensities)
    t, q, i = np.loadtxt(tqi_fname, usecols=(0, 1, 2), unpack=True)
    q = q * 20 * np.pi  # crystallography Q, in nm-1

    # consider only intensities between q: 20-75 nm-1 (2θ: 10.23-39.05)
    qmin, qmax = 20.00, 75.00  # nm  22.90, 68.50  -  20.00, 75.00
    idx = np.where((q >= qmin) & (q <= qmax))
    t_cut = t[idx]
    q_cut = q[idx]
    i_cut = i[idx]

    # interpolate intensities to have a 1000 points x-axis
    t_new = np.linspace(t_cut[0], t_cut[-1], 1000)  # 2θ step = 0.02885
    q_new = np.linspace(qmin, qmax, 1000)  # q step = 0.055
    i_new = np.interp(q_new, q_cut, i_cut)

    # normalise y to auc=1000
    y = normalise_auc(q_new, i_new, 1000)


    # SAVE: intensities only = y
    name = tqi_fname.replace("_001_X_Iexp.tqi", "")

    # original long version
    i_path = os.path.join(original_y_dir, f'{name}_y.txt')
    np.savetxt(i_path, i, fmt='%f')

    # SAVE: once, related x-axis: in 2θ, and in crystallography-Q in nm-1
    if counter == 1:
        # original long version
        t_path = os.path.join(data_dir, 'x_2th_wl0p56_original.txt')
        q_path = os.path.join(data_dir, 'x_Qnm-1_original.txt')
        np.savetxt(t_path, t, fmt='%.15f')
        np.savetxt(q_path, q, fmt='%.15f')
        # processed version
        t_new_path = os.path.join(data_dir, 'x_2th_wl0p56_normalised.txt')
        q_new_path = os.path.join(data_dir, 'x_Qnm-1_normalised.txt')
        np.savetxt(t_new_path, t_new, fmt='%.15f')
        np.savetxt(q_new_path, q_new, fmt='%.15f')

    # return new x-axis in Q nm-1  [q_new]
    #  and   the processed and normalised CLEAN pattern (not noised yet)  [y]
    return q_new, y


def add_noise(clean_y):
    """
    Adds a sinthetic Poissonian noise to a clean simulated pattern.
    :param clean_y: intensities of the input pattern (clean simulation)
    :return: noised pattern

    The level of noise is controlled via the "scale" parameter: the higher
    this value, the lower the noise. It serves as a multiplicative factor on
    the input intensities; the final values are then "re-normalised" with a
    division by the same parameter.
    The "scale" parameter is randomly chosen from within a fixed pool, the
    limits of which have been calculated to mach with SNR calculated from the
    available experimental data.
    """
    low_limit = 13
    high_limit = 600
    scale_range = range(low_limit, high_limit)
    weights = [1/np.log(n) for n in scale_range]  # OR 1/n

    scale = random.choices(scale_range, weights=weights)[0]  # weighted random
    # scale = random.randint(low_limit, high_limit)  # gaussian random

    # add noise
    noised_y = [np.random.poisson(lam=i*scale)/scale for i in clean_y]
    noised_y = np.asarray(noised_y)

    # calculate SNR of the just generated pattern
    noise = noised_y - clean_y
    noise_stdev = noise.std()
    snr = np.mean(noised_y) / noise_stdev

    return noised_y, f'{int(snr):03}'


def normalise_auc(x, y, val):
    """Normalises the area-under-the-curve (auc) of a pattern to some value
    :param x, y: respective axix of the pattern
    :param val: scalar, desired value for the auc
    :new_y: y axis properly rescaled to satisfy the auc requirement """
    auc = np.trapz(x=x, y=y)
    new_y = y * (val / auc)
    return new_y


def numbers_to_fname_string(numbers):
    config = numbers[0]     # -0.469973749555
    atom_tot = numbers[1]   # 150
    atom_ag = numbers[2]    # 20

    # ## CONFIG parameter management
    # Check if the number is negative
    if config < 0:
        # Prefix with 'M', remove comma, and minus sign for negative numbers
        str_config = 'M' + str(config)[1:].replace('.', '')
    else:
        str_config = 'P' + str(config).replace('.', '')
    # Right-Pad the string with zeroes to make it 9 characters long, but
    # ONLY if string is shorter than 9 char; if longer, keeps original lenght
    str_config = str_config.ljust(9, '0')

    # ## TOTAL NUMBER OF ATOMS parameter management
    str_atom_tot = 'n' + str(atom_tot)

    # ## NUMBER OF SILVER (Ag) ATOMS parameter management
    str_atom_ag = 'nAg' + str(atom_ag)

    # M0469973749555_n150_nAg20
    return str_config + '_' + str_atom_tot + '_' + str_atom_ag


def fname_string_to_numbers(file_name):

    conf_str_float, tot_str_int, ag_str_int = file_name.split('_')[-3:]

    # Extract the sign (positive or negative) from the first character
    sign = -1 if conf_str_float[0] == 'M' else 1
    # Insert the decimal point after the first digit (omitting leading M or P)
    transformed_number = float(conf_str_float[1] + '.' + conf_str_float[2:])
    # Apply sign & Cast to float32
    conf_float = np.float32(transformed_number * sign)

    tot_int = int(tot_str_int[1:])

    ag_int = int(ag_str_int[3:])

    return conf_float, tot_int, ag_int


def do_empty_folder(directory_name):
    """Create said directory.
    If exsists already, delete it and re-create empty."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    else:
        shutil.rmtree(directory_name)
        os.makedirs(directory_name)


# --------------------------------------------
#  PGM START
# --------------------------------------------
print(time.asctime())
print('doing...')
start = time.time()

# User defined folder name and x-axis metric
this_dir_name = sys.argv[1]

# folders definition
project_dir = os.getcwd()
main_dir = os.path.join(project_dir, 'DATA')
working_files_dir = os.path.join(project_dir, 'working_files')
xyz_dir = os.path.join(main_dir, 'files_xyz')

# create working directory
data_dir = os.path.join(main_dir, this_dir_name)
do_empty_folder(data_dir)

# create folders where to store the calculated (more resolved) patterns
patterns_dir = os.path.join(data_dir, 'long_patterns')
do_empty_folder(patterns_dir)

# create folder for the final processed data: the Training Set
dataset_dir = os.path.join(data_dir, 'dataset')
do_empty_folder(dataset_dir)

# create folder for the data matrices needed by the training script
matrix_dir = os.path.join(data_dir, 'matrices')
do_empty_folder(matrix_dir)

# load file with all filenames and relative parameters
info_file = pd.read_csv('DATA/labels_unique.dat', sep=' ', header=0,
                        usecols=[0, 1, 3, 4],
                        names=['f_id', 'n_ag', 'n_tot', 'config'])

# prepare process for multi-thread
poolsize = multiprocessing.cpu_count() - 2  # numero processi paralleli (-20 on svr)
print('Processors used: ', poolsize)
pool = ThreadPool(poolsize)
cmd = []

# calculate diffractogram for each .XYZ file
count = 0
for i in range(info_file.shape[0]):  # 200 - info_file.shape[0]
    filename = info_file['f_id'][i]
    fpath = os.path.join(xyz_dir, filename)

    labels = [info_file['config'][i], info_file['n_tot'][i], info_file['n_ag'][i]]

    count += 1
    cmd.append(pool.apply_async(do_diffractogram,
               args=(project_dir, working_files_dir, patterns_dir,
                     fpath, labels, count)
                ))
    # do_diffractogram(project_dir, working_files_dir, patterns_dir,
    #                  fpath, config, count)

waste = [c.get() for c in cmd]
pool.close()
pool.join()


assert info_file.shape[0] == len(os.listdir(patterns_dir))
assert info_file.shape[0] == len(os.listdir(dataset_dir))


# feedback
print(time.asctime())
mid = time.time()
print(f'\nall diffractograms created – {int((mid-start)/3600)} h\n')

# create a matrix for each: filenames, diffractograms, labels
all_patterns = []
all_names = []
all_configs = []
all_atoms_TOT = []
all_atoms_Ag = []

# subsets
CS_patterns = []
CS_names = []
CS_configs = []
CS_all_atoms_TOT = []
CS_all_atoms_Ag = []

JA_patterns = []
JA_names = []
JA_configs = []
JA_all_atoms_TOT = []
JA_all_atoms_Ag = []

# cs_10100_snr065_M0659985169947_n150_nAg20.txt
for entry in os.listdir(dataset_dir):
    pat = np.loadtxt(os.path.join(dataset_dir, entry))
    nam = os.path.splitext(entry)[0]
    config, tot_atoms, ag_atoms = fname_string_to_numbers(nam)

    all_patterns.append(pat)
    all_names.append(nam)
    all_configs.append(config)
    all_atoms_TOT.append(tot_atoms)
    all_atoms_Ag.append(ag_atoms)

    # subsets
    if nam.startswith('cs'):
        CS_patterns.append(pat)
        CS_names.append(nam)
        CS_configs.append(config)
        CS_all_atoms_TOT.append(tot_atoms)
        CS_all_atoms_Ag.append(ag_atoms)

    elif nam.startswith('ja'):
        JA_patterns.append(pat)
        JA_names.append(nam)
        JA_configs.append(config)
        JA_all_atoms_TOT.append(tot_atoms)
        JA_all_atoms_Ag.append(ag_atoms)

# save matrices
np.save(os.path.join(data_dir, matrix_dir, 'matrix_patterns'), all_patterns)
np.save(os.path.join(data_dir, matrix_dir, 'matrix_names'), all_names)
np.save(os.path.join(data_dir, matrix_dir, 'matrix_configs'), all_configs)
np.save(os.path.join(data_dir, matrix_dir, 'matrix_atoms_TOT'), all_atoms_TOT)
np.save(os.path.join(data_dir, matrix_dir, 'matrix_atoms_Ag'), all_atoms_Ag)

# save subset matrices
np.save(os.path.join(data_dir, matrix_dir, 'CS_matrix_patterns'), CS_patterns)
np.save(os.path.join(data_dir, matrix_dir, 'CS_matrix_names'), CS_names)
np.save(os.path.join(data_dir, matrix_dir, 'CS_matrix_configs'), CS_configs)
np.save(os.path.join(data_dir, matrix_dir, 'CS_matrix_atoms_TOT'), CS_all_atoms_TOT)
np.save(os.path.join(data_dir, matrix_dir, 'CS_matrix_atoms_Ag'), CS_all_atoms_Ag)

np.save(os.path.join(data_dir, matrix_dir, 'JA_matrix_patterns'), JA_patterns)
np.save(os.path.join(data_dir, matrix_dir, 'JA_matrix_names'), JA_names)
np.save(os.path.join(data_dir, matrix_dir, 'JA_matrix_configs'), JA_configs)
np.save(os.path.join(data_dir, matrix_dir, 'JA_matrix_atoms_TOT'), JA_all_atoms_TOT)
np.save(os.path.join(data_dir, matrix_dir, 'JA_matrix_atoms_Ag'), JA_all_atoms_Ag)


assert len(all_configs) == len(all_atoms_TOT) == len(all_atoms_Ag)
assert len(all_patterns) == len(all_names) == len(all_configs)
assert info_file.shape[0] == len(all_patterns)

assert len(CS_configs) == len(CS_all_atoms_TOT) == len(CS_all_atoms_Ag)
assert len(CS_patterns) == len(CS_names) == len(CS_configs)
assert len(JA_configs) == len(JA_all_atoms_TOT) == len(JA_all_atoms_Ag)
assert len(JA_patterns) == len(JA_names) == len(JA_configs)


# feedback
stop = time.time()
print(f'\nmatrices created – {(stop-mid)/60:.2f} min\n')
duration = (stop - start) / 3600
print(f'\ndone – {int(duration)} h\n')
print(time.asctime())
print('\n\n')
