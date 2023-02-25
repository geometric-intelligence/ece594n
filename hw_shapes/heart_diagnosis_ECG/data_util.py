import os
import glob
import pandas as pd
import numpy as np
import subprocess
import geomstats.geometry.spd_matrices as spd
import geomstats.backend as gs
import scipy.io
import matplotlib.pyplot as plt

manifold = spd.SPDMatrices(12)

base_path = './WFDB/'


WFDB_dict = {"426177001":"SB Sinus Bradycardia ", #  <--
            "426783006":"SR	Sinus Rhythm", #  <--
            "427084000":"ST	Sinus Tachycardia", #  <--
            "164889003":"AFIB Atrial Fibrillation", # <--
            "426761007":"SVT Supraventricular Tachycardia", 
            "713422000":"AT Atrial Tachycardia",
            "164890007":"AF	Atrial Flutter",
            "251146004":"LVQRSAL lower voltage QRS in all lead",
            "17366009":"Not added",
            "164865005":"Not added",
            "233897008":"AVRT Atrioventricular Reentrant Tachycardia",
            "251166008":"Not added",
            "164934002":"TWC T wave Change",
            "164931005":"STTU ST tilt up"
            }

def get_all_file_paths():
    """
    Returns list of all patient_ids
    """
    patient_ids = []
    mat_files = glob.glob(base_path + '*.mat')
    for path in mat_files:
        patient_ids.append(os.path.split(path)[-1][:-4])
    return patient_ids


def get_random_patient_id():
    """
    Returns a random patient_id from the basepath
    """
    mat_files = glob.glob(base_path + '*.mat')
    idx = np.random.randint(0, len(mat_files))
    patient_id = os.path.split(mat_files[idx])[-1][:-4]
    return patient_id

def get_rhythm_id(patient_id : str):
    """
    Returns rhythm acronym for a given patient id
    """
    hea_file = open(base_path + patient_id + '.hea')
    ids = hea_file.readlines()[15][5:].split(',')
    if len(ids)==1:    # trimming \n depending on number of diagnoses
        rhythm_id = ids[0][:-1]
    else:
        rhythm_id = ids[0]
    return rhythm_id


def get_single_data(patient_id : str = None):
    """
    Returns ECG data (12 x 5000) array for a patient along with rhythm name

    If no patient_id is given, it returns for a  random patient in list
    """
    rhythm_acr = ""
    if patient_id==None:
        patient_id = get_random_patient_id()
    ecg_data = scipy.io.loadmat(base_path+patient_id+'.mat')['val']
    rhythm_name = WFDB_dict[get_rhythm_id(patient_id=patient_id)]
    return ecg_data, rhythm_name

def check_on_manifold(cov_mat : np.array):
    """
    Checks if the given patient data is on the SPD manifold
    """
    return(gs.all(manifold.belongs(cov_mat)))

    #print("Percentage of cov mat on manifold: {:.2f}".format((on_manifold/num_files)*100))


def compute_corr_mat(patient_id: str = None, plot_corr = True):
    """
    Computes (and plots) correlation matrix (12 x 12) array for a patient with given id

    If no patient_id is given, it plots for a random patient in list
    """
    if patient_id==None:
        patient_id = get_random_patient_id()
    ecg_data, _ = get_single_data(patient_id)
    corr_mat = np.corrcoef(ecg_data)
    
    if plot_corr:
        plt.imshow(corr_mat, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()

    return corr_mat

def plot_ecg(patient_id: str = None, lead:int = 0):
    """
    Plots time series ECG data (5000 samples) at given lead and patient id

    If no patient_id is given, it plots for a random patient in list
    
    lead : 0-based indexing, by default plots all leads
    """
    if patient_id==None:
        patient_id = get_random_patient_id()
    data, _ = get_single_data(patient_id=patient_id)
    
    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    if lead==0:
        for lead in range(12):
            plt.plot(data[lead,:])
    else:
        plt.plot(data[lead, :])
    plt.xlim([-5, 5005])
    plt.grid()
    plt.xlabel('Samples')
    plt.ylabel('microV')
    plt.show()

def get_patients_with_rhythm_id(rhythm_id: str=None):
    result = []
    patient_ids = get_all_file_paths()
    for patient_id in patient_ids:
        curr_rhythm_id = get_rhythm_id(patient_id=patient_id)
        if curr_rhythm_id == rhythm_id:
            result.append(patient_id)
    return result

