import itertools
import pathlib
import sys

TOPDIR = pathlib.Path(__file__).resolve().parents[1]

import numpy as np

sys.path.append(str(TOPDIR))
from NASBase.file_utils import pickle_load, json_dump
from settings import Settings

SEED = 1234

LOAD_SUPERNET_PREPROCESSED_FILES = {
    'CIFAR10': {
        0.005: 'load_supernet_NVM1MB_02092024/result_load_supernet_preprocessed.pkl',
        0.000470: 'load_supernet_NVM1MB_testcap_470uF_02282024/result_load_supernet_preprocessed.pkl',
        0.000100: 'load_supernet_NVM1MB_testcap_100uF_02282024/result_load_supernet_preprocessed.pkl',
    },
    'HAR': {
        0.005: 'load_supernet_NVM1MB_HAR_5mF_allblvlparams_03152024/result_load_supernet_NVM1MB_HAR_5mF_allblvlparams_03152024.pkl'
    },
    'KWS': {
    }
}

# TS_PERC_RANGES = [25, 50, 75]   # 25%, 50% , 75% of the feasible solutions
TS_PERC_RANGES = [25, 75]   # 25%, 75% of the feasible solutions
IMO_PERC_RANGES = {
    'CIFAR10': [2],
    'HAR': [46, 4],  # IMO 46% for LAT 25%; IMO 4% for LAT 75%
}

def threshold_selection(global_settings: Settings, dataset, ccap):
    # TODO: support more datasets and ccap
    preprocessed_pickle_filename = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + 'load_supernet/' + LOAD_SUPERNET_PREPROCESSED_FILES[dataset][ccap]

    # Check plot_load_supernet.py for the data structure of this pickle file
    load_supernet_results = pickle_load(preprocessed_pickle_filename)

    settings_per_dataset = global_settings.NAS_SETTINGS_PER_DATASET[dataset]

    all_latency_intpow_data = []
    all_imo_data = []

    for width_multiplier, input_resolution in itertools.product(settings_per_dataset['WIDTH_MULTIPLIER'], settings_per_dataset['INPUT_RESOLUTION']):
        key = f'[{width_multiplier},{input_resolution}]'
        cur_supernet_results = load_supernet_results[key]

        all_latency_intpow_data.extend(cur_supernet_results['LATENCY_INTPOW'])
        all_imo_data.extend(cur_supernet_results['IMC'])

    all_latency_intpow_data.sort()
    all_imo_data.sort()

    imo_perc_ranges_for_dataset = IMO_PERC_RANGES[dataset]

    latency_thresholds = np.percentile(all_latency_intpow_data, TS_PERC_RANGES)
    imo_thresholds = np.percentile(all_imo_data, imo_perc_ranges_for_dataset)

    return {
        'LATENCY_THRESHOLDS': latency_thresholds,
        'IMO_THRESHOLDS': imo_thresholds,

        'TS_PERC_RANGES': TS_PERC_RANGES,
        'IMO_PERC_RANGES': imo_perc_ranges_for_dataset,
        
        'IMO_PERC_THRESHOLD_DICT' : {int(imo_perc_ranges_for_dataset[i]): imo_thresholds[i] for i in range(len(imo_perc_ranges_for_dataset)) },
        'LAT_PERC_THRESHOLD_DICT' : {int(TS_PERC_RANGES[i]): latency_thresholds[i] for i in range(len(TS_PERC_RANGES)) }
    }

def threshold_selection_all(global_settings: Settings):
    all_thresholds = {}

    for dataset, file_list in LOAD_SUPERNET_PREPROCESSED_FILES.items():
        for ccap in file_list.keys():
            key = f'{dataset}-ccap{ccap}'
            all_thresholds[key] = threshold_selection(global_settings, dataset, ccap)

    json_dump(TOPDIR / 'settings' / 'all-thresholds.json', all_thresholds)

def main():
    np.random.seed(SEED) # set random seed

    global_settings = Settings() # default settings

    threshold_selection_all(global_settings)

if __name__ == '__main__':
    main()
