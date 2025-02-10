import itertools
import pathlib
import pprint
import sys

TOPDIR = pathlib.Path(__file__).resolve().parents[1]

import numpy as np

sys.path.append(str(TOPDIR))
from NASBase.file_utils import pickle_load, json_dump, json_load
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
    'MSWC': {
        0.005: 'load_supernet_KWS_2D_52_48_stem32_4MBNVM_01122025/result_load_supernet_KWS_2D_52_48_stem32_4MBNVM_01122025.pkl'
    }
}

# TS_PERC_RANGES = [25, 50, 75]   # 25%, 50% , 75% of the feasible solutions
TS_PERC_RANGES = [25, 75]   # 25%, 75% of the feasible solutions
IMO_PERC_RANGES = {
    'CIFAR10': [2],
    'HAR': [46, 4],  # IMO 46% for LAT 25%; IMO 4% for LAT 75%
    'MSWC': [20, 4],  # IMO 20% for LAT 25%; IMO 4% for LAT 75%
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

def simulate_thresholds(dataset, ssopt_results_filenames):
    all_ssopt_results = [json_load(ssopt_results_filename) for ssopt_results_filename in ssopt_results_filenames]
    all_thresholds = json_load(str(TOPDIR / 'settings' / 'all-thresholds.json'))

    ccap = list(LOAD_SUPERNET_PREPROCESSED_FILES[dataset].keys())[0]
    imo_thresholds = all_thresholds[f'{dataset}-ccap{ccap}']['IMO_THRESHOLDS']

    assert len(all_ssopt_results) == len(imo_thresholds)

    all_supernet_stats = []
    for ssopt_results, imo_threshold in zip(all_ssopt_results, imo_thresholds):
        per_supernet_stats = {}
        for subnet_result in ssopt_results['all_subnet_results']:
            supernet_choice = tuple(subnet_result['supernet_choice'])

            if supernet_choice not in per_supernet_stats:
                per_supernet_stats[supernet_choice] = {
                    'num_subnets': 0,
                    'average_flops': 0,
                }

            if subnet_result['imc_prop'] <= imo_threshold:
                per_supernet_stats[supernet_choice]['num_subnets'] += 1
                per_supernet_stats[supernet_choice]['average_flops'] += subnet_result['perf_e2e_contpow_flops']

        for supernet_choice in per_supernet_stats.keys():
            num_subnets = per_supernet_stats[supernet_choice]['num_subnets']
            if num_subnets:
                per_supernet_stats[supernet_choice]['average_flops'] /= num_subnets

        pprint.pprint(per_supernet_stats)
        all_supernet_stats.append(per_supernet_stats)

    return all_supernet_stats

def main():
    np.random.seed(SEED) # set random seed

    global_settings = Settings() # default settings

    threshold_selection_all(global_settings)

    if len(sys.argv) > 1:
        simulate_thresholds('MSWC', sys.argv[1:])

if __name__ == '__main__':
    main()
