import argparse
from enum import IntEnum, Enum
import json
import os, sys
from pathlib import Path
import pprint
import ast

from NASBase.model.mnas_ss import (
    # CIFAR 10
    EXP_FACTORS_CIFAR10,
    KERNEL_SIZES_CIFAR10,
    MOBILENET_NUM_LAYERS_EXPLICIT_CIFAR10,
    SUPPORT_SKIP_CIFAR10,

    WIDTH_MULTIPLIER_CIFAR10,
    INPUT_RESOLUTION_CIFAR10,

    MOBILENET_V2_NUM_OUT_CHANNELS_CIFAR10,

    # HAR
    EXP_FACTORS_HAR,
    KERNEL_SIZES_HAR,
    MOBILENET_NUM_LAYERS_EXPLICIT_HAR,
    SUPPORT_SKIP_HAR,

    WIDTH_MULTIPLIER_HAR,
    INPUT_RESOLUTION_HAR,

    MOBILENET_V2_NUM_OUT_CHANNELS_HAR,
    
    # MSWC
    EXP_FACTORS_MSWC,
    KERNEL_SIZES_MSWC,
    MOBILENET_NUM_LAYERS_EXPLICIT_MSWC,
    SUPPORT_SKIP_MSWC,

    WIDTH_MULTIPLIER_MSWC,
    INPUT_RESOLUTION_MSWC,

    MOBILENET_V2_NUM_OUT_CHANNELS_MSWC,
)

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CURRENT_HOME_PATH = os.path.dirname(CURRENT_DIR_PATH)

class Stages(IntEnum):
    SS_OPT = 1
    TRAIN_SUPERNET = 2
    EVO_SEARCH = 3
    FINE_TUNE = 4

class SSOptPolicy(str, Enum):
    FLOPS = 'FLOPS'
    IMC = 'IMC'
    OURS = 'OURS'

SETTINGS_CATEGORIES = (
    'GLOBAL_SETTINGS',
    'CUDA_SETTINGS',
    'PLATFORM_SETTINGS',
    'NAS_SETTINGS_GENERAL',
    'NAS_SSOPTIMIZER_SETTINGS',
    'NAS_EVOSEARCH_SETTINGS',
    'TINAS',
    'NAS_SETTINGS_PER_DATASET',
    'DUMPER_SETTINGS',
    'LOG_SETTINGS',
    'NAS_TESTING_SETTINGS',
)


class Settings(object): ##default settintgs & discription
    
    GLOBAL_SETTINGS = {
        # should be different for HAR/MSWC
        'EXP_SUFFIX' : "test",
        'USE_REMOTE_LOGGER' : True,
        'REMOTE_LOGGER_RUN_NAME_SUFFIX': '',
        'REMOTE_LOGGER_GROUP_NAME_SUFFIX': '',
        'REMOTE_LOGGER_EXTRA_TAGS': [],
    }    
    
    # ----------------------------------------------------
    # CUDA SETTINGS
    # ----------------------------------------------------
    CUDA_SETTINGS = {
        'GPUIDS' : "0,1,2,3", # GPU card
    }   
    # ----------------------------------------------------
    # PLATFORM SPECIFIC SETTINGS
    # ----------------------------------------------------
    PLATFORM_SETTINGS = {
        'MCU_TYPE' : 'MSP430',   # MSP430   |  MSP432
        'REHM' : 2800, # 75, 300.0,  # ehm equivalent resistance (ohm)
        'VSUP' : 5.892, # 5.0, 3.0 (V)
        'EAV_SAFE_MARGIN' : 0.60, # 0.10, 0.15, 0.20, ..., 0.55, # pessimistic safety margin - available energy will be reduced by this ratio
        'DATA_SZ' : 2, # data size in bytes
        'POW_TYPE' : 'INT',
        'CPU_CLOCK': 16000000,         

        # Constraints:
        # * VM_CAPACITY for vm
        # * NVM_CAPACITY for nvm
        # * CCAP, VON and VOFF for energy per power cycle (cap_energy)
        # * LAT_E2E_REQ for latency
        # * IMC_CONSTRAINT for imc
        #
        # A constraint is skipped if the value is <= 0
        #'VM_CAPACITY' : (4096-2048), 
        'VM_CAPACITY' : (4096-2048),  # in bytes, 400 bytes for application stack
        'NVM_CAPACITY' : 1000000,    # total capacity across two FRAM chips
        'NVM_CAPACITY_ALLOCATION' : [1000000, 1000000], # two FRAM chips model, one for features and another for weights
        'LAT_E2E_REQ' : 10000, # by default no e2e latency req (seconds)
        'CCAP' : 0.005, # 0.005, #0.04, # 0.005, # capacitance (F)
        'VON' : 4.535, # 3.279, 2.99 (V)
        'VOFF' : 3.290, # 3.193, 2.8 (V)
        'IMC_CONSTRAINT': 50,  # 50 means 50%
        
        # obtained by running tests on current EHM
        'REHM_TABLE':
            {
                "0.005" : 2800,
                "0.00047" : 3000,
                "0.0001":  3100
            }
    }

    # ----------------------------------------------------
    # NAS TOOL SETTINGS 
    # ----------------------------------------------------
    NAS_SETTINGS_GENERAL = {        
        'SEED'  : 123,    
        'MODEL_FN_TYPE' : 'MODEL_FN_CONV2D',      # [MODEL_FN_CONV2D | MODEL_FN_CONV1D | MODEL_FN_FC]
        'STAGES': '1,2,3,4',  # run all stages by default
               
        # related to training        
        'CHECKPOINT_DIR' : CURRENT_HOME_PATH + '/TiNAS/NASBase/checkpoints/', # str(Path.home()) + '/DL-sandpit2/TiNAS/NASBase/checkpoints/',                
        'DATASET' : 'CIFAR10',
        
        # check MCUNet training parameters

        # optimizer settings
        # keep the same for HAR/MSWC - not optimizing training process
        'TRAIN_OPT_MOM' : 0.9,    # momentum
        'TRAIN_OPT_WD' : 5e-5, # 4e-5, #3e-4,    # weight decay

        'TRAIN_BATCHNORM_EPSILON': 1e-5,
        
        # debug related
        'TRAIN_PRINT_FREQ' : 100,   # print frequency of training        

        'SEARCH_TIME_TESTING': False,
    }
    
    # Search space optimization default settings
    # SUBNET_SAMPLE_SIZE may be different for different search spaces
    NAS_SSOPTIMIZER_SETTINGS = {
        'SUBNET_SAMPLE_SIZE' : 1000,
        'VALID_SUBNETS_THRESHOLD': 0.1, # 0.05 or some other ratios
        'DO_RESAMPLING': False,
        'SSOPT_POLICY' : SSOptPolicy.FLOPS,
        # specify which constraints to consider
        # VM, NVM, ENERGY should be always checked
        'SSOPT_CONSTRAINTS': 'CHK_PASS_STORAGE,CHK_PASS_SPATIAL,CHK_PASS_ATOMICITY,CHK_PASS_RESPONSIVENESS,CHK_PASS_IMC',
        'SSOPT_RESULTS_FNAME' : CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + GLOBAL_SETTINGS['EXP_SUFFIX'] + '_ssoptlog.json',
        'SSOPT_TRAINED_SUPERNET_FNAME' : CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + GLOBAL_SETTINGS['EXP_SUFFIX'] + '_trsupnetresults.json'        
    }
    
    # Evolutionary search default settings
    # POP_SIZE and GENERATIONS are per-dataset
    NAS_EVOSEARCH_SETTINGS = {
        'POP_SIZE' : 64,
        'GENERATIONS' : 30,  # MCUNet also converges very fast
        # how PARENT_RATIO, MUT_PROB and MUT_RATIO affect convergence?
        'PARENT_RATIO' : 0.2,  # use the same as MCUNet # 0.1, 0.2, 0.3
        # PARENT_RATIO 0.2 is also used in oneshot
        # Check different probs and ratios
        'MUT_PROB': 0.05,
        'MUT_RATIO': 0.5,  # 0.2, 0.4, 0.5, 0.6
        'EVOSEARCH_LOGFNAME' : CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + GLOBAL_SETTINGS['EXP_SUFFIX'] + "_evosearchlog.json",   
        'EVOSEARCH_SCORE_TYPE' : 'ACC_IMC',  # 'ACC'
        'EVOSEARCH_TRIALS': 1,
        
        # This checks only NVM constraints, and it is only for getting results in a shorter time.
        # In general, all of VM, NVM and energy budget should be checked.
        'EVOSEARCH_BYPASS_EFFICIENCY' : False,

        # **For testing only**: keep sampled initial population in a file for reuse.
        # Useful for testing multiple mutation and crossover strategies
        'EVOSEARCH_INITIAL_POPULATION_FNAME': None,
        
        'EVOSEARCH_ENABLE_EVOMEMORY' : False, # use caching mechanism

        'FIXED_NUM_CPU_WORKERS': 16,
        'FIXED_NUM_CPU_WORKERS_MUTATION': 16,
        'FIXED_NUM_CPU_WORKERS_CROSSOVER': 16,
        
        'DEBUG_ENABLED' : False,
             
    }

    # should also be per-dataset - override them in settings/xxx-har.json
    TINAS = {
        'STAGE1_SETTINGS': {
            'DROPPING_BLOCK_LEVEL': {
                'EXP_FACTORS': [],
                'KERNEL_SIZES': [],
                'MOBILENET_NUM_LAYERS_EXPLICIT': [],
                'SUPPORT_SKIP': [],
            },
            'DROPPING_NET_LEVEL': {
                'WIDTH_MULTIPLIER': [],
                'INPUT_RESOLUTION': []
            },
            'DROPPING_ENABLED': False,
        },
        'STAGE2_SETTINGS': {
            # default: not dropping (the default NN search space for the specified dataset)
            # dropped: all blocks use the same dropped choices
            'BLOCK_SEARCH_SPACE': 'default',

            # For G1, G2
            # mutate_default: all blocks use the same probability
            # mutate_blockwise_prob: each block has its own mutation probability
            'MUTATION_OPERATOR': 'mutate_default',
            # For G2
            # Probablities: 0.05, 0.1, 0.2
            # Make block 3 and 4 use the same probability
            # Set N=1 for hyperparameter search
            # itertools.product is useful
            # Default
            'MUT_PROB_PER_BLOCK': [0.05, 0.05, 0.05, 0.05],
            # For exploitation, later during evo search, same prob as default
            'MUT_PROB_PER_BLOCK_EXPLOITATION': [0.05, 0.05, 0.05, 0.05],
            # For exploration, earlier during evo search, higher for first block
            'MUT_PROB_PER_BLOCK_EXPLORATION': [0.2, 0.05, 0.05, 0.05],
            # After N generations, switching from exploration to exploitation
            'BEST_STABLE_GENERATIONS': 5,
        }
    }

    
    NAS_TESTING_SETTINGS = {
        'TRAINED_SUPERNET_FNAME': "",  # for independent testing
        # 'TRAINED_SUPERNET_SSOPT_LOGFNAME': "",        
        # Fine-tuning uses evo search result at the specified generation. If this is None, Fine-tuning uses the last generation.
        'FINETUNE_BASE_GENERATION': None,
    }
    
    
    NAS_SETTINGS_PER_DATASET = {        
                    
        # NAS settings specific for datasets
        'CIFAR10' : {
            
            # model related
            'NUM_BLOCKS'  : 4,
            'NUM_CLASSES' : 10,
            # original STEM_C_OUT = 32, but this incurs two issues:
            # (1) makes the stem too costly: high latency and high NVM usage
            # (2) affects IMC sensitivity: as the stem is fixed in terms of
            #     net-level and block-level parameters, IMC sensitivity is
            #     low for some parameters (ex: width multiplier), which
            #     contradicts what MCUNet shows
            'STEM_C_OUT'  : 16,
            'INPUT_CHANNELS' : 3,            
            'STRIDE_FIRST' : 2,
            'DOWNSAMPLE_BLOCKS' : [0,1,2,3],  #[3,5], #[4, 6],        
            'OUT_CH_PER_BLK' : MOBILENET_V2_NUM_OUT_CHANNELS_CIFAR10,
            'FIRST_BLOCK_HARD_CODED': False,
            'USE_1D_CONV': False,

            # block-level parameters
            'EXP_FACTORS': EXP_FACTORS_CIFAR10,
            'KERNEL_SIZES': KERNEL_SIZES_CIFAR10,
            'MOBILENET_NUM_LAYERS_EXPLICIT': MOBILENET_NUM_LAYERS_EXPLICIT_CIFAR10,
            'SUPPORT_SKIP': SUPPORT_SKIP_CIFAR10,

            # net-level parameters
            'WIDTH_MULTIPLIER': WIDTH_MULTIPLIER_CIFAR10,
            'INPUT_RESOLUTION': INPUT_RESOLUTION_CIFAR10,
            
            # training related
            'TRAIN_DATADIR' : CURRENT_HOME_PATH + '/TiNAS/NASBase/dataset/CIFAR10/', # str(Path.home()) + '/DL-sandpit2/TiNAS/NASBase/dataset/CIFAR10/',
            'TRAIN_OPT_LR' : 0.025, # 0.1, #0.05,    # learning rate
            # check MCUNet training parameters
            'TRAIN_SUPERNET_BATCHSIZE' :256,
            'TRAIN_SUBNET_BATCHSIZE' : 100,
            'VAL_BATCHSIZE' : 500,
            # Use lower supernet epochs, to have more variation
            'TRAIN_SUPERNET_EPOCHS' : 400,  # try 400~500 for 70% accuracy
            'TRAIN_SUBNET_EPOCHS' : 20,
            'FINETUNE_SUBNET_EPOCHS' : 20,
            'FINETUNE_BATCHSIZE': 100,
            'FINETUNE_OPT_LR' : 0.025,  # fine-tune learning rate
                
        },            
        
        'HAR':{
            'NUM_BLOCKS'  : 3,
            'NUM_CLASSES' : 6,
            'STEM_C_OUT'  : 10,  # original = 32, but this makes the stem too costly
            'INPUT_CHANNELS' : 9,
            'STRIDE_FIRST' : 2,
            'DOWNSAMPLE_BLOCKS' : [0,1,2],
            'OUT_CH_PER_BLK' : MOBILENET_V2_NUM_OUT_CHANNELS_HAR,
            'FIRST_BLOCK_HARD_CODED': False,
            'USE_1D_CONV': True,

            # block-level parameters
            'EXP_FACTORS': EXP_FACTORS_HAR,
            'KERNEL_SIZES': KERNEL_SIZES_HAR,
            'MOBILENET_NUM_LAYERS_EXPLICIT': MOBILENET_NUM_LAYERS_EXPLICIT_HAR,
            'SUPPORT_SKIP': SUPPORT_SKIP_HAR,

            # net-level parameters
            'WIDTH_MULTIPLIER': WIDTH_MULTIPLIER_HAR,
            'INPUT_RESOLUTION': INPUT_RESOLUTION_HAR,

            'TRAIN_OPT_LR' : 0.05,    # learning rate

            # https://github.com/healthDataScience/deep-learning-HAR/blob/master/HAR-CNN.ipynb
            'TRAIN_SUPERNET_BATCHSIZE': 64,
            'TRAIN_SUBNET_BATCHSIZE': 64,
            'VAL_BATCHSIZE': 64,

            'TRAIN_SUPERNET_EPOCHS' : 200,
            'FINETUNE_SUBNET_EPOCHS' : 15,
            'FINETUNE_BATCHSIZE': 64,
            'FINETUNE_OPT_LR' : 0.025,  # fine-tune learning rate
        },
        
        'MSWC':{
            'NUM_BLOCKS'  : 4,
            'NUM_CLASSES' : 300,
            'STEM_C_OUT'  : 32,  # original = 32, but this makes the stem too costly
            'INPUT_CHANNELS' : 1,
            'STRIDE_FIRST' : 2,
            'DOWNSAMPLE_BLOCKS' : [1,2,3],
            'OUT_CH_PER_BLK' : MOBILENET_V2_NUM_OUT_CHANNELS_MSWC,
            'FIRST_BLOCK_HARD_CODED': False,
            'USE_1D_CONV': False,

            # block-level parameters
            'EXP_FACTORS': EXP_FACTORS_MSWC,
            'KERNEL_SIZES': KERNEL_SIZES_MSWC,
            'MOBILENET_NUM_LAYERS_EXPLICIT': MOBILENET_NUM_LAYERS_EXPLICIT_MSWC,
            'SUPPORT_SKIP': SUPPORT_SKIP_MSWC,

            # net-level parameters
            'WIDTH_MULTIPLIER': WIDTH_MULTIPLIER_MSWC,
            'INPUT_RESOLUTION': INPUT_RESOLUTION_MSWC,

            'TRAIN_MULTI_GPU': True,

            'TRAIN_OPT_LR' : 0.01,    # learning rate
            
            # https://ieeexplore.ieee.org/document/10241972 uses 1600
            'TRAIN_SUBNET_BATCHSIZE': 100,
            'VAL_BATCHSIZE': 400,  # Does not affect accuracy. Use as large value as possible without exceeding GPU memory
            # Use gradient accumulation to get effective batch size 1600
            'GRADIENT_ACCUMULATION_STEPS': 16,

            'TRAIN_SUPERNET_EPOCHS' : 1000,
            'FINETUNE_SUBNET_EPOCHS' : 25,
            'FINETUNE_OPT_LR' : 0.01,  # fine-tune learning rate
            'FINETUNE_BATCHSIZE': 100,

            'BATCH_SKIP_RATIO': 0.6,

            'PREPROCESSING_PARAMETERS': {
                'NUM_WORKERS': 4,
                'SUBSET': {
                    # Use 50/50 most common words in English/Spanish
                    # Similar way and smaller than https://ieeexplore.ieee.org/document/10241972
                    'en': 50,
                    'es': 50,
                },
                # MFCC parameters are from Harvard repo
                # https://github.com/harvard-edge/multilingual_kws/blob/d7625b87ee525e9c302e12071e2458a0154c0469/multilingual_kws/embedding/input_data.py#L129
                'WINDOW_MS': 30,
                'STRIDE_MS': 20,
                'N_MELS': 40,  # dct_coefficient_count in tensorflow
                'N_MFCC': 40,
                # (padding_left,padding_right, padding_top,padding_bottom)
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                # Original size: 40x51
                'PADDINGS': (0, 1, 6, 6),
            },
        },
    
    }
    
        
        
        
        
    #     'NUM_CLASS' : 10,
    #     'EPOCH': 25,
    #     'TRIALS': 100,
    #     'TRAIN_BATCHSIZE' : 256, # batchsize of the child models
    #     'EXPLORATION' : 0.1,  # high exploration for the first 1000 steps
    #     'REGULARIZATION' : 1e-3,  # regularization strength
    #     'CONTROLLER_CELLS' : 32,  # number of cells in RNN controller
    #     'EMBEDDING_DIM' : 20,  # dimension of the embeddings for each state
    #     'ACCURACY_BETA' : 0.8,  # beta value for the moving average of the accuracy
    #     'CLIP_REWARDS' : 0.0,  # clip rewards in the [-0.05, 0.05] range
    #     'RESTORE_CONTROLLER' : False,  # restore controller to continue training    
    #     'IMG_CHANNEL' : 3, 
    #     'IMG_SIZE' : 32,
        
    # }
    
    
    # ----------------------------------------------------
    # DATASET SPECIFIC SETTINGS 
    # ----------------------------------------------------
    
    

    # ----------------------------------------------------
    # DNN DUMPER SETTINGS
    # ----------------------------------------------------
    DUMPER_SETTINGS = {
        'DUMP_DIR' : '' #<where to store the solutions (*.h5 model, *.h model)
    }


    # ----------------------------------------------------
    # DNN DUMPER SETTINGS
    # ----------------------------------------------------
    LOG_SETTINGS = {    
        'TRAIN_LOG_DIR' : CURRENT_HOME_PATH + '/TiNAS/NASBase/train_log/', # str(Path.home()) + '/DL-sandpit2/TiNAS/NASBase/train_log/',
        'TRAIN_LOG_FNAME' : "train_info.csv", 
        'LOG_LEVEL' : 1, 
        'REMOTE_LOGGING_SYNC_DIR' : CURRENT_HOME_PATH + '/TiNAS/wandb_dir/'
    }

    def get_dict(self):
        result = {}
        for settings_category in SETTINGS_CATEGORIES:
            result[settings_category] = getattr(self, settings_category)
        return result
        
    
    def __init__(self):
        pass
    
    def __str__(self):
        result = ''
        for settings_category in SETTINGS_CATEGORIES:
            result += settings_category + ':=' + '\n'
            result += pprint.pformat(getattr(self, settings_category)) + "\n"
        return result

    # __getstate__ and __setstate__ needed to preserve settings across multiprocessing workers
    # https://docs.python.org/3.9/library/pickle.html#object.__getstate__

    def __getstate__(self):
        ret = {}
        for key, value in type(self).__dict__.items():
            if key.startswith('__'):
                continue
            ret[key] = value
        return ret

    def __setstate__(self, state):
        for key, value in state.items():
            d = type(self).__dict__[key]
            if isinstance(d, dict):
                d.update(value)


def load_settings(fname):
    # load json
    if os.path.exists(fname):
        json_data=open(fname)
        file_data = json.load(json_data)
        return file_data
    else:
        sys.exit("ERROR - file does not exist : " + fname)
        return None


def _update_settings(default_settings, new_settings):
    for k, v in new_settings.items():
        if k not in default_settings:
            print(f'Warning: unknown key {k}')
            continue

        # adds new items, overwrites existing items
        if isinstance(default_settings[k], dict):
            # Update nested dictionaries recursively
            # Inspired by https://stackoverflow.com/a/3233356
            default_settings[k] = _update_settings(default_settings.get(k, {}), v)
        else:
            default_settings[k] = v
    return default_settings

def apply_settings_file(test_settings, settings_filenames):
    for settings_filename in settings_filenames.split(','):
        settings_json = load_settings(settings_filename)

        pprint.pprint(settings_json)

        apply_settings_json(test_settings, settings_json)

def apply_settings_json(test_settings, settings_json):
    for settings_category in SETTINGS_CATEGORIES:
        if settings_category in settings_json:
            old_settings = getattr(test_settings, settings_category)
            new_settings = _update_settings(old_settings, settings_json[settings_category])
            setattr(test_settings, settings_category, new_settings)

def parse_and_apply_saved_settings(global_settings, saved_settings, categories_to_update=None):
    settings_category = None
    cur_section = ''

    for line in saved_settings.split('\n'):
        line = line.strip()

        if line.endswith(':='):
            if settings_category:
                if categories_to_update is None or settings_category in categories_to_update:
                    old_settings = getattr(global_settings, settings_category)
                    _update_settings(old_settings, ast.literal_eval(cur_section))
                cur_section = ''
            settings_category = line.split(':')[0]
        else:
            cur_section += line

def arg_parser(test_settings):
    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument('--gpuid',    type=str, default=argparse.SUPPRESS,  help="GPU selection")
    
    parser.add_argument('--dataset',  type=str,  default=argparse.SUPPRESS,  help="supported dataset including : 1. CIFAR10 (default), 2. HAR. 3 MSWC")
    
    parser.add_argument('--ccap',    type=float, default=argparse.SUPPRESS,   help="capacitor size")
    parser.add_argument('--latreq',    type=float, default=argparse.SUPPRESS,   help="end-to-end latency requirement")
    parser.add_argument('--imcreq',    type=float, default=argparse.SUPPRESS,   help="end-to-end IMC requirement")
    parser.add_argument('--rehm',   type=float, default=argparse.SUPPRESS,   help="EHM equivalent resistance")

    parser.add_argument('--seed',    type=int, default=argparse.SUPPRESS,   help="seed for randomness, default is 123")
    parser.add_argument('--suffix',   type=str, default=argparse.SUPPRESS,   help="experiment run name suffix")
    parser.add_argument('--settings', type=str, default=argparse.SUPPRESS, help="settings files to load")
    parser.add_argument('--settings-json', type=str, default=argparse.SUPPRESS, help="settings data to load")
    parser.add_argument('--stages', type=str, default=argparse.SUPPRESS, help="stages to run, as comma-separated integers : " + ', '.join('{}. {}'.format(p.value, p.name) for p in Stages))
    parser.add_argument('--ss-opt-policy', type=str, default=argparse.SUPPRESS, help='search space optimization policy : ' + ', '.join(p.value for p in SSOptPolicy))
    parser.add_argument('--tr-sup-fname', type=str, default=argparse.SUPPRESS, help="the filename of the trained supernet") # for independent testing
    parser.add_argument('--tr-sup-config', type=str, default=argparse.SUPPRESS, help="the config of the trained supernet") # for independent testing

    parser.add_argument('--no-rlogger', action="store_true", default=False,  help="switch off remote logger")
    parser.add_argument('--rlogger-proj-name', type=str, default=argparse.SUPPRESS, help='Project name for the remote logger')

    args = parser.parse_args()

    
    # first apply settings file
    if 'settings' in args:
        print('ARG_IMPORT_SETTINGS : %s'%(args.settings))
        apply_settings_file(test_settings, args.settings)
    if 'settings_json' in args:
        print('ARG_IMPORT_SETTINGS_JSON : %r' % (args.settings_json))
        apply_settings_json(test_settings, json.loads(args.settings_json))
        
    # then apply custom fine-grain settings
    if 'gpuid' in args:
        print('ARG_SET_GPUIDS : ', args.gpuid)
        test_settings.CUDA_SETTINGS['GPUIDS'] = args.gpuid
    if 'dataset' in args:
        print('ARG_SET_DATASET : ', args.dataset)
        test_settings.NAS_SETTINGS_GENERAL['DATASET'] = args.dataset
    
    if 'seed' in args:
        print('ARG_SET_SEED : ', args.seed)
        test_settings.NAS_SETTINGS_GENERAL['SEED'] = args.seed
    if 'suffix' in args:
        print('ARG_SET_SUFFIX : ', args.suffix)
        test_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] = args.suffix

    if 'ccap' in args:
        print('ARG_SET_CCAP : ', args.ccap)
        test_settings.PLATFORM_SETTINGS['CCAP'] = args.ccap
    if 'latreq' in args:
        print('ARG_SET_LATREQ : ', args.latreq)
        test_settings.PLATFORM_SETTINGS['LAT_E2E_REQ'] = args.latreq
    if 'imcreq' in args:
        print('ARG_SET_LATREQ : ', args.imcreq)
        test_settings.PLATFORM_SETTINGS['IMC_CONSTRAINT'] = args.imcreq
    if 'rehm' in args:
        print('ARG_SET_REHM : ', args.rehm)
        test_settings.PLATFORM_SETTINGS['REHM'] = args.rehm
    if 'stages' in args:
        print('ARG_STAGES : ', args.stages)
        test_settings.NAS_SETTINGS_GENERAL['STAGES'] = args.stages
    if 'tr_sup_fname' in args:
        print('ARG_TRAINED_SUPERNET_FNAME : ', args.tr_sup_fname)
        test_settings.NAS_TESTING_SETTINGS['TRAINED_SUPERNET_FNAME'] = args.tr_sup_fname
    if 'tr-sup-config' in args:
        print('ARG_TRAINED_SUPERNET_CONFIG : ', args.tr_sup_config)
        test_settings.NAS_TESTING_SETTINGS['TRAINED_SUPERNET_CONFIG'] = args.tr_sup_config

    print('ARG_NO_RLOGGER : ', args.no_rlogger)
    test_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER'] = not args.no_rlogger
    if 'rlogger_proj_name' in args:
        print('ARG_RLOGGER_PROJ_NAME : ', args.rlogger_proj_name)
        test_settings.GLOBAL_SETTINGS['RLOGGER_PROJECT_NAME'] = args.rlogger_proj_name
        
    test_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_RESULTS_FNAME'] = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + test_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + '_ssoptlog.json'
    test_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_TRAINED_SUPERNET_FNAME'] = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + test_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + '_trsupnetresults.json'
    test_settings.NAS_EVOSEARCH_SETTINGS['EVOSEARCH_LOGFNAME'] = CURRENT_HOME_PATH + "/TiNAS/NASBase/train_log/" + test_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + "_evosearchlog.json"

    if 'rehm' not in args:
        cap_str = str(test_settings.PLATFORM_SETTINGS['CCAP'])
        estimated_rehm = test_settings.PLATFORM_SETTINGS['REHM_TABLE'][cap_str]
        test_settings.PLATFORM_SETTINGS['REHM'] = estimated_rehm

    print('Updated settings:')
    print(str(test_settings))





    return test_settings

