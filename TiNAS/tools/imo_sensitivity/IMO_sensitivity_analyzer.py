import os, sys
from re import sub
import time
from datetime import datetime
import math
import random
from pprint import pprint
#import subprocess
import itertools
import multiprocessing
import numpy as np
from itertools import cycle, chain
import copy
import ast
import traceback
import argparse



import torch
import torch.nn as nn
#import torchvision
#from torchvision import datasets
from torchinfo import summary
#from torch.utils.tensorboard import SummaryWriter


sys.path.append("../..")
from settings import Settings, arg_parser, load_settings

from run_scripts.server_info import SERVER_GPU_IDS
from logger.remote_logger import RemoteLogger

#sys.path.append("..")
import NASBase.utils as utils
import NASBase.file_utils as file_utils
from NASBase.model.mnas_arch import MNASSuperNet, MNASSubNet
from NASBase.model.mnas_ss import *
from NASBase.model.common_utils import (
    get_network_dimension, get_network_obj, 
    netobj_to_string, netobj_to_pyobj, 
    iter_blk_choices, iter_net_choices,
    round_to_nearest_even_num, round_up_to_nearest_even_num,
    blkchoices_to_blkchoices_ixs, blkchoices_ixs_to_blkchoices,    
    get_subnet, get_supernet, get_sampled_subnet_configs,
    get_dummy_net_input_tensor,
)
from NASBase.model.mnas_subnet_train import train, get_dataset
from NASBase.hw_cost.Modules_inas_v1.IEExplorer.plat_perf import PlatPerf
from NASBase.hw_cost.Modules_inas_v1.CostModel.cnn import pass_constraint_storage

from NASBase import multiprocessing_helper as mp_helper



# PAPER_SEC3 | PAPER_SEC4_NETLEVEL | PAPER_SEC4_BLOCKLEVEL_1 | PAPER_SEC4_BLOCKLEVEL_2 | PAPER_SEC4_BLOCKLEVEL_EXP_KS_NL | RECALC_PERF_COST_SEC3 | RECALC_PERF_COST_SEC4

#EXPERIMENT = 'PAPER_SEC4_NETLEVEL_WITHNVM' 
DATASET = "CIFAR10"



AVAILABLE_GPUIDS = [0,1,2,3,4,5,6,7,8,9]    
#AVAILABLE_GPUIDS = [0,1,2,3]   
#AVAILABLE_GPUIDS = [4,5,6,7,8]   
#AVAILABLE_GPUIDS = [0,1]   

TEST_CHOICE_LEVEL = PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = EXP_FNAME_SUFFIX = DEFAULT_NET_CHOICES = TEST_NET_CHOICES_LIST = BLOCK_LEVEL_TEST_NSAMPLES = ENABLE_NVM_CONSTRAINT = \
EXP_LOG_SUBDIR = NET_LEVEL_TEST_NSAMPLED = TEST_BLOCK_LEVEL_SUBTESTS = NET_LEVEL_FULLY_RANDOM_SUBNETS = \
PERF_RECALC_GPUID_LST = PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = SUBNET_TRAIN_EPOCHS = None



def update_gpuids(server):  
    global AVAILABLE_GPUIDS
      
    if server == "gs093":   AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS093        
    elif server == "gs168": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS168        
    elif server == "gs200": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS200        
    elif server == "gs154": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS154         
    elif server == "gs155": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS155        
    elif server == "gs173": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS173        
    elif server == "gs186": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS186        
    elif server == "gs188": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS188        
    elif server == "gs201": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS201        
    elif server == "gs202": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS202        
    elif server == "gs204": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS204        
    elif server == "gs205": AVAILABLE_GPUIDS = SERVER_GPU_IDS.GS205  
    else:
        sys.exit("update_gpuids:: Error - unknown server {}".format(server))
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in AVAILABLE_GPUIDS)
        

def _get_default_misc_settings_for_dataset(dataset):
    if dataset == "CIFAR10":
        def_settings = {
            "TEST_NET_CHOICES_LIST" : [[0.2, 32], [0.2, 28], [1.0, 32], [1.0, 28]],
            "SUBNET_TRAIN_EPOCHS" : 20,
            "DEFAULT_NET_CHOICES" : [1.0, 32],
        }
        
    elif dataset == "HAR":
        def_settings = {
            "TEST_NET_CHOICES_LIST" : [[0.5, 128], [0.5, 64], [1.0, 128], [1.0, 64]],
            "SUBNET_TRAIN_EPOCHS" : 10,
            "DEFAULT_NET_CHOICES" : [1.0, 128]            
        }
    else:
        sys.exit("_get_default_misc_settings_for_dataset::Error - unknown dataset=", dataset)
        
    # include settings from settings.py
    def_settings["EXP_FACTORS"] = Settings.NAS_SETTINGS_PER_DATASET[dataset]["EXP_FACTORS"],
    def_settings["KERNEL_SIZES"] = Settings.NAS_SETTINGS_PER_DATASET[dataset]["KERNEL_SIZES"],
    def_settings["SUPPORT_SKIP"] = Settings.NAS_SETTINGS_PER_DATASET[dataset]["SUPPORT_SKIP"],            
    def_settings["MOBILENET_NUM_LAYERS_EXPLICIT"] = Settings.NAS_SETTINGS_PER_DATASET[dataset]["MOBILENET_NUM_LAYERS_EXPLICIT"],
    def_settings["WIDTH_MULTIPLIER"] = Settings.NAS_SETTINGS_PER_DATASET[dataset]["WIDTH_MULTIPLIER"],
    def_settings["INPUT_RESOLUTION"] = Settings.NAS_SETTINGS_PER_DATASET[dataset]["INPUT_RESOLUTION"],    
            
    return def_settings


def update_exp_constants(exp_type, dataset='CIFAR10'):
    
    # -- @TODO better to not use globals, to be fixed in future release
    global AVAILABLE_GPUIDS
    global TEST_CHOICE_LEVEL, PARAM_SUPERNET_BLK_CHOICES_TST_TYPE, EXP_FNAME_SUFFIX, DEFAULT_NET_CHOICES, TEST_NET_CHOICES_LIST, BLOCK_LEVEL_TEST_NSAMPLES, ENABLE_NVM_CONSTRAINT
    global EXP_LOG_SUBDIR, NET_LEVEL_TEST_NSAMPLED, TEST_BLOCK_LEVEL_SUBTESTS, NET_LEVEL_FULLY_RANDOM_SUBNETS
    global PERF_RECALC_GPUID_LST, PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE
    global SUBNET_TRAIN_EPOCHS
    
    date_str = datetime.today().strftime('%m%d%Y')
        
    # ===== params for different experiments (constants)
    if exp_type == None:
        sys.exit("update_exp_constants::Error -  exp_type empty")
    
    # ==== accuracy vs imo (random samples, multiple supernets) ====
    if (exp_type == 'PAPER_SEC3'):
        
        TEST_CHOICE_LEVEL = ["BLOCK_LEVEL"] # NET_LEVEL , BLOCK_LEVEL    
        PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = {"BLOCK_LEVEL": "default"}
        EXP_LOG_SUBDIR = "sec3_1MBNVM_{}/".format(date_str)
        EXP_FNAME_SUFFIX = "_sec3_"
        
        DEFAULT_NET_CHOICES = None
        TEST_NET_CHOICES_LIST = _get_default_misc_settings_for_dataset(dataset)["TEST_NET_CHOICES_LIST"]
        BLOCK_LEVEL_TEST_NSAMPLES = 300
        ENABLE_NVM_CONSTRAINT = True
        SUBNET_TRAIN_EPOCHS = _get_default_misc_settings_for_dataset(dataset)["SUBNET_TRAIN_EPOCHS"]

    # ==== dnn characteristics vs accuracy vs imo ====
    
    # no nvm constraint, fewer samples
    elif (exp_type == 'PAPER_SEC4_NETLEVEL'):    
        TEST_CHOICE_LEVEL = ["NET_LEVEL"]   #["NET_LEVEL", "BLOCK_LEVEL"]    
        EXP_LOG_SUBDIR = "sec4_net_level_1MBNVM_{}/".format(date_str)
        EXP_FNAME_SUFFIX = "_sec4_netlvl_"        
        PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = {"NET_LEVEL": "default"}                
        NET_LEVEL_TEST_NSAMPLED = 50
        ENABLE_NVM_CONSTRAINT = False
        NET_LEVEL_FULLY_RANDOM_SUBNETS = False
        SUBNET_TRAIN_EPOCHS = _get_default_misc_settings_for_dataset(dataset)["SUBNET_TRAIN_EPOCHS"]
        
    # with nvm constraint, more samples, fully random
    elif (exp_type == 'PAPER_SEC4_NETLEVEL_WITHNVM'):    
        TEST_CHOICE_LEVEL = ["NET_LEVEL"]   #["NET_LEVEL", "BLOCK_LEVEL"]    
        EXP_LOG_SUBDIR = "sec4_net_level_wnvm_1MBNVM_{}/".format(date_str)
        EXP_FNAME_SUFFIX = "_sec4_netlvl_wnvm_"           
        PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = {"NET_LEVEL": "default"}        
        NET_LEVEL_TEST_NSAMPLED = 100
        ENABLE_NVM_CONSTRAINT = True
        NET_LEVEL_FULLY_RANDOM_SUBNETS = True   
        SUBNET_TRAIN_EPOCHS = _get_default_misc_settings_for_dataset(dataset)["SUBNET_TRAIN_EPOCHS"]
    
    # -- splitting the block-level param study into two parts, so they can run on different servers/gpus --
    elif (exp_type == 'PAPER_SEC4_BLOCKLEVEL_1'):    
        TEST_CHOICE_LEVEL = ["BLOCK_LEVEL"]   #["NET_LEVEL", "BLOCK_LEVEL"]    
        EXP_LOG_SUBDIR = "sec4_block_level_part1_1MBNVM_{}/".format(date_str)
        EXP_FNAME_SUFFIX = "_sec4_blklvl_p1_"        
        TEST_BLOCK_LEVEL_SUBTESTS = ["EXP_FACTORS", "KERNEL_SIZES"]     #["EXP_FACTORS", "KERNEL_SIZES", "NUM_LAYERS", "SUPPORT_SKIP"]                
        PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = {"BLOCK_LEVEL": "all"}
        DEFAULT_NET_CHOICES = _get_default_misc_settings_for_dataset(dataset)["DEFAULT_NET_CHOICES"]
        TEST_NET_CHOICES_LIST = [DEFAULT_NET_CHOICES]
        BLOCK_LEVEL_TEST_NSAMPLES = 300        
        ENABLE_NVM_CONSTRAINT = False   
        SUBNET_TRAIN_EPOCHS = _get_default_misc_settings_for_dataset(dataset)["SUBNET_TRAIN_EPOCHS"]     
        
    elif (exp_type == 'PAPER_SEC4_BLOCKLEVEL_2'):    
        TEST_CHOICE_LEVEL = ["BLOCK_LEVEL"]   #["NET_LEVEL", "BLOCK_LEVEL"]    
        EXP_LOG_SUBDIR = "sec4_block_level_part2_1MBNVM_{}/".format(date_str)
        EXP_FNAME_SUFFIX = "_sec4_blklvl_p2_"                
        TEST_BLOCK_LEVEL_SUBTESTS = ["NUM_LAYERS", "SUPPORT_SKIP"]
        PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = {"BLOCK_LEVEL": "all"}
        DEFAULT_NET_CHOICES = _get_default_misc_settings_for_dataset(dataset)["DEFAULT_NET_CHOICES"]
        TEST_NET_CHOICES_LIST = [DEFAULT_NET_CHOICES]
        BLOCK_LEVEL_TEST_NSAMPLES = 300        
        ENABLE_NVM_CONSTRAINT = False
        SUBNET_TRAIN_EPOCHS = _get_default_misc_settings_for_dataset(dataset)["SUBNET_TRAIN_EPOCHS"]
    

    # elif (exp_type == 'PAPER_SEC4_BLOCKLEVEL_EXP_KS_NL'):    
    #     TEST_CHOICE_LEVEL = ["BLOCK_LEVEL"]  
    #     EXP_LOG_SUBDIR = "sec4_block_level_exp_ks_nl_1MBNVM_{}/".format(date_str)
    #     EXP_FNAME_SUFFIX = "_sec4_blklvl_exp_ks_nl_"
    #     TEST_BLOCK_LEVEL_SUBTESTS = ["EXP_FACTORS", "KERNEL_SIZES", "NUM_LAYERS"]   
    #     PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = {"BLOCK_LEVEL": "all_reduced"}
    #     DEFAULT_NET_CHOICES = _get_default_misc_settings_for_dataset(dataset)["DEFAULT_NET_CHOICES"]
    #     TEST_NET_CHOICES_LIST = [DEFAULT_NET_CHOICES]
    #     BLOCK_LEVEL_TEST_NSAMPLES = 300    
    #     ENABLE_NVM_CONSTRAINT = False
    #     SUBNET_TRAIN_EPOCHS = _get_default_misc_settings_for_dataset(dataset)["SUBNET_TRAIN_EPOCHS"]
        

    # ==== revise (recalc) the perf cost of subnets in all json files in a specific folder ====
    # ==== to be used for different deployment HW, new cost models etc. accuracy does not need to be reestimated ====
    elif (exp_type == 'RECALC_PERF_COST_SEC3'):    
        EXP_LOG_SUBDIR = "sec3_recalc_{}/".format(date_str)
        EXP_FNAME_SUFFIX = "_sec3_"
        PERF_RECALC_GPUID_LST = np.arange(10)       
        PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = "default"     
        ENABLE_NVM_CONSTRAINT = True

    elif (exp_type == 'RECALC_PERF_COST_SEC4'):    
        EXP_LOG_SUBDIR = "sec4_recalc_{}/".format(date_str)
        EXP_FNAME_SUFFIX = "test"
        PERF_RECALC_GPUID_LST = np.arange(16)
        PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE = "all"     
        ENABLE_NVM_CONSTRAINT = False
        TEST_BLOCK_LEVEL_SUBTESTS = ["EXP_FACTORS", "KERNEL_SIZES", "NUM_LAYERS", "SUPPORT_SKIP"]    

    else:
        sys.exit("update_exp_constants::Error! invalid exp_type")
        
    updated_consts = {
        "AVAILABLE_GPUIDS"                                : AVAILABLE_GPUIDS,
        "TEST_CHOICE_LEVEL"                               : TEST_CHOICE_LEVEL, 
        "PARAM_SUPERNET_BLK_CHOICES_TST_TYPE"             : PARAM_SUPERNET_BLK_CHOICES_TST_TYPE, 
        "EXP_FNAME_SUFFIX"                                : EXP_FNAME_SUFFIX, 
        "DEFAULT_NET_CHOICES"                             : DEFAULT_NET_CHOICES, 
        "TEST_NET_CHOICES_LIST"                           : TEST_NET_CHOICES_LIST, 
        "BLOCK_LEVEL_TEST_NSAMPLES"                       : BLOCK_LEVEL_TEST_NSAMPLES, 
        "ENABLE_NVM_CONSTRAINT"                           : ENABLE_NVM_CONSTRAINT,
        "EXP_LOG_SUBDIR"                                  : EXP_LOG_SUBDIR, 
        "NET_LEVEL_TEST_NSAMPLED"                         : NET_LEVEL_TEST_NSAMPLED, 
        "TEST_BLOCK_LEVEL_SUBTESTS"                       : TEST_BLOCK_LEVEL_SUBTESTS, 
        "NET_LEVEL_FULLY_RANDOM_SUBNETS"                  : NET_LEVEL_FULLY_RANDOM_SUBNETS,
        "PERF_RECALC_GPUID_LST"                           : PERF_RECALC_GPUID_LST, 
        "PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE" : PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE,
        "SUBNET_TRAIN_EPOCHS"                             : SUBNET_TRAIN_EPOCHS
    }
    return updated_consts
    





###########################################
# EXT SETTINGS CLASSES
###########################################
class SettingsINTpow(Settings):
    PLATFORM_SETTINGS = copy.deepcopy(Settings.PLATFORM_SETTINGS)

class SettingsCONTpow(Settings):
    PLATFORM_SETTINGS = copy.deepcopy(Settings.PLATFORM_SETTINGS)


###########################################
# HELPERS
###########################################

def _get_remote_logger_init_params(exp_type, server, global_settings, updated_consts, run_name_suffix=""):
    
    init_params = {
        "rlog_proj_name" : "IMO_sensitivity_analysis",
        "rlog_run_name" : exp_type+run_name_suffix,
        "rlog_run_group" : exp_type,
        "rlog_run_config" : {
            
                "script_name" : "IMO_sensitivity_analyzer.py",
                "server"   : server,         
                "exp_start" : datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "settings_obj" : global_settings.get_dict(),
                "exp_constants" : {
                    "EXP_FNAME_SUFFIX": updated_consts["EXP_FNAME_SUFFIX"],            
                    "TEST_CHOICE_LEVEL" : updated_consts["TEST_CHOICE_LEVEL"],
                    "PARAM_SUPERNET_BLK_CHOICES_TST_TYPE" : updated_consts["PARAM_SUPERNET_BLK_CHOICES_TST_TYPE"],
                    "EXP_FNAME_SUFFIX" : updated_consts["EXP_FNAME_SUFFIX"],
                    "DEFAULT_NET_CHOICES" : updated_consts["DEFAULT_NET_CHOICES"],
                    "TEST_NET_CHOICES_LIST" : updated_consts["TEST_NET_CHOICES_LIST"],
                    "BLOCK_LEVEL_TEST_NSAMPLES" : updated_consts["BLOCK_LEVEL_TEST_NSAMPLES"],
                    "ENABLE_NVM_CONSTRAINT" : updated_consts["ENABLE_NVM_CONSTRAINT"],
                    "EXP_LOG_SUBDIR" : updated_consts["EXP_LOG_SUBDIR"],
                    "NET_LEVEL_TEST_NSAMPLED" : updated_consts["NET_LEVEL_TEST_NSAMPLED"],
                    "TEST_BLOCK_LEVEL_SUBTESTS" : updated_consts["TEST_BLOCK_LEVEL_SUBTESTS"],
                    "NET_LEVEL_FULLY_RANDOM_SUBNETS" : updated_consts["NET_LEVEL_FULLY_RANDOM_SUBNETS"],
                    "PERF_RECALC_GPUID_LST" : updated_consts["PERF_RECALC_GPUID_LST"],
                    "PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE" : updated_consts["PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE"],
                    "SUBNET_TRAIN_EPOCHS" : updated_consts["SUBNET_TRAIN_EPOCHS"]
                },
            },
        "rlog_run_tags" : [exp_type, 
                           updated_consts["EXP_LOG_SUBDIR"] if updated_consts["EXP_LOG_SUBDIR"]!=None else "", 
                           updated_consts["EXP_FNAME_SUFFIX"] if updated_consts["EXP_FNAME_SUFFIX"]!=None else ""
                           ]
    }
    
    return init_params
    
    
def _get_remote_logger_obj(cmd_args, global_settings, updated_consts, run_name_suffix=""):
    if ('no-rlogger' not in cmd_args):        
        rl_init_params =  _get_remote_logger_init_params(cmd_args.exp_type, cmd_args.server, global_settings, updated_consts, run_name_suffix=run_name_suffix)
        RLogger = RemoteLogger(
                                proj_name = rl_init_params['rlog_proj_name'],                            
                                run_name = rl_init_params['rlog_run_name'],
                                config = rl_init_params['rlog_run_config'], 
                                tags=rl_init_params["rlog_run_tags"],
                                group=rl_init_params["rlog_run_group"]
                                )
        RLogger.init()
        return RLogger
        
    else:
        return None
    

def _change_constraints_settings(sett, dataset=DATASET):
    # ---- SET CONSTRAINTS
    # ignore NVM constraint (set to large value) 
    # to avoid many subnets being excluded in the IMO analysis
    large_nvm_capacity = 200000000
    sett.PLATFORM_SETTINGS['NVM_CAPACITY'] = large_nvm_capacity
    sett.PLATFORM_SETTINGS['NVM_CAPACITY_ALLOCATION'] = [int(large_nvm_capacity/2), int(large_nvm_capacity/2)]
    
    # only want to get a rough idea for subnet accuracy
    sett.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUBNET_EPOCHS'] = _get_default_misc_settings_for_dataset(dataset)["SUBNET_TRAIN_EPOCHS"]
    
    
# parse test class key from subnet name
def _parse_testclass_key_snname(subnet_name):    
    split_sn_name = subnet_name.split("-")
    preffix = split_sn_name[1].lower()        
    blk_test_class = split_sn_name[2].lower()
    cix = int(split_sn_name[3])
    sidx = int(split_sn_name[4])
    return preffix, blk_test_class, cix, sidx
    

def parametric_supernet_choices(dataset=DATASET):
    
    # types of options
    net_search_options = {        
        'WIDTH_MULTIPLIER' : _get_default_misc_settings_for_dataset(dataset)["WIDTH_MULTIPLIER"],
        'INPUT_RESOLUTION' : _get_default_misc_settings_for_dataset(dataset)["INPUT_RESOLUTION"],
    }
    
    supernet_choices = {
        'test_netop_widthmult':[],
        'test_netop_inputres' : []           
    }
     
    # -- different width multipliers
    lst_widthmult = net_search_options['WIDTH_MULTIPLIER'] 
    lst_inputres = [np.max(net_search_options["INPUT_RESOLUTION"])]
    supernet_choices['test_netop_widthmult'] = iter_net_choices(lst_widthmult, lst_inputres)    
    
    # -- different input resolutions
    lst_widthmult = [np.max(net_search_options["WIDTH_MULTIPLIER"])]
    lst_inputres = net_search_options['INPUT_RESOLUTION']
    supernet_choices['test_netop_inputres'] = iter_net_choices(lst_widthmult, lst_inputres)
    
    return supernet_choices, net_search_options
    
    

def parametric_supernet_blk_choices(tst_type='all', dataset=DATASET):
    
    # types of options per block
    blk_search_options_reduced = {        
        'EXP_FACTORS' : [3, 4, 6],   
        'KERNEL_SIZES': [3, 5, 7],        
        'MOBILENET_NUM_LAYERS_EXPLICIT': [1, 2, 3],                
        'SUPPORT_SKIP' : [False, True],  
    }
    
    blk_search_options_all = {        
        'EXP_FACTORS' : [1, 2, 3, 4, 6, 8],   
        'KERNEL_SIZES': [1, 3, 5, 7, 9],        
        'MOBILENET_NUM_LAYERS_EXPLICIT': [1, 2, 3, 4],                
        'SUPPORT_SKIP' : [False, True],  
    }
    
    blk_search_options_default = {        
        'EXP_FACTORS' : _get_default_misc_settings_for_dataset(dataset)["EXP_FACTORS"],   
        'KERNEL_SIZES': _get_default_misc_settings_for_dataset(dataset)["KERNEL_SIZES"],        
        'MOBILENET_NUM_LAYERS_EXPLICIT': _get_default_misc_settings_for_dataset(dataset)["MOBILENET_NUM_LAYERS_EXPLICIT"],                
        'SUPPORT_SKIP' : _get_default_misc_settings_for_dataset(dataset)["SUPPORT_SKIP"],  
    }    
    
    
    # # permutations to test
    supernet_blk_choices = {}
    
    # default block choices - only one option per feature
    if(tst_type=='default_single'):        
        k_expfactors = [3] # MBCONV3
        k_kernelsizes = [5] # 5x5    
        k_num_layers_explicit = [2] # 2 layers per block    
        k_support_skip = [True]   # skip enabled      
        supernet_blk_choices['test_default_single'] = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip) # MBCONV3, 5x5, 2 layers per block, skip enabled 
        
        return supernet_blk_choices, blk_search_options_default                    
    
    
    elif(tst_type=='default'):        
        k_expfactors = _get_default_misc_settings_for_dataset(dataset)["EXP_FACTORS"] 
        k_kernelsizes = _get_default_misc_settings_for_dataset(dataset)["KERNEL_SIZES"]
        k_num_layers_explicit = _get_default_misc_settings_for_dataset(dataset)["MOBILENET_NUM_LAYERS_EXPLICIT"]
        k_support_skip = _get_default_misc_settings_for_dataset(dataset)["SUPPORT_SKIP"]
        supernet_blk_choices['test_default'] = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip) 
        
        return supernet_blk_choices, blk_search_options_default                    
    
    
    elif (tst_type=='all'):
        
        if "EXP_FACTORS" in TEST_BLOCK_LEVEL_SUBTESTS:
            # -- different conv types
            k_expfactors = blk_search_options_all['EXP_FACTORS']
            k_kernelsizes = [3]   # 3x3
            k_num_layers_explicit = [2] # 2 layers per block    
            k_support_skip = [True] # True
            choices = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)
            supernet_blk_choices["test_convtypes"] = choices
            
        if "KERNEL_SIZES" in TEST_BLOCK_LEVEL_SUBTESTS:
            # -- different K sizes
            k_expfactors = [3]    # MBCONV3
            k_kernelsizes = blk_search_options_all['KERNEL_SIZES']
            k_num_layers_explicit = [2] # 2 layers per block    
            k_support_skip = [True] # True
            choices = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)
            supernet_blk_choices["test_ksizes"] = choices
        
        if "NUM_LAYERS" in TEST_BLOCK_LEVEL_SUBTESTS:        
            # -- different num layers per block 
            k_expfactors = [3]    # MBCONV3
            k_kernelsizes = [3] # 3x3    
            k_num_layers_explicit = blk_search_options_all['MOBILENET_NUM_LAYERS_EXPLICIT']    
            k_support_skip = [True] # True
            choices = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)
            supernet_blk_choices["test_numlayers"] = choices
        
        if "SUPPORT_SKIP" in TEST_BLOCK_LEVEL_SUBTESTS:
            # -- different skip support params per block
            k_expfactors = [3]    # MBCONV3
            k_kernelsizes = [3] # 3x3    
            k_num_layers_explicit = [2] # 2 layers per block    
            k_support_skip = blk_search_options_all['SUPPORT_SKIP']
            choices = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)
            supernet_blk_choices["test_supskip"] = choices    

        #pprint(supernet_blk_choices); sys.exit()

        return supernet_blk_choices, blk_search_options_all     
    
    
    # elif (tst_type=='all_reduced'):
        
    #     if "EXP_FACTORS" in TEST_BLOCK_LEVEL_SUBTESTS:
    #         # -- different conv types
    #         k_expfactors = blk_search_options_reduced['EXP_FACTORS']
    #         k_kernelsizes = [3]   # 3x3
    #         k_num_layers_explicit = [2] # 2 layers per block    
    #         k_support_skip = [True] # True
    #         choices = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)
    #         supernet_blk_choices["test_convtypes"] = choices
            
    #     if "KERNEL_SIZES" in TEST_BLOCK_LEVEL_SUBTESTS:
    #         # -- different K sizes
    #         k_expfactors = [3]    # MBCONV3
    #         k_kernelsizes = blk_search_options_reduced['KERNEL_SIZES']
    #         k_num_layers_explicit = [2] # 2 layers per block    
    #         k_support_skip = [True] # True
    #         choices = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)
    #         supernet_blk_choices["test_ksizes"] = choices
        
    #     if "NUM_LAYERS" in TEST_BLOCK_LEVEL_SUBTESTS:        
    #         # -- different num layers per block 
    #         k_expfactors = [3]    # MBCONV3
    #         k_kernelsizes = [3] # 3x3    
    #         k_num_layers_explicit = blk_search_options_reduced['MOBILENET_NUM_LAYERS_EXPLICIT']    
    #         k_support_skip = [True] # True
    #         choices = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)
    #         supernet_blk_choices["test_numlayers"] = choices
        
    #     if "SUPPORT_SKIP" in TEST_BLOCK_LEVEL_SUBTESTS:
    #         # -- different skip support params per block
    #         k_expfactors = [3]    # MBCONV3
    #         k_kernelsizes = [3] # 3x3    
    #         k_num_layers_explicit = [2] # 2 layers per block    
    #         k_support_skip = blk_search_options_reduced['SUPPORT_SKIP']
    #         choices = iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)
    #         supernet_blk_choices["test_supskip"] = choices    

    #     #pprint(supernet_blk_choices); sys.exit()

    #     return supernet_blk_choices, blk_search_options_reduced     
    
    else:
        sys.exit("Error - invalid parametric_supernet_blk_choices - tst_type")               
                        





def get_subnet_pytorch_obj(subnet_name, sidx, sn_single_choice_per_block, net_choices, global_settings, blk_search_options=None, 
                           dataset=DATASET): 
    
    if net_choices != None:
        width_multiplier = net_choices[0]
        input_resolution = net_choices[1]
    else:
        #width_multiplier = 1.0; input_resolution = 32
        sys.exit("get_subnet_pytorch_obj:: Error - undefined net_choices")
    
    input_tensor = get_dummy_net_input_tensor(global_settings, input_resolution) 
        
    if (blk_search_options == None): 
        supernet_blk_choices = iter_blk_choices(
            Settings.NAS_SETTINGS_PER_DATASET[dataset]["EXP_FACTORS"], 
            Settings.NAS_SETTINGS_PER_DATASET[dataset]["KERNEL_SIZES"], 
            Settings.NAS_SETTINGS_PER_DATASET[dataset]["MOBILENET_NUM_LAYERS_EXPLICIT"],
            Settings.NAS_SETTINGS_PER_DATASET[dataset]["SUPPORT_SKIP"]
            )    
    else:
        supernet_blk_choices = iter_blk_choices(blk_search_options['EXP_FACTORS'], 
                                                blk_search_options['KERNEL_SIZES'], 
                                                blk_search_options['MOBILENET_NUM_LAYERS_EXPLICIT'], 
                                                blk_search_options['SUPPORT_SKIP']
                                                )                
    # print("supernet_blk_choices::: "); print(supernet_blk_choices)
    # print("sn_single_choice_per_block::: "); print(sn_single_choice_per_block)    
    # sys.exit()

    subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(supernet_blk_choices, sn_single_choice_per_block)
    subnet_obj = get_subnet(global_settings, dataset, supernet_blk_choices, subnet_choice_per_blk_ixs, sidx, 
                            width_multiplier=width_multiplier, input_resolution=input_resolution,
                            subnet_name=subnet_name)   
            
    return subnet_obj


# remove is exceeds nvm capacity 
def remove_subnets_exceed_nvm(global_settings, subnet_batch, supernet_blk_choices, width_multiplier, input_resolution,
                              dataset=DATASET):
    print("remove_subnets_exceed_nvm::Enter")
    filtered_batch = []
    for six, subnet_config in enumerate(subnet_batch):
        
        # -- get subnet obj
        #each_subnet_config = each_subnet_config.tolist()       
        subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(supernet_blk_choices, subnet_config)        
        subnet_pyt = get_subnet(global_settings, dataset, supernet_blk_choices, subnet_choice_per_blk_ixs, six, 
                                width_multiplier=width_multiplier, input_resolution=input_resolution)
        input_tensor = get_dummy_net_input_tensor(global_settings, input_resolution)        
        subnet_dims = get_network_dimension(subnet_pyt, input_tensor)         
        subnet_obj = get_network_obj(subnet_dims)             
        
        all_layers_fit_nvm, _, network_nvm_usage = pass_constraint_storage(subnet_obj, global_settings.PLATFORM_SETTINGS)
        #print([all_layers_fit_nvm, _, network_nvm_usage])
    
        if all_layers_fit_nvm:
            filtered_batch.append(subnet_config)
            
        print("filtered_batch size = {}".format(len(filtered_batch)), end='\r')
        #time.sleep(1)
        
    return filtered_batch


def remove_duplicates(subnet_batch_ixs, supernet_blk_choices):    
    # easier to work with indices
    # sets can't work with 2d lists, so convert to str
    subnet_batch_str = [str(s) for s in subnet_batch_ixs]        
    filtered_batch_ixs = [ast.literal_eval(s) for s in set(subnet_batch_str)]      
    return filtered_batch_ixs


# take a batch of subnets, change their net choices
def subnet_batch_apply_netchoices(global_settings, subnet_configs, lst_blkchoices, net_choices, dataset=DATASET):
    width_multiplier, input_resolution = net_choices[0], net_choices[1]
    all_subnets = []
    for six, subnet_config in enumerate(subnet_configs):  
        # -- get subnet obj
        subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(supernet_blk_choices, subnet_config)        
        subnet_pyt = get_subnet(global_settings, dataset, supernet_blk_choices, subnet_choice_per_blk_ixs, six, 
                                width_multiplier=width_multiplier, input_resolution=input_resolution)
        input_tensor = get_dummy_net_input_tensor(global_settings, input_resolution)        
        subnet_dims = get_network_dimension(subnet_pyt, input_tensor)         
        subnet_obj = get_network_obj(subnet_dims) 
        all_subnets.append(subnet_obj)
    return all_subnets
    


def generate_subnet_options_alltests(global_settings, supernet_blk_choices, blk_search_options=None, net_choices=None, net_prefix="", n_samples=100,
                                     dataset=DATASET):
    print("generate_subnet_options_alltests::Enter")
    num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']        
    all_subnets = {}
    n_subsamples = int(math.ceil(n_samples/10))   
    #n_subsamples = 1000 
    
    assert(net_choices != None)
    
    # -- for each test class --
    cix = 0
    for each_blkchoice_class, lst_blkchoices in supernet_blk_choices.items():     
        all_subnets[each_blkchoice_class] = []
        
        if len(lst_blkchoices) > 0:
               
            choices_per_block = [list(x) for x in itertools.product(lst_blkchoices, repeat=num_blocks)]
            print("test_class = {}, blk SS size = {}".format(each_blkchoice_class, len(choices_per_block)))
            
            sampled_subnet_configs = []
            
            n_samples = np.min([n_samples, len(choices_per_block)])
            
            resampling=0
            while (len(sampled_subnet_configs)<n_samples):           
                tmp_subnet_configs = random.sample(choices_per_block, np.min([n_subsamples, len(choices_per_block)]))
                #print("tmp_subnet_configs len=", len(tmp_subnet_configs))
                
                if ENABLE_NVM_CONSTRAINT == True:
                    # check if valid, remove invalid ones
                    tmp_subnet_configs = remove_subnets_exceed_nvm(global_settings, tmp_subnet_configs, lst_blkchoices, net_choices[0], net_choices[1]) 
                
                sampled_subnet_configs.extend(tmp_subnet_configs)                                
                sampled_subnet_configs = remove_duplicates(sampled_subnet_configs, lst_blkchoices)
                
                print("num valid unique subnets = ", len(sampled_subnet_configs))    
                resampling+=1
                
                if resampling>100: # tried to resample many times but still cannot find enough unique subnets
                    print("------- WARNING: cannot find sufficient number of unique valid subnets - [supnet={},{}][sz={}]".format(
                                                                                                                                net_choices[0], net_choices[1], 
                                                                                                                                len(sampled_subnet_configs)))
                    break
            
            # generate all subnets for test class
            for sidx, sn_single_choice_per_block in enumerate(sampled_subnet_configs):                
                subnet_name = "SN-{}-{}-{}-{}".format(net_prefix.upper(), each_blkchoice_class.upper(), cix, sidx)
                revised_sidx = "{}-{}".format(cix, sidx)        
                all_subnets[each_blkchoice_class].append([subnet_name, revised_sidx, blk_search_options, sn_single_choice_per_block, net_choices])
        else:
            pass
            
        cix+=1
    
    return all_subnets

    


###########################################
# GPU WORKERS
###########################################
def get_subnet_perf_data(subnet_info, updated_consts, dataset=DATASET):
    
    # init
    subnet_name="UNKNOWN_SUBNET";subnet_cpb=[];subnet_obj=[]
    max_val_acc=-10; e2e_lat_intpow=-10; ip_tot_npc=-10; e2e_lat_contpow=-10; e2e_lat_contpow_fp=-10; int_mng_cost_proportion=-10; int_mng_cost_proportion_cpfp=-10
    exec_design_intpow=[]; error_intpow=None; error_contpow_fp=None
    
    subnet_name, revised_sidx, blk_search_options, sn_single_choice_per_block, net_choices = subnet_info
    
    # create settings    
    global_settings_intpow = SettingsINTpow() # default settings
    global_settings_intpow.PLATFORM_SETTINGS["POW_TYPE"] =  "INT"
    global_settings_contpow = SettingsCONTpow() # default settings
    global_settings_contpow.PLATFORM_SETTINGS["POW_TYPE"] = "CONT"
    
    UPCONST_ENABLE_NVM_CONSTRAINT = updated_consts["ENABLE_NVM_CONSTRAINT"]
    
    #print("UPCONST_ENABLE_NVM_CONSTRAINT = ", UPCONST_ENABLE_NVM_CONSTRAINT)
    if (UPCONST_ENABLE_NVM_CONSTRAINT == False):
        _change_constraints_settings(global_settings_intpow, dataset=dataset)
        _change_constraints_settings(global_settings_contpow, dataset=dataset)
        
    #print(global_settings_intpow.PLATFORM_SETTINGS['NVM_CAPACITY']); print(global_settings_contpow.PLATFORM_SETTINGS['NVM_CAPACITY']); 
    #print(global_settings_intpow.NAS_SETTINGS_PER_DATASET[DATASET]['TRAIN_SUBNET_EPOCHS']); print(global_settings_contpow.NAS_SETTINGS_PER_DATASET[DATASET]['TRAIN_SUBNET_EPOCHS']);     
    #print("----> " , global_settings_intpow.PLATFORM_SETTINGS["POW_TYPE"], global_settings_contpow.PLATFORM_SETTINGS["POW_TYPE"])
    
    try:  
        subnet_pyobj = get_subnet_pytorch_obj(subnet_name, revised_sidx, sn_single_choice_per_block, net_choices, global_settings_intpow, 
                                                blk_search_options=blk_search_options) 
                
        subnet_name = subnet_pyobj.name
        subnet_cpb = subnet_pyobj.choice_per_block
        width_multiplier, input_resolution = subnet_pyobj.net_choices
        
        # -- get subnet costs
        #net_input = torch.rand(1, 3, input_resolution, input_resolution)
        net_input = get_dummy_net_input_tensor(global_settings_intpow, input_resolution)
        subnet_dims = get_network_dimension(subnet_pyobj, input_tensor = net_input)         
        subnet_obj = get_network_obj(subnet_dims)       
                    
        # get perf for INT pow            
        e2e_lat_intpow, exec_design_intpow, error_intpow = _helper_get_network_perf(subnet_obj, global_settings_intpow)
        
        ip_tot_npc = np.sum([l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1        # total power cycles
        ip_tot_rc = np.sum([l['L_rc_tot'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1    # total recharge time
        ip_tot_rb = np.sum([l['cost_brk']['rb'][1] * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
        ip_tot_backup = np.sum([ (l['cost_brk']['bd'][1] + l['cost_brk']['bl'][1]) * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
        ip_tot_recovery = np.sum([ (l['cost_brk']['fd'][1] + l['cost_brk']['fl'][1]) * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
        ip_tot_computation = np.sum([ l['cost_brk']['cp'][1] * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
        
                    
        # get perf for CONT pow                 
        # cont pow performance - fixed params : same as intpow
        e2e_lat_contpow_fp, exec_design_contpow_fp, error_contpow_fp = _helper_get_network_perf(subnet_obj, global_settings_contpow, fixed_params=exec_design_intpow)                        
                                
        # Calc IMC (vs. contpow_fp)
        active_time = (e2e_lat_intpow - ip_tot_rc)        
        int_mng_cost_proportion_cpfp = ((active_time-e2e_lat_contpow_fp)/active_time)*100
        
        # error reporting
        if any(x == -1 for x in [ip_tot_npc, ip_tot_rc, ip_tot_rb, e2e_lat_intpow, e2e_lat_contpow_fp]):                
            error_net_perf = True
        else:
            error_net_perf = False
            
        _preffix, _blk_test_class, _cix, _sidx = _parse_testclass_key_snname(subnet_name)
            
            
        subnet_perf_data = {            
            "subnet_name" : subnet_name,
            "subnet_obj" : netobj_to_pyobj(subnet_obj),
            "test_class" : _preffix + "--" + _blk_test_class,
            "subnet_choice_per_blk": subnet_cpb,
            "net_choices" : subnet_pyobj.net_choices,
            
            # validation accuracy - will be populated later
            #"max_val_acc": None,    
            
            # perf int pow 
            "error_net_perf"  : error_net_perf,          
            "perf_e2e_intpow_lat": e2e_lat_intpow,
            "perf_exec_design_intpow": exec_design_intpow,
            "perf_error_intpow": error_intpow,
          
            # perf cont pow - fixed params
            "perf_e2e_contpow_fp_lat": e2e_lat_contpow_fp,
            "perf_exec_design_contpow_fp": exec_design_contpow_fp,
            "perf_error_contpow_fp": error_contpow_fp,          
                        
            # proportion
            "imc_prop" :  int_mng_cost_proportion_cpfp,
        }        
                            
    except Exception as e:            
        error_net_perf = True
        subnet_perf_data = None
        pprint(e)
        tb = traceback.format_exc()
        print(tb)
        print("subnet_cpb: ", subnet_cpb)
        
    return subnet_perf_data, error_net_perf


def _helper_get_network_perf(subnet_obj, global_settings: Settings, fixed_params=None):        
    performance_model = PlatPerf(global_settings.NAS_SETTINGS_GENERAL, global_settings.PLATFORM_SETTINGS)
    time_performance, exec_design, error = performance_model.get_inference_latency(subnet_obj, fixed_params=fixed_params)    
    return time_performance, exec_design, error


def mpworker_get_updated_perfcost(worker_id, updated_consts, ix, len_gpuids_lst, subnet_batch):
    
    print("[WORKERID-{}]:: Enter : has {} jobs ".format(worker_id, len(subnet_batch)))
    
    batched_results = []
    for each_subnet in subnet_batch:
        perf_result,_ = get_subnet_perf_data(each_subnet[:-1], updated_consts)    # last item in each_subnet includes old perf data, so exclude it   
        batched_results.append(perf_result)
        
        old_perf_results = each_subnet[-1]        
        subnet_name = each_subnet[0]
        
        new_npc = np.sum([l['npc'] for l in perf_result["perf_exec_design_intpow"]])
        
        print("[WID-{}-progress={}%] Finished perf_recalc subnet: {}, Lat_INT:[old={:0.2f}, new={:0.2f}], npc:[old={}, new={}], imc_pop:[old={:0.2f}, new={:0.2f}]".format(
                                worker_id, 
                                round(((ix+1)/len_gpuids_lst)*100, 2), 
                                subnet_name,  
                                old_perf_results["perf_e2e_intpow_lat"], perf_result["perf_e2e_intpow_lat"], 
                                old_perf_results["npc"], new_npc,                                                                                                                               
                                old_perf_results["imc_prop"], perf_result["imc_prop"], 
              ))
    
    return batched_results
    
    

def gpu_worker(gpuid, subnet_list, updated_consts, cmd_args, run_train):    
    print("GPUID [%d] :: Enter : has %d jobs " % (gpuid, len(subnet_list)))
    
    UPCONST_EXP_LOG_SUBDIR = updated_consts["EXP_LOG_SUBDIR"]
    UPCONST_EXP_FNAME_SUFFIX = updated_consts["EXP_FNAME_SUFFIX"]
    
    # create settings    
    global_settings_tmp = SettingsINTpow() # default settings
    global_settings_tmp.PLATFORM_SETTINGS["POW_TYPE"] =  "INT"
    
    global_settings_tmp.LOG_SETTINGS['TRAIN_LOG_DIR'] = global_settings_tmp.LOG_SETTINGS['TRAIN_LOG_DIR'] + UPCONST_EXP_LOG_SUBDIR
    logfname = global_settings_tmp.LOG_SETTINGS['TRAIN_LOG_DIR']  + "results_gpuid-" + str(gpuid) + UPCONST_EXP_FNAME_SUFFIX + '_results_exp_acc_imc.json'
    
    rlog = _get_remote_logger_obj(cmd_args, global_settings_tmp, updated_consts, run_name_suffix="_gpu{}".format(gpuid))
    
    #print(global_settings_tmp.PLATFORM_SETTINGS['NVM_CAPACITY']); print(global_settings_tmp.PLATFORM_SETTINGS['NVM_CAPACITY']); 
    #print(global_settings_tmp.NAS_SETTINGS_PER_DATASET[DATASET]['TRAIN_SUBNET_EPOCHS']); print(global_settings_tmp.NAS_SETTINGS_PER_DATASET[DATASET]['TRAIN_SUBNET_EPOCHS']); 
    
    print("----> " , global_settings_tmp.PLATFORM_SETTINGS["POW_TYPE"], global_settings_tmp.PLATFORM_SETTINGS["POW_TYPE"])
    
    # init
    subnet_name="UNKNOWN_SUBNET";
    max_val_acc=-10; e2e_lat_intpow=-10; ip_tot_npc=-10; e2e_lat_contpow=-10; e2e_lat_contpow_fp=-10; int_mng_cost_proportion=-10; int_mng_cost_proportion_cpfp=-10
    exec_design_intpow=[]; exec_design_contpow=[]; error_intpow=None; error_contpow=None;error_contpow_fp=None 
    
    # process subnet batch allocated to this GPU worker    
    subnet_results = {}
    for i, each_subnet in enumerate(subnet_list):   
        
        subnet_name, revised_sidx, blk_search_options, sn_single_choice_per_block, net_choices = each_subnet
        subnet_perf_data, error_net_perf = get_subnet_perf_data(each_subnet, updated_consts)
        
        if run_train == True:
            subnet_pyobj = get_subnet_pytorch_obj(subnet_name, revised_sidx, sn_single_choice_per_block, net_choices, global_settings_tmp, 
                                        blk_search_options=blk_search_options) 
            max_val_acc = train(subnet_pyobj, gpuid, global_settings_tmp,
                                dataset=DATASET)    # -- get accuracy
        else:
            max_val_acc = -1
            
        if subnet_perf_data != None:
            subnet_perf_data["max_val_acc"] = max_val_acc   # update dict with accuracy
           
        
        if (error_net_perf == False):
            subnet_name = subnet_perf_data["subnet_name"]        
            error_net_perf = subnet_perf_data["subnet_name"]        
            e2e_lat_intpow = subnet_perf_data["perf_e2e_intpow_lat"]        
            e2e_lat_contpow_fp = subnet_perf_data["perf_e2e_contpow_fp_lat"]        
            int_mng_cost_proportion_cpfp = subnet_perf_data["imc_prop"]
            print("[GPU-{}-progress={}%] Finished processing subnet: {}, max_val_acc = {}, e2eLat: under INTpow={}, CONTpow={}, ic_prop_cpfp={}".format(
                    gpuid, 
                    round(((i+1)/len(subnet_list))*100,2),
                    subnet_name, max_val_acc, 
                    e2e_lat_intpow, e2e_lat_contpow_fp, 
                    int_mng_cost_proportion_cpfp))
            
        else:            
            print("[GPU-{}-progress={}%] ERROR processing subnet: {}, max_val_acc = {}, e2eLat: under INTpow={}, CONTpow={}, ic_prop_cpfp={}".format(
                gpuid, 
                round(((i+1)/len(subnet_list))*100,2),
                subnet_name, max_val_acc, 
                e2e_lat_intpow, e2e_lat_contpow_fp, 
                int_mng_cost_proportion_cpfp))
            
        rlog.log({"gpuid" : gpuid,
                    "progress" : round(((i+1)/len(subnet_list))*100,2),
                    "error" : error_net_perf,
                    "subnet_name" : subnet_name,
                    "max_val_acc" : max_val_acc,
                    "int_mng_cost_proportion_cpfp" : int_mng_cost_proportion_cpfp})
        
        subnet_results[subnet_name] = subnet_perf_data
        
        # overwrite json - keep updating the json file
        file_utils.delete_file(logfname)   
        file_utils.json_dump(logfname, subnet_results)        
        
        time.sleep(2)
    
    # overwrite json
    file_utils.delete_file(logfname)   
    file_utils.json_dump(logfname, subnet_results)
    
    rlog.finish()
        
      
###########################################
# RECALC
###########################################
def run_perfcost_recalc(gpuids_lst, test_type, updated_consts):
    print("======= run_perfcost_recalc::ENTER ============")
    
    global_settings = Settings() # default settings
    
    # go through every json result file and create subnet_list
    _, blk_search_options_testclass_netlvl = parametric_supernet_blk_choices(tst_type="default")
    _, blk_search_options_testclass_blklvl = parametric_supernet_blk_choices(tst_type="all")
    
    
    global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + EXP_LOG_SUBDIR
    
    rlog = _get_remote_logger_obj(cmd_args, global_settings, updated_consts)
    
    num_errors = 0
    
    for ix, each_gpu_id in enumerate(gpuids_lst):
        
        #logfname = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR'] + EXP_LOG_SUBDIR +  "results_gpuid-" + str(each_gpu_id) + global_settings.GLOBAL_SETTINGS['EXP_SUFFIX'] + '_results_exp_acc_imc.json'
        logfname = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR']  + "results_gpuid-" + str(each_gpu_id) + EXP_FNAME_SUFFIX + '_results_exp_acc_imc.json'
        exp_data = file_utils.json_load(logfname) 
        print(">>> logfname : ", logfname)
        
        all_recal_subnets_lst = []
        
        # get subnet list
        for each_subnet_name, each_subnet_data in exp_data.items():
        
            sidx = 0        
            test_class = each_subnet_data["test_class"]
            if "test_netop" in test_class:
                blk_search_options = blk_search_options_testclass_netlvl
            else:
                blk_search_options = blk_search_options_testclass_blklvl
            
            sn_single_choice_per_block = each_subnet_data["subnet_choice_per_blk"]
            net_choices = each_subnet_data["net_choices"]
            old_perf_data = {
                "perf_e2e_intpow_lat" : each_subnet_data["perf_e2e_intpow_lat"],
                "perf_e2e_contpow_fp_lat" : each_subnet_data["perf_e2e_contpow_fp_lat"],
                "npc" : np.sum([l['npc'] for l in each_subnet_data["perf_exec_design_intpow"]]),
                "imc_prop" : each_subnet_data["imc_prop"]
            }
            recalc_subnet_item = [each_subnet_name, sidx, blk_search_options, sn_single_choice_per_block, net_choices, old_perf_data]            
            all_recal_subnets_lst.append(recalc_subnet_item)
    
        # ---------- Procress batch
        batched_perf_results = []    
        available_cpus = mp_helper.get_max_num_workers(worker_type='CPU')
        #available_cpus = 2
        batched_subnets = np.array_split(all_recal_subnets_lst, available_cpus)
        
        all_worker_results = mp_helper.run_multiprocessing_workers(
            num_workers=available_cpus,
            worker_func= mpworker_get_updated_perfcost,
            worker_type='CPU',
            common_args=(updated_consts, ix, len(gpuids_lst)),
            worker_args=(batched_subnets),
        )
        
        # loop through all worker results
        print("======= run_perfcost_recalc:: updating recalced perf")
        
        for worker_result in all_worker_results:
            
            for each_sn in worker_result:
                sn_name = each_sn["subnet_name"]
                
                # -- update subnet --
                exp_data[sn_name]["error_net_perf"] = each_sn["error_net_perf"]
                if each_sn["error_net_perf"] == True:
                    num_errors+=1
                
                # perf int pow
                exp_data[sn_name]["perf_e2e_intpow_lat"] = each_sn["perf_e2e_intpow_lat"]
                exp_data[sn_name]["perf_exec_design_intpow"] = each_sn["perf_exec_design_intpow"]
                exp_data[sn_name]["perf_error_intpow"] = each_sn["perf_error_intpow"]
                
                # perf cont pow - fixed params
                exp_data[sn_name]["perf_e2e_contpow_fp_lat"] = each_sn["perf_e2e_contpow_fp_lat"]
                exp_data[sn_name]["perf_exec_design_contpow_fp"] = each_sn["perf_exec_design_contpow_fp"]
                exp_data[sn_name]["perf_error_contpow_fp"] = each_sn["perf_error_contpow_fp"]
                
                # proportion
                exp_data[sn_name]["imc_prop"] = each_sn["imc_prop"]
                        
        # save updated JSON data
        print("======= run_perfcost_recalc:: updating JSON: ", logfname)
        file_utils.json_dump(logfname, exp_data)
        
        rlog.log({
                    "progress" : round(((ix+1)/len(gpuids_lst))*100, 2),                                                             
                    "tot_errors" : num_errors,
                    })
        
    rlog.finish()
        
        
        
###########################################
# ARGUMENTS
###########################################              
def arg_parser():
    parser = argparse.ArgumentParser('test_parametric: Parser User Input Arguments')
    parser.add_argument('--exp_type',   type=str, default=argparse.SUPPRESS,  help="experiment type")    
    parser.add_argument('--server',  type=str,  default=argparse.SUPPRESS,  help="server running this script")    
    parser.add_argument('--seed',    type=int, default=argparse.SUPPRESS,   help="seed for randomness, default is 123")
    parser.add_argument('--no-rlogger', action="store_true",   help="switch off remote logger")
    
    args = parser.parse_args()    
    return args


###########################################
# MAIN
###########################################
if __name__ == '__main__':
    
    print ("========= PARAMETRIC TEST ==============")
    
    # ========== CONFIG ==========
    global_settings = Settings() # default settings
    cmd_args = arg_parser()
    
    print("finished argparse !!")
    
    if 'seed' in cmd_args:
        print('ARG_SET_SEED : ', cmd_args.seed)
        utils.set_seed(cmd_args.seed)
    else:    
        utils.set_seed(global_settings.NAS_SETTINGS_GENERAL['SEED'])        
    
    # update experiment configs
    if 'server' in cmd_args: 
        update_gpuids(cmd_args.server) 
    else :
        update_gpuids(None) 
    
    if 'exp_type' in cmd_args: 
        updated_consts = update_exp_constants(cmd_args.exp_type) 
    else :
        updated_consts = update_exp_constants(None)         
        
    #pprint(AVAILABLE_GPUIDS); sys.exit()
    #pprint(EXP_FNAME_SUFFIX)
    #sys.exit()
    
    # only want to get a rough idea for subnet accuracy    
    global_settings.NAS_SETTINGS_PER_DATASET[DATASET]['TRAIN_SUBNET_EPOCHS'] = _get_default_misc_settings_for_dataset(DATASET)["SUBNET_TRAIN_EPOCHS"]
    global_settings.NAS_SETTINGS_GENERAL['DATASET'] = DATASET
        
    # ---- EXP SETUP 
    logdir = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR']  + EXP_LOG_SUBDIR
    file_utils.dir_create(logdir)    
    
    os.environ["WANDB_DIR"] = global_settings.LOG_SETTINGS["REMOTE_LOGGING_SYNC_DIR"]
    os.environ["WANDB_SILENT"] = "true"    
    
    # ---- RECALC TYPE EXPERIMENT ?
    if ("RECALC_PERF_COST" in cmd_args.exp_type):
        #sys.exit()
        run_perfcost_recalc(PERF_RECALC_GPUID_LST, PERF_RECALC_PARAM_SUPERNET_BLK_CHOICES_TST_TYPE, updated_consts)
        sys.exit()
    
    
    # ---- GET SUBNETS, WITH CHOICES
    all_subnets = {}
    
    # -- formulate net-level testing subnets
    if "NET_LEVEL" in TEST_CHOICE_LEVEL:
        
        print("---- performing NET_LEVEL test -----")
    
        # get choices
        supernet_choices, net_search_options = parametric_supernet_choices()    
        supernet_blk_choices, blk_search_options = parametric_supernet_blk_choices(tst_type=PARAM_SUPERNET_BLK_CHOICES_TST_TYPE['NET_LEVEL'])
        # print ("---------"); # pprint(supernet_blk_choices); # print ("---------")
        
        if NET_LEVEL_FULLY_RANDOM_SUBNETS == False:
            # get temp batch of candidates
            each_blkchoice_class = "default"
            tmp_subnet_batch = generate_subnet_options_alltests(global_settings, supernet_blk_choices, blk_search_options=blk_search_options,
                                                        net_choices = DEFAULT_NET_CHOICES,
                                                        net_prefix = "{}.{}.{}".format(each_blkchoice_class, 0, 0),
                                                        #net_prefix="{}_{}".format(each_net_choice[0], each_net_choice[1]),
                                                        n_samples=NET_LEVEL_TEST_NSAMPLED)            
            v_subnet_batch = list(chain.from_iterable(list(tmp_subnet_batch.values())))     
             
            # get subnets
            i=0
            for each_net_choice_class, lst_netchoices in supernet_choices.items():            
                all_subnets[each_net_choice_class] = []
                j=0
                for each_option in lst_netchoices:    
                    tmp_subnets = []                            
                    
                    cix=0
                    net_prefix = "{}.{}.{}".format(each_net_choice_class, i, j)
                    for sidx, (subnet_name, revised_sidx, blk_search_options, sn_single_choice_per_block, net_choices) in enumerate(v_subnet_batch):
                        
                        subnet_name = "SN-{}-{}-{}-{}".format(net_prefix.upper(), each_blkchoice_class.upper(), cix, sidx)
                        revised_sidx = "{}-{}".format(cix, sidx)        
                        tmp_subnets.append([subnet_name, revised_sidx, blk_search_options, sn_single_choice_per_block, each_option])
                 
                    all_subnets[each_net_choice_class].extend(tmp_subnets)
                    j+=1            
                i+=1
                    
        else:        
            # get subnets
            i=0
            for each_net_choice_class, lst_netchoices in supernet_choices.items():            
                all_subnets[each_net_choice_class] = []
                j=0
                for each_option in lst_netchoices:
                    print(each_option)
                    # subnets = generate_subnets_alltests(supernet_blk_choices, blk_search_options=blk_search_options,
                    #                                     net_choices = each_option,
                    #                                     net_prefix = "{}.{}.{}".format(each_net_choice_class, i, j),
                    #                                     n_samples=2)  # 50 samples per supernet config
                    subnets = generate_subnet_options_alltests(global_settings, supernet_blk_choices, blk_search_options=blk_search_options,
                                                        net_choices = each_option,
                                                        net_prefix = "{}.{}.{}".format(each_net_choice_class, i, j),
                                                        n_samples=NET_LEVEL_TEST_NSAMPLED)  # 50 samples per supernet config
                                        
                    #print([(k,len(v)) for k, v in subnets.items()])                            
                    v_subnets = list(chain.from_iterable(list(subnets.values())))                    
                    #v_subnets_sampled = random.sample(v_subnets, 100)    
                    
                    all_subnets[each_net_choice_class].extend(v_subnets)
                    #print(len(all_subnets[each_net_choice_class]))
                    
                    j+=1            
                i+=1
                            
    
    # -- formulate blk-level testing subnets
    if "BLOCK_LEVEL" in TEST_CHOICE_LEVEL:
        
        print("---- performing BLOCK_LEVEL test -----")
        
        for each_net_choice in TEST_NET_CHOICES_LIST:   
            print(each_net_choice)     
            # get choices
            supernet_blk_choices, blk_search_options = parametric_supernet_blk_choices(tst_type=PARAM_SUPERNET_BLK_CHOICES_TST_TYPE['BLOCK_LEVEL']) 
            #supernet_blk_choices, blk_search_options = parametric_supernet_blk_choices(tst_type="default") 
            # get subnets                    
            subnets = generate_subnet_options_alltests(global_settings, supernet_blk_choices, blk_search_options=blk_search_options, 
                                                net_choices=each_net_choice,
                                                net_prefix="{}_{}".format(each_net_choice[0], each_net_choice[1]),
                                                n_samples=BLOCK_LEVEL_TEST_NSAMPLES       # sample size per test class
                                                )  
            
            # update dict (extend)
            for k, v in subnets.items():
                if k in all_subnets:
                    all_subnets[k].extend(v)
                else:
                    all_subnets[k] = v
            
    
    pprint([(k, len(v)) for k,v in all_subnets.items()])
    print("total=", np.sum([len(v) for k,v in all_subnets.items()])) 
    print("unique_subnet_names ==>")       
    for k,v in all_subnets.items():
        ss_names = [vv[0] for vv in v]        
        pprint(["ss_names", k, len(set(ss_names))])            
        unique_net_choices = set([tuple(vv[-1])  for vv in v])
        pprint(["unique_net_choices", k, len(unique_net_choices)])            
    
    #sys.exit()
    
    # ---- PROCESS SUBNETS (accuracy , hwcost)
    
    # dispatch workers (1 per gpu, each processing a batch of subnets)    
    all_subnets_lst = list(chain.from_iterable(all_subnets.values()))
    #pprint(len(all_subnets_lst)); sys.exit()    
    batched_subnets = np.array_split(all_subnets_lst, len(AVAILABLE_GPUIDS))   
       
    #pprint([[snet.name for snet in sb] for sb in batched_subnets]); sys.exit()       
        
    processes = []
    multiprocessing.set_start_method('spawn')
    for gpuid, each_subnet_batch in zip(AVAILABLE_GPUIDS, batched_subnets):
        p = multiprocessing.Process(target=gpu_worker, args=(gpuid, each_subnet_batch, updated_consts, cmd_args, True))
        processes.append(p)                

    print("Starting processes..")
    for i, p in enumerate(processes):
        p.start()

    # Ensure all of the processes have finished
    print("Joining processes")
    for i, p in enumerate(processes):
        p.join() 
    
    print("\n------ All jobs complete ------\n\n")
    
        
    
