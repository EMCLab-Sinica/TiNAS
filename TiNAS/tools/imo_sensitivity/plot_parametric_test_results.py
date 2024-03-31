from re import L
import sys, os 
from pprint import pprint
import ntpath
import itertools

import numpy as np
import matplotlib as mpl
from torch import exp_
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#plt.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 12  # Set your desired font size here


from scipy.spatial import ConvexHull
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import splprep, splev

import pandas as pd
import json
from collections import OrderedDict 

import mplcursors

import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append("../..")
from NASBase.file_utils import json_load, file_exists, pickle_dump, pickle_load

from NASBase.hw_cost.Modules_inas_v1.CostModel.cnn import est_FLOPS_cost_layer
import NASBase.hw_cost.Modules_inas_v1.CostModel.common as common

from IMO_sensitivity_analyzer import parametric_supernet_blk_choices, _parse_testclass_key_snname

import plot.plot_common as plt_comm

# blk choice tests
#LOG_DIR = "<ADD_DATA_DIR>"

# net choice tests
LOG_DIR = "<ADD_DATA_DIR>"

LOAD_PROCESSED_DATA = True

GPUID_LIST = np.arange(16)


TEST_CLASSES = [
    "test_convtypes",
    "test_ksizes",
    "test_numlayers",
    "test_supskip",
    "test_netop_widthmult",
    "test_netop_inputres"
]

x_axis_ticks_based_on_test_class = {        
        'test_convtypes' : [1, 2, 3, 4, 6, 8],   
        'test_ksizes': [1, 3, 5, 7, 9],        
        'test_numlayers': [1, 2, 3, 4],                
        'test_supskip' : [False, True],  
        'test_netop_widthmult' : [False, True],  
        'test_netop_inputres' : [28, 32],  
}


# ============================================================================ #


#############################
# HELPERS
#############################
def _label_lst_choice_per_blk(choice_per_blk):    
    s = '_'.join([str(elem) for elem in choice_per_blk])
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace(" ", "")
    s = s.replace(",", "")
    
    #print(s); sys.exit()
    return s


     

#############################
# COLLECT DATA
#############################


def get_data_and_stats():
    
    pkl_fname = LOG_DIR + "result_parametric_test_preprocessed.pkl"    
    
    if LOAD_PROCESSED_DATA == True:
        # load        
        print("pickle_load: ", pkl_fname)
        all_processed_data = pickle_load(pkl_fname)
        return all_processed_data
        
    else:
    
        # get data for all gpus, all subnets executed in each gpu    
        all_stats = {
                    "all_subnet_testclass" : [],
                    "all_subnet_choice_per_blk" : [],
                    "all_subnet_choice_per_blk_lbl" : [],
                    
                    #"all_net_choices" : [],
                    "all_net_widthmult" :[],
                    "all_net_inputres" :[],
                    "all_subnet_acc": [],
                    "all_subnet_icostprop": [],
                    "all_subnet_inte2elat": [],   
                    "all_subnet_numops": [],                 
                    "all_subnet_avgch_per_op": [],
                    "all_subnet_avgksize_per_op": [],
                    "all_subnet_modeksize" : [],
                    "all_subnet_num_skipblks": [],
                    
                    "all_subnet_avg_expfac_per_blk" :[],
                    "all_subnet_avg_ksize_per_blk" :[],
                    "all_subnet_avg_nlayers_per_blk" : [],
                    "all_subnet_avg_skip_per_blk" : [],
                    
                    "all_subnet_flops": [],
                    "all_subnet_intpow_compcost": [],
                    "all_subnet_contpow_compcost": [],
                    "all_subnet_npc": [],
                    "all_subnet_intpow_backup_cost": [],
                    "all_subnet_intpow_recovery_cost": [],     
                    "all_subnet_contpow_nvmwrite_cost": [],
                    "all_subnet_contpow_nvmread_cost": [],       
                    
                    "all_subnet_blk0_expfac": [], 
                    "all_subnet_blk0_ksize": [],
                    "all_subnet_blk0_nl": [],      
                    "all_subnet_blk1_expfac": [],
                    "all_subnet_blk1_ksize": [],
                    "all_subnet_blk1_nl": [],                          
                    }
        
        for each_gpuid in GPUID_LIST:     
            #exp_data = json_load(LOG_DIR + "results_gpuid-{}_results_exp_acc_imc.json".format(str(each_gpuid))) 
            exp_data = json_load(LOG_DIR + "results_gpuid-{}test_results_exp_acc_imc.json".format(str(each_gpuid))) 
                    
            for each_subnet_k, each_subnet_data in exp_data.items():
                
                #print(each_gpuid, each_subnet_k)            
                #get_netx_graph(each_subnet_data['subnet_obj'])
                
                test_class = each_subnet_data["test_class"]
                subnet_choice_per_blk = each_subnet_data["subnet_choice_per_blk"]
                subnet_choice_per_blk_lbl = _label_lst_choice_per_blk(subnet_choice_per_blk)
                net_choices = each_subnet_data["net_choices"]
                
                width_multiplier = net_choices[0]
                input_resolution = net_choices[1]
                #width_multiplier = 1.0
                #input_resolution = 32
                
                acc = each_subnet_data['max_val_acc']*100
                e2e_lat_intpow = each_subnet_data['perf_e2e_intpow_lat']
                int_mng_cost_proportion = plt_comm.get_subnet_icostprop(each_subnet_data)
                numops = plt_comm.get_subnet_num_ops(each_subnet_data)            
                mean_numch_per_op,  min_numch_per_op, max_numch_per_op = plt_comm.get_subnet_avgch_per_op(each_subnet_data)
                
                avg_expfac_per_blk = plt_comm.get_subnet_avg_expf_per_blk(each_subnet_data)
                avg_ksize_per_blk = plt_comm.get_subnet_avg_ksize_per_blk(each_subnet_data)
                avg_layers_per_blk = plt_comm.get_subnet_avg_numlay_per_blk(each_subnet_data)
                avg_skip_per_blk = plt_comm.get_subnet_num_skipen_blks(each_subnet_data)
                
                tot_flops, tot_macs = plt_comm.get_subnet_flops(each_subnet_data)
                tot_intpow_compcost, tot_contpow_compcost = plt_comm.get_subnet_compcost(each_subnet_data)
                tot_npc = plt_comm.get_subnet_numpowcycles(each_subnet_data)            
                tot_intpow_backup_cost, tot_intpow_recovery_cost, tot_contpow_nvmwr_cost, tot_contpow_nvmrd_cost = plt_comm.get_subnet_backuprecovery_cost(each_subnet_data)
                mean_ksize_per_op = plt_comm.get_subnet_avgkernelsize_per_op(each_subnet_data)
                mode_ksize = plt_comm.get_subnet_modekernelsize_per_op(each_subnet_data)
                skip_supp_num_blocks = plt_comm.get_subnet_num_skipsupp_blocks(each_subnet_data)
                            
                all_stats["all_subnet_testclass"].append(test_class)
                all_stats["all_subnet_choice_per_blk"].append(subnet_choice_per_blk)    
                all_stats["all_subnet_choice_per_blk_lbl"].append(subnet_choice_per_blk_lbl)    
                #all_stats["all_net_choices"].append(net_choices)
                all_stats["all_net_widthmult"].append(width_multiplier)
                all_stats["all_net_inputres"].append(input_resolution)            
                all_stats["all_subnet_acc"].append(acc)
                all_stats["all_subnet_icostprop"].append(int_mng_cost_proportion)
                all_stats["all_subnet_inte2elat"].append(e2e_lat_intpow)
                all_stats["all_subnet_numops"].append(numops)            
                all_stats["all_subnet_avgch_per_op"].append(mean_numch_per_op)
                all_stats["all_subnet_avgksize_per_op"].append(mean_ksize_per_op)        
                all_stats["all_subnet_modeksize"].append(mode_ksize)        
                all_stats["all_subnet_num_skipblks"].append(skip_supp_num_blocks)                    
                
                all_stats["all_subnet_avg_expfac_per_blk"].append(avg_expfac_per_blk)        
                all_stats["all_subnet_avg_ksize_per_blk"].append(avg_ksize_per_blk)                                    
                all_stats["all_subnet_avg_nlayers_per_blk"].append(avg_layers_per_blk) 
                all_stats["all_subnet_avg_skip_per_blk"].append(avg_skip_per_blk) 
                
                all_stats["all_subnet_flops"].append(tot_flops)
                all_stats["all_subnet_intpow_compcost"].append(tot_intpow_compcost)   
                all_stats["all_subnet_contpow_compcost"].append(tot_contpow_compcost)   
                all_stats["all_subnet_npc"].append(tot_npc)            
                all_stats["all_subnet_intpow_backup_cost"].append(tot_intpow_backup_cost)            
                all_stats["all_subnet_intpow_recovery_cost"].append(tot_intpow_recovery_cost)        
                all_stats["all_subnet_contpow_nvmwrite_cost"].append(tot_contpow_nvmwr_cost)        
                all_stats["all_subnet_contpow_nvmread_cost"].append(tot_contpow_nvmrd_cost)      
                
                
                # blk level choices
                all_stats["all_subnet_blk0_expfac"].append(plt_comm.get_blk_choice_value(subnet_choice_per_blk, 0, "EXP_FACTORS"))      
                all_stats["all_subnet_blk0_ksize"].append(plt_comm.get_blk_choice_value(subnet_choice_per_blk, 0, "KERNEL_SIZES"))      
                all_stats["all_subnet_blk0_nl"].append(plt_comm.get_blk_choice_value(subnet_choice_per_blk, 0, "NUM_LAYERS"))      
                all_stats["all_subnet_blk1_expfac"].append(plt_comm.get_blk_choice_value(subnet_choice_per_blk, 1, "EXP_FACTORS"))      
                all_stats["all_subnet_blk1_ksize"].append(plt_comm.get_blk_choice_value(subnet_choice_per_blk, 1, "KERNEL_SIZES"))      
                all_stats["all_subnet_blk1_nl"].append(plt_comm.get_blk_choice_value(subnet_choice_per_blk, 1, "NUM_LAYERS"))              
                
        # save   
        print("pickle_dump: ", pkl_fname)     
        pickle_dump(pkl_fname, all_stats)
        
        return all_stats



################ SKIP SUPPORT ######################   

def plot_numskipblks_vs_acc_icostprop(all_data):
    df = pd.DataFrame(data = all_data)      
    df1 = df[df['all_subnet_testclass'].str.contains("test_supskip")] 
    
    param = "all_subnet_avg_skip_per_blk"        #all_subnet_num_skipblks

    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_icostprop', 
                              "# Skip Enable Blocks", "Intermittency Cost (%)",
                              figsize=(5,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_acc', 
                              "# Skip Enable Blocks", "Accuracy (%)",
                              figsize=(5,3), savefig=True)
    
    
def plot_numskipblks_vs_backuprecoverycomp(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_supskip")] 
    
    param = "all_subnet_avg_skip_per_blk"        #all_subnet_num_skipblks
    
    # mean kernel size
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_backup_cost', 
                              "# Skip Enable Blocks", "Total Backup Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_recovery_cost', 
                              "# Skip Enable Blocks", "Total Recovery Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax3 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_compcost', 
                              "# Skip Enable Blocks", "Computation Cost (latency)",
                              figsize=(4,3), savefig=True)



def plot_numskipblks_vs_contpow_nvmrdwrcomp(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_supskip")] 
    
    param = "all_subnet_avg_skip_per_blk"        #all_subnet_num_skipblks
    
    # mean kernel size
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_nvmwrite_cost', 
                              "# Skip Enable Blocks", "Total NVM Write Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_nvmread_cost', 
                              "# Skip Enable Blocks", "Total NVM Read Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax3 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_compcost', 
                              "# Skip Enable Blocks", "Computation Cost (latency)",
                              figsize=(4,3), savefig=True)


################ NUM LAYERS ######################

def plot_numlayers_vs_acc_icostprop(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)      
    df1 = df[df['all_subnet_testclass'].str.contains("test_numlayers")] 
    
    #pd.set_option('display.max_columns', None)
    #print(df.head(10)); sys.exit()
    #df1.info(); sys.exit()
    
    param = "all_subnet_numops" # "all_subnet_avg_nlayers_per_blk"        #all_subnet_numops
    xlbl = "# Layers per block (mean)"      # "# Layers (total)"
    
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_icostprop', 
                              xlbl, "Intermittency Cost (%)",
                              figsize=(5,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_acc', 
                              xlbl, "Accuracy (%)",
                              figsize=(5,3), savefig=True)


def plot_numlayers_vs_backuprecoverycomp(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_numlayers")] 
    
    param = "all_subnet_numops" # "all_subnet_avg_nlayers_per_blk"        #all_subnet_numops
    xlbl = "# Layers per block (mean)"      # "# Layers (total)"
        
    # mean kernel size
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_backup_cost', 
                              xlbl, "Total Backup Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_recovery_cost', 
                              xlbl, "Total Recovery Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax3 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_compcost', 
                              xlbl, "Computation Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    

def plot_numlayers_vs_contpow_nvmrdwrcomp(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_numlayers")] 
    
    param = "all_subnet_numops" # "all_subnet_avg_nlayers_per_blk"        #all_subnet_numops
    xlbl = "# Layers per block (mean)"      # "# Layers (total)"
    
    # mean kernel size
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_nvmwrite_cost', 
                              xlbl, "Total NVM Write Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_nvmread_cost', 
                              xlbl, "Total NVM Read Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax3 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_compcost', 
                              xlbl, "Computation Cost (latency)",
                              figsize=(4,3), savefig=True)



################ CONVTYPE ###################### 

def plot_convtype_vs_acc_icostprop_colored(all_data):
    df = pd.DataFrame(data = all_data)      
    df1 = df[df['all_subnet_testclass'].str.contains("test_convtypes")] 
    #print(df.head(10)); sys.exit()
    #df1.info(); sys.exit()
    
    xlbl = "Exp. Factor per block (mean)"     
    ylbl =  "Intermittency Cost (%)"       
    
    # -- colored markers
    col_map = {1: '#377eb8', 2: '#ff7f00', 3: '#e41a1c', 4: '#4daf4a', 6: '#636363', 8: '#636363' }
    cols = list(df1['all_subnet_blk0_expfac'].map(col_map)) 
    
        
    x_data = df1['all_subnet_avg_expfac_per_blk']    
    x_data_blk0_expfac = list(df1['all_subnet_blk0_expfac'])
    y_data = df1['all_subnet_icostprop']
    
    
    fig, ax = plt.subplots()
    for x,y,c,b0_expfac in zip(x_data, y_data, cols, x_data_blk0_expfac):           
        ax.scatter(x,y,color=c, marker='x')        
        
    
    # only plot the linreg for blk0=[1,2]
    for blk0_expfac in [1,2,3,4]:
        xx = df1[df1['all_subnet_blk0_expfac'] == blk0_expfac]['all_subnet_avg_expfac_per_blk']
        yy = df1[df1['all_subnet_blk0_expfac'] == blk0_expfac]['all_subnet_icostprop']
        
        m, b = np.polyfit(xx, yy, 1)
        ax.plot(xx, m*xx+b,color=col_map[blk0_expfac])
        
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)        
    ax.grid(True)
        
        


def plot_convtype_vs_acc_icostprop(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)      
    df1 = df[df['all_subnet_testclass'].str.contains("test_convtypes")] 
    #print(df.head(10)); sys.exit()
    #df1.info(); sys.exit()
    param = "all_subnet_avg_expfac_per_blk" # all_subnet_avgch_per_op
    xlbl = "Exp. Factor per block (mean)"      # "# OFM channels (mean)"
       
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_icostprop', 
                              xlbl, "Intermittency Cost (%)",
                              figsize=(5,3), savefig=True,
                              )
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_acc', 
                              xlbl, "Accuracy (%)",
                              figsize=(5,3), savefig=True)
    
    #mplcursors.cursor(ax2).connect("add", lambda sel: sel.annotation.set_text(df1.iloc[sel.target.index]['all_subnet_choice_per_blk']))


def plot_convtype_vs_backuprecoverycomp(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_convtypes")] 
    
    param = "all_subnet_avg_expfac_per_blk" # all_subnet_avgch_per_op
    xlbl = "Exp. Factor per block (mean)"      # "# OFM channels (mean)"

    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_backup_cost', 
                                    xlbl, "Total Backup Cost (latency)",
                                    figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_recovery_cost', 
                                    xlbl, "Total Recovery Cost (latency)",
                                    figsize=(4,3), savefig=True)
    
    ax3, lines = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_compcost', 
                                    xlbl, "Computation Cost (latency)",
                                    figsize=(4,3), savefig=True)
    
    
    # mplcursors.cursor(ax1).connect("add", lambda sel: sel.annotation.set_text(df1.iloc[sel.target.index]['all_subnet_choice_per_blk']))
    # mplcursors.cursor(ax2).connect("add", lambda sel: sel.annotation.set_text(df1.iloc[sel.target.index]['all_subnet_choice_per_blk']))
    #mplcursors.cursor(ax3).connect("add", lambda sel: sel.annotation.set_text(df1.iloc[sel.target.index]['all_subnet_choice_per_blk']))

    

    
    
def plot_convtype_vs_contpow_nvmrdwr(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_convtypes")] 
    
    param = "all_subnet_avg_expfac_per_blk" # all_subnet_avgch_per_op
    xlbl = "Exp. Factor per block (mean)"      # "# OFM channels (mean)"

    # mean kernel size
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_avgch_per_op', 'all_subnet_contpow_nvmwrite_cost', 
                              xlbl, "Total NVM Write Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, 'all_subnet_avgch_per_op', 'all_subnet_contpow_nvmread_cost', 
                              xlbl, "Total NVM Read Cost (latency)",
                              figsize=(4,3), savefig=True)


            
################ KERNEL SIZE ######################

def plot_ksizes_vs_backuprecoverycomp_per_op(all_data):
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_ksizes")]
        
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_choice_per_blk', 'all_subnet_intpow_backup_cost', 
                              "Kernel Size (mean)", "Total Backup Cost (latency)",
                              figsize=(5,3), savefig=False) 
    ax1.set_xticklabels(df1['all_subnet_choice_per_blk'], rotation = 30, fontsize=7)
    
    
def plot_ksizes_vs_contpow_nvmrdwrcomp(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_ksizes")] 

    param = "all_subnet_avg_ksize_per_blk"   # all_subnet_avgksize_per_op
    xlbl = "Kernel Size per block (mean)"

    # mean kernel size
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_nvmwrite_cost', 
                              xlbl, "Total NVM Write Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_nvmread_cost', 
                              xlbl, "Total NVM Read Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax3 = plt_comm.plt_scatter(df1, param, 'all_subnet_contpow_compcost', 
                              xlbl, "Total Computation Cost (latency)",
                              figsize=(4,3), savefig=True)

    
def plot_ksizes_vs_backuprecoverycomp(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_ksizes")] 
    
    param = "all_subnet_avg_ksize_per_blk"   # all_subnet_avgksize_per_op, all_subnet_modeksize
    xlbl = "Kernel Size per block (mean)"       # Kernel Size (mode)
    
    # mean kernel size
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_backup_cost', 
                              xlbl, "Total Backup Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_recovery_cost', 
                              xlbl, "Total Recovery Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax3 = plt_comm.plt_scatter(df1, param, 'all_subnet_intpow_compcost', 
                              xlbl, "Total Computation Cost (latency)",
                              figsize=(4,3), savefig=True)
    

def plot_ksizes_vs_acc_icostprop(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)      
    df1 = df[df['all_subnet_testclass'].str.contains("test_ksizes")] 
    #print(df.head(10)); sys.exit()
    #df1.info(); sys.exit()
    param = "all_subnet_avg_ksize_per_blk"   # all_subnet_avgksize_per_op, all_subnet_modeksize
    xlbl = "Kernel Size per block (mean)"       # Kernel Size (mode)
       
    ax1 = plt_comm.plt_scatter(df1, param, 'all_subnet_icostprop', 
                              xlbl, "Intermittency Cost (%)",
                              figsize=(5,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, param, 'all_subnet_acc', 
                              xlbl, "Accuracy (%)",
                              figsize=(5,3), savefig=True)

    #mplcursors.cursor(ax1).connect("add", lambda sel: sel.annotation.set_text(df1.iloc[sel.target.index]['all_subnet_choice_per_blk']))



    

def plot_ksizes_vs_acc_icostprop_colored(all_data):
    ksizes = [1, 3, 5, 7, 9]    
    #col_map = {1: '#377eb8', 3: '#e41a1c', 5: '#4daf4a', 7: '#ff7f00', 9: '#984ea3' }   
    #col_map = {1: '#377eb8', 3: '#e41a1c', 5: '#4daf4a', 7: '#ff7f00', 9: '#969696' }   # <-- ok multicol
    
    col_map = {9: '#000000', 7: '#08306b', 5: '#2171b5', 3: '#4292c6', 1: '#a6bddb' }   # <-- blues
    
    
    
    #col_map = {1: '#377eb8', 3: '#ff7f00', 5: '#636363', 7: '#ff7f00', 9: '#377eb8' }   
    #col_map = {1: '#377eb8', 3: '#abd9e9', 5: '#969696', 7: '#ffffbf', 9: '#ff7f00' }   
    
    #marker_lst = ['x', 'o', 's', "^", "*"]
    marker_lst = ['v', 's', '^', "o", "D"]
    marker_map = {k: marker_lst[i] for i,k in enumerate(ksizes)}
    
    def _cluster_silhouette_score(df, ymetric, xmetric, ksize_metric):    
        y_data = df[ymetric]
        x_data = df[xmetric]
        dataset = pd.DataFrame({'y_data': y_data, 'x_data': x_data}, columns=['y_data', 'x_data'])
        ksize_cluster = df[ksize_metric]    
        ss = silhouette_score(dataset, ksize_cluster)
        return ss

    def _plot(blkid, ax, ymetric):
        if blkid==0:
            param = "all_subnet_blk0_ksize"
        elif blkid==1:
            param = "all_subnet_blk1_ksize"
            
        # -- colored markers
        
        #col_map = {1: '#377eb8', 3: '#ff7f00', 5: '#e41a1c', 7: '#4daf4a', 9: '#636363' }
        
        #col_map = {1: '#08519c', 3: '#3182bd', 5: '#6baed6', 7: '#bdd7e7', 9: '#eff3ff' }   # blues
        
        
        
        #col_map = {k: sns.dark_palette("blue", len(ksizes)).as_hex()[i] for i,k in enumerate(ksizes)} 
        cols = list(df1[param].map(col_map)) 
         
        markers = list(df1[param].map(marker_map)) 
        
        x_data = df1['all_subnet_avg_ksize_per_blk']    
        x_data_blk_ksize = list(df1[param])
        y_data = df1[ymetric]        
        
        for x,y,c,m,blk_ksize in zip(x_data, y_data, cols, markers, x_data_blk_ksize):           
            ax.scatter(x,y,color=c, marker=m, alpha=1.0, s=25, edgecolor='none',zorder=2)       
            
            #pprint([x,y])
            
        ax.grid(color='gray', linestyle='dashed', axis='both')         
        ax.set_axisbelow(True)        
        
        
        
        #ax.set_axisbelow(True)
        #ax.grid(False)
                    
        # plot the linreg
        ll = []; labels = []
        for i,blk_ksize in enumerate([1, 3, 5, 7, 9]):
            xx = df1[df1[param] == blk_ksize]['all_subnet_avg_ksize_per_blk']
            yy = df1[df1[param] == blk_ksize][ymetric]            
            m, b = np.polyfit(xx, yy, 1)
            #l, = ax.plot(xx, m*xx+b,color=col_map[blk_ksize], label="K={}".format(blk_ksize))
            #ll.append(l)
            labels.append("K={}".format(blk_ksize))
            
            # Compute the convex hull
            points = np.column_stack((xx, yy))
            hull = ConvexHull(points)
            
            
            # xtmp = ?
            # ytmp = ?
            # xtmp.append(x[0])
            # y.append(y[0])

            # tck, _ = splprep([x, y], s = 0, per = True)
            # xx, yy = splev(np.linspace(0, 1, 100), tck, der = 0)
            # ax.plot
            
            all_xpoints = [point[0] for point in points[hull.vertices]]
            all_ypoints = [point[1] for point in points[hull.vertices]]
            
            print("--")
            pprint([all_xpoints, all_ypoints])
            print("--")
            all_xpoints.append(all_xpoints[0])
            all_ypoints.append(all_ypoints[0])
            tck, _ = splprep([all_xpoints, all_ypoints], s = 0, per = True)
            xx_tmp, yy_tmp = splev(np.linspace(0, 1, 100), tck, der = 0)
            ax.fill(xx_tmp, yy_tmp, '--', linewidth=0.5, color=col_map[blk_ksize], alpha=0.2,zorder=1)
            
            # Plot the convex hull
            # for simplex in hull.simplices:
                
            #     print("---")
            #     print(points[simplex, 0], points[simplex, 1])
            #     print("---")
                
            #     ax.plot(points[simplex, 0], points[simplex, 1], '--', linewidth=0.5, color=col_map[blk_ksize])
            
            
        #ax.grid(True)
        ax.set_xticks([1,3,5,7,9])
        ax.set_xticklabels([str(x) for x in [1,3,5,7,9]])
        
        return ll, labels
    
    df = pd.DataFrame(data = all_data)      
    df1 = df[df['all_subnet_testclass'].str.contains("test_ksizes")] 
    #print(df.head(10)); sys.exit()
    #df1.info(); sys.exit()
    
    xlbl = "Kernel Size (block-wise avg.)"     
    ylbl =  "IMO (%)"     
    
    #fig, axes = plt.subplots(2,2, sharex='col', sharey='row')  # r, c
    fig, axes = plt.subplots(1,2, sharex='col', sharey='row')  # r, c
    
    # imo
    legs, labels = _plot(0, axes[0], 'all_subnet_icostprop')
    print("Silhouette Score (blk0_ksize): {}".format(_cluster_silhouette_score(df1, 'all_subnet_icostprop', 'all_subnet_avg_ksize_per_blk', "all_subnet_blk0_ksize")))
    legs, labels = _plot(1, axes[1], 'all_subnet_icostprop')
    print("Silhouette Score (blk1_ksize): {}".format(_cluster_silhouette_score(df1, 'all_subnet_icostprop', 'all_subnet_avg_ksize_per_blk', "all_subnet_blk1_ksize")))
    
    # accuracy
    #legs, labels = _plot(0, axes[0,0], 'all_subnet_acc')
    #legs, labels = _plot(1, axes[0,1], 'all_subnet_acc')
    
        
    axes[0].set_xlabel(xlbl); 
    axes[1].set_xlabel(xlbl)    
    
    
    axes[0].set_ylabel(ylbl)        
    #axes[0].set_ylabel(ylbl)        
    
    
    #axes[0,0].grid(True); axes[0,0].grid(True); axes[0,0].grid(True);
    #axes[1].grid(True);
    
    def _create_dummy_line(**kwds):
        return mlines.Line2D([], [], **kwds)
    # Create the legend
    lines = [
        ('K=1', {'color': col_map[1], 'linestyle': '-', 'marker': marker_map[1]}),
        ('K=3', {'color': col_map[3], 'linestyle': '-', 'marker': marker_map[3]}),
        ('K=5', {'color': col_map[5], 'linestyle': '-', 'marker': marker_map[5]}),
        ('K=7', {'color': col_map[7], 'linestyle': '-', 'marker': marker_map[7]}),
        ('K=9', {'color': col_map[9], 'linestyle': '-', 'marker': marker_map[9]}),
    ]
    l=fig.legend(
        # Line handles
        [_create_dummy_line(**l[1]) for l in lines],
        # Line titles
        [l[0] for l in lines],
        loc='upper center',
        ncol=len(labels)
    )
    
    
    # l = fig.legend(handles=legs, 
    #        loc="upper center", ncol=len(labels)) 
    l.set_draggable(True)



################ WIDTH MULTIPLIER ######################  

def plot_widthmult_vs_acc_icostprop_fixed_samples(all_data):
    df = pd.DataFrame(data = all_data)          
    df1 = df[df['all_subnet_testclass'].str.contains("test_netop_widthmult")] 
    
    fixed_samples = [
        "632True_332True_111False_612True",
        "672False_333True_612True_353True",
        "112True_172True_312False_113False",
        "172False_373False_653False_111True",
    ]

    #colors = sns.rainbow(np.linspace(0, 1, len(ys)))
    sns_cols = sns.color_palette("bright", len(fixed_samples))
    
    #print(df1.head(10)); sys.exit()
    #df1.info(); sys.exit()
    #sys.exit()
#.isin
    wm_unqiue = sorted(df1['all_net_widthmult'].unique())
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    
    for i, each_sample in enumerate(fixed_samples):
        all_imo = []
        all_acc = []
        for wm in wm_unqiue:
            imo = df1[(df1["all_net_widthmult"] == wm) & (df1["all_subnet_choice_per_blk_lbl"] == each_sample)]["all_subnet_icostprop"].values
            acc = df1[(df1["all_net_widthmult"] == wm) & (df1["all_subnet_choice_per_blk_lbl"] == each_sample)]["all_subnet_acc"].values
            
            all_imo.append(imo)
            all_acc.append(acc)
            
        ax1.plot(wm_unqiue, all_imo, color=sns_cols[i], marker='x', label=each_sample)
        l=ax1.legend()
        l.set_draggable(True)
            
            #print([each_sample, wm, imo, acc])
            
            


def plot_widthmult_vs_acc_icostprop(all_data):
    
    df = pd.DataFrame(data = all_data)      
    #df1 = df[df['all_subnet_testclass'] == "test_netop_widthmult"] 
    df1 = df[df['all_subnet_testclass'].str.contains("test_netop_widthmult")] 
    
    
    
    #
    
    df2 = df1[(df1["all_net_widthmult"]==0.2) & (df1["all_net_inputres"]==32)]["all_subnet_choice_per_blk_lbl"]
    pprint(df2); sys.exit()
    
    #print(df.head(10)); sys.exit()
    #df1.info(); sys.exit()
    #x= df1.loc[:,['all_net_widthmult', 'all_subnet_icostprop', 'all_subnet_acc', 'all_subnet_testclass']].head(); print(x); sys.exit()
       
    ax1 = plt_comm.plt_scatter(df1, 'all_net_widthmult', 'all_subnet_icostprop', 
                              "Width Multiplier", "Intermittency Cost (%)",
                              figsize=(5,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, 'all_net_widthmult', 'all_subnet_acc', 
                              "Width Multiplier", "Accuracy (%)",
                              figsize=(5,3), savefig=True)


def plot_widthmult_vs_backuprecoverycomp(all_data):    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_netop_widthmult")] 
    
    ax1 = plt_comm.plt_scatter(df1, 'all_net_widthmult', 'all_subnet_intpow_backup_cost', 
                              "Width Multiplier", "Total Backup Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, 'all_net_widthmult', 'all_subnet_intpow_recovery_cost', 
                              "Width Multiplier", "Total Recovery Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax3 = plt_comm.plt_scatter(df1, 'all_net_widthmult', 'all_subnet_intpow_compcost', 
                              "Width Multiplier", "Total Computation Cost (latency)",
                              figsize=(4,3), savefig=True)



################ INPUT RESOLUTION ######################  

def plot_inputres_vs_acc_icostprop(all_data):
    
    df = pd.DataFrame(data = all_data)      
    #df1 = df[df['all_subnet_testclass'] == "test_netop_inputres"] 
    df1 = df[df['all_subnet_testclass'].str.contains("test_netop_inputres")] 
    
    #print(df.head(10)); sys.exit()
    #df1.info(); sys.exit()
    #x= df1.loc[:,['all_net_inputres', 'all_subnet_icostprop', 'all_subnet_acc', 'all_subnet_testclass']].head(); print(x); sys.exit()
       
    ax1 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_icostprop', 
                              "Input Resolution", "Intermittency Cost (%)",
                              figsize=(5,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_acc', 
                              "Input Resolution", "Accuracy (%)",
                              figsize=(5,3), savefig=True)


def plot_inputres_vs_backuprecoverycomp(all_data):    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_netop_inputres")] 
    
    ax1 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_intpow_backup_cost', 
                              "Input Resolution", "Total Backup Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_intpow_recovery_cost', 
                              "Input Resolution", "Total Recovery Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax3 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_intpow_compcost', 
                              "Input Resolution", "Total Computation Cost (latency)",
                              figsize=(4,3), savefig=True)
    

def plot_inputres_vs_contpow_nvmrdwrcomp(all_data):
    # plot mem vs. acc    
    df = pd.DataFrame(data = all_data)  
    df1 = df[df['all_subnet_testclass'].str.contains("test_netop_inputres")]
    
    # mean kernel size
    ax1 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_contpow_nvmwrite_cost', 
                              "Input Resolution", "Total NVM Write Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    ax2 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_contpow_nvmread_cost', 
                              "Input Resolution", "Total NVM Read Cost (latency)",
                              figsize=(4,3), savefig=True)
    
    # ax3 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_contpow_compcost', 
    #                           "Input Resolution", "Computation Cost (latency)",
    #                           figsize=(4,3), savefig=True)




########################################################
################### COMBINED ###########################

 
def plot_combined_acc_imo(all_data):

    TESTCLASSES = ["test_ksizes", "test_convtypes", "test_numlayers", "test_supskip", "test_netop_widthmult", "test_netop_inputres"]
    TESTCLASSES_LBL = ["Kernel Size\n(block-wise avg.)", "Expansion Factor\n(block-wise avg.)", "# Layers\n(block-wise avg.)", "# Skip Enable\n(block-wise avg.)", "Width Multiplier", "Input Resolution"]
    METRICS = ["all_subnet_avg_ksize_per_blk", "all_subnet_avg_expfac_per_blk", "all_subnet_avg_nlayers_per_blk", "all_subnet_avg_skip_per_blk",
               "all_net_widthmult", "all_net_inputres"]
    
    ANNOTATE_RANGE_ENABLE_TESTCLASSES = ["test_ksizes", "test_convtypes", "test_numlayers", "test_supskip"]
    ANNOTATE_RANGEAVG_ENABLE_TESTCLASSES = ["test_netop_widthmult", "test_netop_inputres"]
    
    REGPLOT_ORDER_PER_TESTCLASS = [2, 2, 1, 1, None, None]
    
    fig, axs = plt.subplots(2, len(TESTCLASSES), sharex='col')
        
    
    df = pd.DataFrame(data = all_data)    
    
    # -- accuracy (top) --    
      
    # df_ks = df[df['all_subnet_testclass'].str.contains("test_ksizes")]   
    # df_ef = df[df['all_subnet_testclass'].str.contains("test_convtypes")]   
    # df_nl = df[df['all_subnet_testclass'].str.contains("test_numlayers")]   
    # df_sk = df[df['all_subnet_testclass'].str.contains("test_supskip")]   
    # df_wm = df[df['all_subnet_testclass'].str.contains("test_netop_widthmult")]   
    # df_ir = df[df['all_subnet_testclass'].str.contains("test_netop_inputres")]  
    
    # df_lst = [df_ks, df_ef, df_nl, df_sk, df_wm, df_ir] 
    scatter_col = '#2171b5'
    
    for i, tstclass in enumerate(TESTCLASSES):
        df1 = df[df['all_subnet_testclass'].str.contains(tstclass)]   
        
        # Accuracy
        plt_comm.plt_scatter(df1, METRICS[i], 'all_subnet_acc', 
                              TESTCLASSES_LBL[i], "Accuracy (%)",
                              ax=axs[0,i], col = scatter_col, alpha=0.5)    
        #axs[0,i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))    

        # IMO %
        plt_comm.plt_scatter(df1, METRICS[i], 'all_subnet_icostprop', 
                              TESTCLASSES_LBL[i], "IMO (%)",
                              ax=axs[1,i], col = scatter_col, alpha=0.5,
                              order=REGPLOT_ORDER_PER_TESTCLASS[i],
                              show_avg=True if tstclass in ["test_netop_widthmult", "test_netop_inputres"] else False)
        #if tstclass not in ["test_supskip"]:
        #    axs[1,i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))   
        
        axs[1,i].grid(True, color='gray', linestyle='dashed')
        axs[1,i].set_axisbelow(True)    
        axs[0,i].grid(True, color='gray', linestyle='dashed')
        axs[0,i].set_axisbelow(True)  
        
        
        # set xlim
        if tstclass in ["test_ksizes", "test_convtypes", "test_numlayers", "test_netop_inputres"]:
            xrange = x_axis_ticks_based_on_test_class[tstclass]
            axs[1,i].set_xticks(xrange)
            axs[1,i].set_xticklabels(xrange)
        elif tstclass in ["test_supskip"]:
            pass
        
        elif tstclass in ["test_netop_widthmult"]:    
            pass
        
        elif tstclass in ["test_netop_widthmult"]:    
            pass
        
        
        if i>0:
            y_axis = axs[0,i].axes.get_yaxis().set_label_text('')   # acc
            y_axis = axs[1,i].axes.get_yaxis().set_label_text('')   # imo
    
        axs[0,i].axes.get_xaxis().set_label_text('')
        
        
        # titles
        #fig.suptitle("block-level parameters", x=0.2)
        
        arrow_top_offset = 1.0
        arrow_bottom_offset = 1.0
        
        # -- annotate range --
        if tstclass in ANNOTATE_RANGE_ENABLE_TESTCLASSES:
            y_acc_minmax = [np.min(df1['all_subnet_acc']), np.max(df1['all_subnet_acc'])]
            y_imo_minmax = [np.min(df1['all_subnet_icostprop']), np.max(df1['all_subnet_icostprop'])]
            #print(METRICS[i], y_imo_minmax); sys.exit()
            #xaxis_mid = np.mean(df1[METRICS[i]])
            y_mid = (y_imo_minmax[0]+ 
                    (y_imo_minmax[1]-y_imo_minmax[0])/2)            
            yrange = round(y_imo_minmax[1]-y_imo_minmax[0])
            yrange_txt = "{}%".format(            
                yrange if yrange>1 else round(y_imo_minmax[1]-y_imo_minmax[0],1)
            )
            axs[1,i].annotate('', xy=(0.1, y_imo_minmax[0]*arrow_top_offset), 
                                  xytext=(0.1, y_imo_minmax[1]*arrow_bottom_offset), 
                                  xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'),
                                  arrowprops=dict(arrowstyle='<|-|>', color='r', linewidth=1.5, shrinkA=0.0, shrinkB=0.0, joinstyle='miter')
                             )
            t = axs[1,i].annotate('{}'.format(yrange_txt), xy=(0.15, y_mid), xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'), color='r')
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
        
        if tstclass in ANNOTATE_RANGEAVG_ENABLE_TESTCLASSES:
            round_by=2
            df1.sort_values(by=METRICS[i], inplace=True)
            bins = df1.groupby(df1[METRICS[i]].round(round_by)).mean()            
            pprint(bins["all_subnet_icostprop"])
            y_avg_minmax = [np.min(bins["all_subnet_icostprop"]), np.max(bins["all_subnet_icostprop"])]
            pprint(y_avg_minmax)
            y_mid = (y_avg_minmax[0]+
                     (y_avg_minmax[1]-y_avg_minmax[0])/2)
            yrange = round(y_avg_minmax[1]-y_avg_minmax[0])            
            yrange_txt = "{}%".format(            
                yrange if yrange>1.1 else int(round(y_avg_minmax[1]-y_avg_minmax[0],1))
            )
            if yrange > 3:
                arrow_style = dict(arrowstyle='<|-|>', color='r', linewidth=1.5, shrinkA=0.0, shrinkB=0.0, joinstyle='miter')
            else:
                arrow_style = dict(arrowstyle='|-|', color='r', linewidth=1.5, shrinkA=0.0, shrinkB=0.0, mutation_scale=1)
            
            axs[1,i].annotate('', xy=(0.1, y_avg_minmax[0]), 
                                    xytext=(0.1, y_avg_minmax[1]), 
                                    xycoords=('axes fraction', 'data'), 
                                    textcoords=('axes fraction', 'data'),
                                    arrowprops=arrow_style
                                )
            
                
            t = axs[1,i].annotate('{}'.format(yrange_txt), xy=(0.15, y_mid), xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'), color='r')
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
            
    
    # -- IMO (bottom) --
    

    # df = pd.DataFrame(data = all_data)      
    # #df1 = df[df['all_subnet_testclass'] == "test_netop_inputres"] 
    # df1 = df[df['all_subnet_testclass'].str.contains("test_netop_inputres")] 
    
    # #print(df.head(10)); sys.exit()
    # #df1.info(); sys.exit()
    # #x= df1.loc[:,['all_net_inputres', 'all_subnet_icostprop', 'all_subnet_acc', 'all_subnet_testclass']].head(); print(x); sys.exit()
       
    # ax1 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_icostprop', 
    #                           "Input Resolution", "Intermittency Cost (%)",
    #                           figsize=(5,3), savefig=True)
    
    # ax2 = plt_comm.plt_scatter(df1, 'all_net_inputres', 'all_subnet_acc', 
    #                           "Input Resolution", "Accuracy (%)",
    #                           figsize=(5,3), savefig=True)






if __name__ == '__main__':
    
    all_stats = get_data_and_stats()
    
    # == BLOCK LEVEL TESTS ==
    
    # -- kernel sizes
    # plot_ksizes_vs_acc_icostprop(all_stats)
    # plot_ksizes_vs_backuprecoverycomp(all_stats)
    # plot_ksizes_vs_contpow_nvmrdwrcomp(all_stats)
    
    #plot_ksizes_vs_acc_icostprop_colored(all_stats)    # <---
    
        
    # -- conv type (expansion factor)
    # plot_convtype_vs_acc_icostprop(all_stats)    
    # plot_convtype_vs_backuprecoverycomp(all_stats)
    # plot_convtype_vs_contpow_nvmrdwr(all_stats)
    
    #plot_convtype_vs_acc_icostprop_colored(all_stats)    
    
    # -- number of layers
    # plot_numlayers_vs_acc_icostprop(all_stats)
    # plot_numlayers_vs_backuprecoverycomp(all_stats)
    # plot_numlayers_vs_contpow_nvmrdwrcomp(all_stats)
    
    # -- skip enabled
    #plot_numskipblks_vs_acc_icostprop(all_stats)
    #plot_numskipblks_vs_backuprecoverycomp(all_stats)
    #plot_numskipblks_vs_contpow_nvmrdwrcomp(all_stats)
    
    # == BLOCK LEVEL TESTS ==
    
    # -- width multiplier
    #plot_widthmult_vs_acc_icostprop_fixed_samples(all_stats)
    #plot_widthmult_vs_acc_icostprop(all_stats)
    #plot_widthmult_vs_backuprecoverycomp(all_stats)
    
    # -- input resolution
    # plot_inputres_vs_acc_icostprop(all_stats)
    # plot_inputres_vs_backuprecoverycomp(all_stats)
    # plot_inputres_vs_contpow_nvmrdwrcomp(all_stats)    
    
    
    # == COMBINED PLOT ==
    plot_combined_acc_imo(all_stats)        # <---
    
    
    plt.show()    