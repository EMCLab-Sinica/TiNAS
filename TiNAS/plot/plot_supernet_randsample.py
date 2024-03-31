import sys, os 
from pprint import pprint
from itertools import cycle

from cycler import cycler

import numpy as np
import random
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

mpl.use('Qt5Agg')

import pandas as pd
import json
from collections import OrderedDict 
from scipy.signal import find_peaks

from scipy.stats import gmean

sys.path.append("..")
from NASBase.file_utils import json_load, file_exists, pickle_dump, pickle_load

from NASBase.model.common_utils import (
    blkchoices_ixs_to_blkchoices, blkchoices_to_blkchoices_ixs, get_network_dimension, get_network_obj, get_subnet, get_supernet, iter_blk_choices, netobj_to_pyobj, 
    get_sampled_subnet_configs
)

from settings import Settings

import plot_common as plt_comm

RESULTS_FILE = "../NASBase/train_log/TiNAS-M-threshold-0.1_supernet_mnas_best.pth_result_load_supernet.json"
LOG_DIR = "../NASBase/train_log/supernet_randsample/"

DATASET = "HAR"

PKL_FNAME = "result_{}.pkl".format(LOG_DIR.split("/")[-2]) 

JSON_FNAME_PREFIX = "result_load_supernet_{}".format(DATASET)

LOAD_PROCESSED_DATA = False

#WIDTH_MULTIPLIER = [0.1]
#INPUT_RESULUTION = [24]


SAMPLE_SIZE = 1000



def get_data(fname):
    exp_data = json_load(fname)
    return exp_data



def get_processed_data():
    pkl_fname = LOG_DIR + PKL_FNAME
    
    if LOAD_PROCESSED_DATA == True:
        # load        
        print("pickle_load: ", pkl_fname)
        all_processed_data = pickle_load(pkl_fname)
        return all_processed_data
        
    else:
        
        if DATASET in ["CIFAR10", "HAR"]:
            net_search_options = {
                'WIDTH_MULTIPLIER' : Settings.NAS_SETTINGS_PER_DATASET[DATASET]['WIDTH_MULTIPLIER'],        
                'INPUT_RESOLUTION' : Settings.NAS_SETTINGS_PER_DATASET[DATASET]['INPUT_RESOLUTION'],    
            }     
            pprint(net_search_options)       
        else:
            sys.exit("Error - invalid DATASET: ", DATASET)
        
        
        
        all_processed_data = {}
        for i, each_wm in enumerate(net_search_options['WIDTH_MULTIPLIER']):
            for j, each_ir in enumerate(net_search_options['INPUT_RESOLUTION']):
                k = "[{},{}]".format(each_wm, each_ir)
                print("==== get_processed_data:: getting data : [{},{}] ======".format(each_wm, each_ir))            
                
                logfname = LOG_DIR + JSON_FNAME_PREFIX + "_{}_{}.json".format(each_wm, each_ir)
                
                if file_exists(logfname):
                    exp_data = get_data(logfname)
                else:
                    continue
        
                # get metric data
                all_latency_intpow_data = []; all_latency_contpow_data = []; all_imo_data = []; all_flops_data = []
                all_vmsize_data = []; all_energy_utilization_data = []
                for s in random.sample(exp_data['lat_results'], SAMPLE_SIZE):                
                
                    all_latency_contpow_data.append(s['perf_e2e_contpow_fp_lat'])
                    all_latency_intpow_data.append(s['perf_e2e_intpow_lat'])
                    all_vmsize_data.append(max(layer['vm'] for layer in s['perf_exec_design_intpow']))
                    all_energy_utilization_data.append(max(layer['Eu'] for layer in s['perf_exec_design_intpow']))
                    all_imo_data.append(s['imc_prop'])
                    all_flops_data.append(s['flops'])                
                
                all_processed_data[k] = {
                    "IMC":all_imo_data,
                    "FLOPS": all_flops_data,
                    "LATENCY_INTPOW": all_latency_intpow_data,
                    "LATENCY_CONTPOW": all_latency_contpow_data,
                    "VMSIZE": all_vmsize_data,
                    "ENERGY_UTILIZATION": all_energy_utilization_data,
                }
        
        # save   
        print("pickle_dump: ", pkl_fname)     
        pickle_dump(pkl_fname, all_processed_data)
        
        return all_processed_data
    
    
    
def plot_fixedsupernet_scatter(wm_lst, ir_lst):
        
    fig, axs = plt.subplots(len(wm_lst), len(ir_lst))        
    
    for i, each_wm in enumerate(wm_lst):
        for j, each_ir in enumerate(ir_lst):
    
            logfname = LOG_DIR + "result_load_supernet_{}_{}.json".format(each_wm, each_ir)
                    
            if file_exists(logfname):
                exp_data = get_data(logfname)
            else:
                return False
            
            
            title = "({}, {})".format(each_wm, each_ir)
            
            # -- real metrics --
            all_metric_data_lat = [s['perf_e2e_contpow_fp_lat'] for s in random.sample(exp_data['lat_results'], SAMPLE_SIZE)]                
            all_metric_data_imc = [s['imc_prop'] for s in random.sample(exp_data['lat_results'], SAMPLE_SIZE)]
            all_metric_data_flops = [s['flops'] for s in random.sample(exp_data['lat_results'], SAMPLE_SIZE)]                
            
            all_data = {
                        "all_subnet_icostprop": all_metric_data_imc,
                        "all_subnet_inte2elat": all_metric_data_lat,   
                        "all_subnet_flops": all_metric_data_flops,    
                        }
            
                
            df1 = pd.DataFrame(data = all_data)  
            df1.info() 
            plt_comm.plt_scatter(df1, 'all_subnet_icostprop', 'all_subnet_flops', 
                                    "Intermittency Cost (%)", "FLOPS",
                                    cmap='viridis', cmap_metric = 'all_subnet_inte2elat', cmap_label = 'Int. Power E2E Latency',
                                    figsize=(4.5,3.5),
                                    fig_title = title,
                                    ax = axs[i,j])
            #axs[i,j].tick_params(labelbottom=True)
    
    # for i, each_wm in enumerate(wm_lst):
    #     plt.setp(axs[i,len(ir_lst)-1].get_xticklabels(), visible=False)
    #     axs[i,-1].tick_params(labelbottom=True)
    
    


def plot_multisupernet_dist_subnet_hist(all_processed_data, metric):
    
    
    
    if DATASET in ["CIFAR10", "HAR"]:
        net_search_options = {
            'WIDTH_MULTIPLIER' : Settings.NAS_SETTINGS_PER_DATASET[DATASET]['WIDTH_MULTIPLIER'],        
            'INPUT_RESOLUTION' : Settings.NAS_SETTINGS_PER_DATASET[DATASET]['INPUT_RESOLUTION'],    
        }            
    else:
        sys.exit("Error - invalid DATASET: ", DATASET)
        
    
    fig, axs = plt.subplots(len(net_search_options['WIDTH_MULTIPLIER']), len(net_search_options['INPUT_RESOLUTION']), sharex=True, sharey=True)    
    fig.suptitle(metric)
    
    min_imo = 100; max_imo = 0
    
    for i, each_wm in enumerate(net_search_options['WIDTH_MULTIPLIER']):
        for j, each_ir in enumerate(net_search_options['INPUT_RESOLUTION']):            
            print("==== plot_multisupernet_dist_subnet_hist:: plotting : [{},{}] ======".format(each_wm, each_ir))            
            
            # logfname = LOG_DIR + "result_load_supernet_{}_{}.json".format(each_wm, each_ir)
            
            # if file_exists(logfname):
            #     exp_data = get_data(logfname)
            # else:
            #     continue
            
            
            # -- real metrics --
            # if metric == 'LATENCY':
            #     #all_metric_data = [s['perf_e2e_intpow_lat'] for s in exp_data['lat_results']]                
            #     all_metric_data = [s['perf_e2e_contpow_fp_lat'] for s in random.sample(exp_data['lat_results'], SAMPLE_SIZE)]                
            # elif metric == 'IMC':                
            #     all_metric_data = [s['imc_prop'] for s in random.sample(exp_data['lat_results'], SAMPLE_SIZE)]
                
            #     if np.min(all_metric_data) < min_imo: min_imo = np.min(all_metric_data)
            #     if np.max(all_metric_data) > max_imo: max_imo = np.max(all_metric_data)
                
            # elif metric == 'FLOPS':
            #     all_metric_data = [s['flops'] for s in random.sample(exp_data['lat_results'], SAMPLE_SIZE)]                
            # elif metric == 'ACCURACY':
            #     sys.exit("plot_multisupernet_dist_subnet_hist:: Error - metric not implemented yet")
                
            # -- param selection distributions -- 
            # elif metric == "AVG_EXPFACT":
            #    all_metric_data = [blk[0] for net in random.sample(exp_data['lat_results'], SAMPLE_SIZE) for blk in net["sn_cpb"]]
            # elif metric == "AVG_KSIZE":
            #    all_metric_data = [blk[1] for net in random.sample(exp_data['lat_results'], SAMPLE_SIZE) for blk in net["sn_cpb"]]
            # elif metric == "AVG_NUMLAYERS":
            #    all_metric_data = [blk[2] for net in random.sample(exp_data['lat_results'], SAMPLE_SIZE) for blk in net["sn_cpb"]]
            # elif metric == "AVG_SKIPENABLE":
            #    all_metric_data = [blk[3] for net in random.sample(exp_data['lat_results'], SAMPLE_SIZE) for blk in net["sn_cpb"]]
            
            # else:
            #     sys.exit("plot_multisupernet_dist_subnet_hist:: Error - metric not implemented yet")
            
            k = "[{},{}]".format(each_wm, each_ir)
            all_metric_data = all_processed_data[k][metric]
                
            
            print("sample size [{},{}] ={}".format(each_wm, each_ir, len(all_metric_data)))
            
            axs[i, j].hist(all_metric_data, bins=50)
            axs[i, j].axvline(gmean(all_metric_data), color='r', linestyle='dashed', linewidth=1)            
            ax2 = axs[i, j].twinx()
            ax2.boxplot(all_metric_data, vert=False)
            axs[i, j].set_title("{}_{}".format(each_wm, each_ir))
            axs[i, j].title.set_color('blue')            
            axs[i, j].tick_params(axis='both', labelsize=9)
            
            print(min_imo, max_imo)
            


def plot_dist_subnet_latencies_histstep(exp_data):
        
    all_latency = [s['perf_e2e_intpow_lat'] for s in exp_data['lat_results']]
    all_imc = [s['imc_prop'] for s in exp_data['lat_results']]
    all_acc = [s['val_acc'] for s in exp_data['acc_results']]
    
    
    fig, ax = plt.subplots(3)
    
    ax[0].hist(all_latency, bins=50)
    ax[0].set_xlabel('Latency')
        
    ax[1].hist(all_acc, bins=50)
    ax[1].set_xlabel('Accuracy')
    
    ax[2].hist(all_imc, bins=50)
    ax[2].set_xlabel('IMC')
    
    
    
    
    
    # # -- selectable legend
    # lined = {}  # Will map legend lines to original lines.
    # for legline, origline in zip(l.get_lines(), lines):
    #     legline.set_picker(True)  # Enable picking on the legend line.
    #     lined[legline] = origline
        
    # def on_pick(event):
    #     # On the pick event, find the original line corresponding to the legend
    #     # proxy line, and toggle its visibility.
    #     legline = event.artist
    #     origline = lined[legline]
    #     visible = not origline.get_visible()
    #     origline.set_visible(visible)
    #     # Change the alpha on the line in the legend, so we can see what lines
    #     # have been toggled.
    #     legline.set_alpha(1.0 if visible else 0.2)
    #     fig.canvas.draw()
    
    
    # fig.canvas.mpl_connect('pick_event', on_pick)
    
            
            
    



if __name__ == '__main__':
    #exp_data = get_data(RESULTS_FILE)
    #plot_dist_subnet_latencies(exp_data)
    #plot_dist_subnet_latencies_violins(exp_data)    
    #plot_dist_subnet_latencies_histstep(exp_data)
    
    all_processed_data = get_processed_data()
    
    plot_multisupernet_dist_subnet_hist(all_processed_data, 'IMC')
    plot_multisupernet_dist_subnet_hist(all_processed_data, 'LATENCY_INTPOW')
    plot_multisupernet_dist_subnet_hist(all_processed_data, 'FLOPS')
    
    # plot_multisupernet_dist_subnet_hist('AVG_EXPFACT')
    # plot_multisupernet_dist_subnet_hist('AVG_KSIZE')
    # plot_multisupernet_dist_subnet_hist('AVG_NUMLAYERS')
    # plot_multisupernet_dist_subnet_hist('AVG_SKIPENABLE')
    
    # scatter
    #plot_fixedsupernet_scatter([1.0, 0.6, 0.2], [32, 30, 28])
    #plot_fixedsupernet_scatter([1.0, 0.2], [32, 28])
    
    
    
    plt.show()

    
    
    


