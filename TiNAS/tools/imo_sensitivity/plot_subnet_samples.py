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

import pandas as pd
import json
from collections import OrderedDict 
from matplotlib.lines import Line2D

import seaborn as sns

sys.path.append("../..")
from NASBase.file_utils import json_load, file_exists, pickle_dump, pickle_load

from NASBase.hw_cost.Modules_inas_v1.CostModel.cnn import est_FLOPS_cost_layer
import NASBase.hw_cost.Modules_inas_v1.CostModel.common as common

LOG_DIR = "<ADD_DATA_DIR>"

WORKERID_LIST = np.arange(48)
GPUID_LIST = np.arange(10)

import plot_common as plt_comm


LOAD_PROCESSED_DATA = True

EXP_SUFFIX = "_sec3_"
    
FILTER_NET_CHOICES = [
       [0.2, 32], [0.2, 28],
       [1.0, 32], [1.0, 28],
       [0.2, 28]
    
    #[1.0, 32], [0.2, 28],
]

def get_data_and_stats():
    
    pkl_fname = LOG_DIR + "result_test_subnet_data_preprocessed.pkl"    
    
    if LOAD_PROCESSED_DATA == True:
        # load        
        print("pickle_load: ", pkl_fname)
        all_processed_data = pickle_load(pkl_fname)
        return all_processed_data
        
    else:
    
    
        # get data for all gpus, all subnets executed in each gpu    
        all_data = []
        all_stats = {"all_subnet_acc": [],
                    "all_subnet_icostprop": [],
                    "all_subnet_inte2elat": [],   
                    "all_subnet_numops": [],
                    "all_subnet_avgch_per_op": [],
                    "all_subnet_avgksize_per_op": [],
                    "all_subnet_flops": [],
                    "all_subnet_compcost": [],
                    "all_subnet_npc": [],
                    "all_subnet_backup_cost": [],
                    "all_subnet_recovery_cost": [],
                                    
                    "all_subnet_widthmul" : [], "all_subnet_inputres" : [],                  
                                                                    
                    }
        
        for each_wid in GPUID_LIST:    
                
            json_fname = "results_gpuid-{}_results_exp_acc_imc.json".format(str(each_wid)+EXP_SUFFIX)
            exp_data = json_load(LOG_DIR + json_fname) 
            print(json_fname)  
            
                    
            for each_subnet_k, each_subnet_data in exp_data.items():
                
                print(each_subnet_k)  
                
                acc = round(each_subnet_data['max_val_acc']*100, 2)
                e2e_lat_intpow = each_subnet_data['perf_e2e_intpow_lat']
                int_mng_cost_proportion_recalc = plt_comm.get_subnet_icostprop(each_subnet_data)
                int_mng_cost_proportion = each_subnet_data['imc_prop']
                numops = plt_comm.get_subnet_num_ops(each_subnet_data)
                mean_numch_per_op,  min_numch_per_op, max_numch_per_op = plt_comm.get_subnet_avgch_per_op(each_subnet_data)
                tot_flops, tot_macs = plt_comm.get_subnet_flops(each_subnet_data)
                tot_compcost = plt_comm.get_subnet_compcost(each_subnet_data)
                tot_npc = plt_comm.get_subnet_numpowcycles(each_subnet_data)            
                tot_intpow_backup_cost, tot_intpow_recovery_cost, tot_contpow_nvmwrite_cost, tot_contpow_nvmread_cost = plt_comm.get_subnet_backuprecovery_cost(each_subnet_data)
                mean_ksize_per_op = plt_comm.get_subnet_avgkernelsize_per_op(each_subnet_data)
                net_choices = each_subnet_data['net_choices'] # wm, ir
                
                if net_choices in FILTER_NET_CHOICES:
                
                    all_stats["all_subnet_acc"].append(acc)
                    all_stats["all_subnet_icostprop"].append(int_mng_cost_proportion)
                    all_stats["all_subnet_inte2elat"].append(e2e_lat_intpow)
                    all_stats["all_subnet_numops"].append(numops)
                    all_stats["all_subnet_avgch_per_op"].append(mean_numch_per_op)
                    all_stats["all_subnet_avgksize_per_op"].append(mean_ksize_per_op)        
                    all_stats["all_subnet_flops"].append(tot_flops)
                    all_stats["all_subnet_compcost"].append(tot_compcost)   
                    all_stats["all_subnet_npc"].append(tot_npc)            
                    all_stats["all_subnet_backup_cost"].append(tot_intpow_backup_cost)            
                    all_stats["all_subnet_recovery_cost"].append(tot_intpow_recovery_cost)     
                                    
                    all_stats["all_subnet_widthmul"].append(net_choices[0])
                    all_stats["all_subnet_inputres"].append(net_choices[1])
                
        print ("total samples=", len(all_stats["all_subnet_acc"])) 
                
        # save   
        print("pickle_dump: ", pkl_fname)     
        pickle_dump(pkl_fname, all_stats)
        
        return all_stats       
          
        




def plot_ksizes_vs_backuprecovery_flops(all_data):
    # plot mem vs. acc    
    df1 = pd.DataFrame(data = all_data)  
    #df1.info()    
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_avgksize_per_op', 'all_subnet_backup_cost', 
                              "Avg. Kernel Size per Op", "Total Backup Cost (latency)")
    
    ax2 = plt_comm.plt_scatter(df1, 'all_subnet_avgksize_per_op', 'all_subnet_recovery_cost', 
                              "Avg. Kernel Size per Op", "Total Recovery Cost (latency)")
    
    ax3 = plt_comm.plt_scatter(df1, 'all_subnet_avgksize_per_op', 'all_subnet_compcost', 
                              "Avg. Kernel Size per Op", "Computation Cost (latency)")
        
 
    

def plot_acc_icostprop_vs_ksizes(all_data):
    # plot mem vs. acc    
    df1 = pd.DataFrame(data = all_data)      
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_avgksize_per_op', 'all_subnet_icostprop', 
                              "Avg. Kernel Size per Op", "IMO (%)")
    
    ax2 = plt_comm.plt_scatter(df1, 'all_subnet_avgksize_per_op', 'all_subnet_acc', 
                              "Avg. Kernel Size per Op", "Accuracy (%)")
    

def plot_icostprop_vs_backup_recovery(all_data):
    # plot mem vs. acc    
    df1 = pd.DataFrame(data = all_data)              
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_backup_cost', 'all_subnet_icostprop', 
                              "Total Backup Cost (latency)", "IMO (%)")

    ax2 = plt_comm.plt_scatter(df1, 'all_subnet_recovery_cost', 'all_subnet_icostprop', 
                              "Total Recovery Cost (latency)", "IMO (%)")

    


def plot_icostprop_vs_npc(all_data):
    # plot mem vs. acc    
    df1 = pd.DataFrame(data = all_data)  
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_npc', 'all_subnet_icostprop', 
                              "# of power cycles", "IMO (%)")
    
        

def plot_acc_icostprop_vs_flops(all_data):
    # plot mem vs. acc    
    df1 = pd.DataFrame(data = all_data)  
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_flops', 'all_subnet_icostprop', 
                              "FLOPS", "IMO (%)")
    
    ax2 = plt_comm.plt_scatter(df1, 'all_subnet_flops', 'all_subnet_acc', 
                              "FLOPS", "Accuracy Cost (%)")
    
    
def plot_acc_icostprop_vs_numops(all_data):    
    df1 = pd.DataFrame(data = all_data)  
    #df1.info()    
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_numops', 'all_subnet_icostprop', 
                              "# of Operations", "IMO (%)")

    ax2 = plt_comm.plt_scatter(df1, 'all_subnet_numops', 'all_subnet_acc', 
                              "# of Operations", "Accuracy (%)")


def plot_acc_icostprop_vs_avgch(all_data):
    # plot mem vs. acc    
    df1 = pd.DataFrame(data = all_data)  
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_avgch_per_op', 'all_subnet_icostprop', 
                              "Avg. Channels per Op.", "IMO (%)")

    ax2 = plt_comm.plt_scatter(df1, 'all_subnet_avgch_per_op', 'all_subnet_acc', 
                              "Avg. Channels per Op.", "Accuracy (%)")
    
        
    

def plot_icostprop_vs_acc(all_data):    
    # plot mem vs. acc    
    df1 = pd.DataFrame(data = all_data)  
    df1.info() 
    cmap = sns.color_palette("Blues_r_d", as_cmap=True) # Get a CMap
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_icostprop', 'all_subnet_acc', 
                              "IMO (%)", "Accuracy (%)",
                              cmap=cmap, cmap_metric = 'all_subnet_inte2elat', cmap_label = 'Intermittent Inference Latency (s)',
                              figsize=(4.5,3.5), savefig=False)
        


def plot_icostprop_vs_acc_colorsupnets_withhist(all_data):    
    
    def _report_dist_min_max(data, text):
        print(np.min(data), np.max(data), text)
    
    df1 = pd.DataFrame(data = all_data)  
    #df1.info() 
    # ax1 = plt_comm.plt_scatter(df1, 'all_subnet_icostprop', 'all_subnet_acc', 
    #                           "IMO (%)", "Accuracy (%)",                              
    #                           figsize=(4.5,3.5), savefig=False)
    
    wm_unqiue = df1['all_subnet_widthmul'].unique()
    ir_unique = df1['all_subnet_inputres'].unique()    
    
    # -- markers as IR, colors as WM      
    #color_map = dict(zip(wm_unqiue, sns.color_palette("bright", len(wm_unqiue))))     
    #color_map = dict(zip(wm_unqiue, ['#ff7f00', '#377eb8']))   
    color_map = dict(zip(wm_unqiue, ['#969696', '#08519c']))   
    cols = list(df1['all_subnet_widthmul'].map(color_map))    
    mm = list(df1['all_subnet_inputres'].map({32:'x', 28:'o'}))
       
    
    x_data = df1['all_subnet_icostprop']
    y_data = df1['all_subnet_acc']
    
    x_data_wm02 = df1[df1['all_subnet_widthmul'].isin([0.2])]['all_subnet_icostprop']
    x_data_wm10 = df1[df1['all_subnet_widthmul'].isin([1.0])]['all_subnet_icostprop']
    
    y_data_wm02 = df1[df1['all_subnet_widthmul'].isin([0.2])]['all_subnet_acc']
    y_data_wm10 = df1[df1['all_subnet_widthmul'].isin([1.0])]['all_subnet_acc']
    
    
    scatter_axes = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
    x_hist_axes = plt.subplot2grid((4, 4), (0, 0), colspan=3,
                                sharex=scatter_axes)
    y_hist_axes = plt.subplot2grid((4, 4), (1, 3), rowspan=3,
                                sharey=scatter_axes)
    
    
    for x,y,c,m in zip(x_data, y_data, cols, mm):        
        if m=='x':
            scatter_axes.scatter(x,y,color=c, marker=m)
        else:
            scatter_axes.scatter(x,y,color=c, marker=m, facecolors='none')
        
    #x_hist_axes.hist(x_data)
    #y_hist_axes.hist(y_data, orientation='horizontal')
    
    #sns.kdeplot(x_data_wm02, ax=x_hist_axes, color=color_map[0.2], fill=True)
    #sns.kdeplot(x_data_wm10, ax=x_hist_axes, color=color_map[1.0], fill=True)
    #-- sns.kdeplot(x_data, ax=x_hist_axes, color='b', fill=True, vertical=True)
    
    #sns.kdeplot(y_data_wm02, ax=y_hist_axes, color=color_map[0.2], fill=True, vertical=True)
    #sns.kdeplot(y_data_wm10, ax=y_hist_axes, color=color_map[1.0], fill=True, vertical=True)
    
    sns.distplot(x_data_wm02, ax=x_hist_axes, bins=20, color=color_map[0.2], kde=True)
    sns.distplot(x_data_wm10, ax=x_hist_axes, bins=20, color=color_map[1.0], kde=True)
    
    _report_dist_min_max(x_data_wm02, "IMO -> WM=0.2")
    _report_dist_min_max(x_data_wm10, "IMO -> WM=1.0")
    
    sns.distplot(y_data_wm02, ax=y_hist_axes, bins=20, color=color_map[0.2], kde=True, vertical=True)
    sns.distplot(y_data_wm10, ax=y_hist_axes, bins=20, color=color_map[1.0], kde=True, vertical=True)
    
    _report_dist_min_max(y_data_wm02, "ACC -> WM=0.2")
    _report_dist_min_max(y_data_wm10, "ACC -> WM=1.0")
    
    
    # -- format scatter axis --
    scatter_axes.set_xlabel("IMO (%)")
    scatter_axes.set_ylabel("Accuracy (%)")
    scatter_axes.grid(True, color='gray', linestyle='dashed')
    scatter_axes.set_axisbelow(True)            
    scatter_axes.set_xlim([25,60])
    scatter_axes.set_ylim([20,80])
    
    
    # -- format hist axes --    
    x_hist_axes.set_xlabel("")
    y_hist_axes.set_xlabel("")
    y_hist_axes.set_ylabel("")    
    plt.setp(x_hist_axes.get_xticklabels(), visible=False)
    plt.setp(y_hist_axes.get_yticklabels(), visible=False)
    plt.setp(y_hist_axes.get_xticklabels(), visible=False)
    
    x_hist_axes.get_yaxis().set_visible(False)
    x_hist_axes.spines["left"].set_visible(False)
    x_hist_axes.spines["right"].set_visible(False)
    x_hist_axes.spines["top"].set_visible(False)
    
    y_hist_axes.get_xaxis().set_visible(False)    
    y_hist_axes.spines["right"].set_visible(False)
    y_hist_axes.spines["top"].set_visible(False)
    y_hist_axes.spines["bottom"].set_visible(False)
    
    
    # -- legend --
    marker_size=7
    custom_leg = [
            Line2D([], [], marker='o', color=color_map[0.2], markersize=marker_size, markeredgewidth=2, linestyle='None', fillstyle='none'),   # <0.2, 32>            
            Line2D([], [], marker='x', color=color_map[0.2], markersize=marker_size, markeredgewidth=2, linestyle='None'),    # <0.2, 28>
            Line2D([], [], marker='o', color=color_map[1.0], markersize=marker_size, markeredgewidth=2, linestyle='None', fillstyle='none'),    # <1.0, 32>
            Line2D([], [], marker='x', color=color_map[1.0], markersize=marker_size, markeredgewidth=2, linestyle='None')    # <1.0, 28>
            ]
    leg = plt.legend(handles = custom_leg, 
               labels=['W=0.2, R=32', 'W=0.2, R=28', 'W=1.0, R=32', 'W=1.0, R=28'], 
               #labels=['<W=0.2, R=28>', '<W=1.0, R=32>'], 
#               title = "<W, R>",
               bbox_to_anchor= (1.05, 0.5), 
               loc= "lower left",
               ncol=4, columnspacing=0.6, handletextpad=0)
    leg.set_draggable(True)
    
    # -- axis lims
    
  
    
    
    

# def plot_icostprop_vs_acc_colorsupnets(all_data):    
#     # plot mem vs. acc    
#     df1 = pd.DataFrame(data = all_data)  
#     #df1.info() 
#     # ax1 = plt_comm.plt_scatter(df1, 'all_subnet_icostprop', 'all_subnet_acc', 
#     #                           "IMO (%)", "Accuracy (%)",                              
#     #                           figsize=(4.5,3.5), savefig=False)
    
#     wm_unqiue = df1['all_subnet_widthmul'].unique()
#     ir_unique = df1['all_subnet_inputres'].unique()    
    
#     # -- markers as IR, colors as WM      
#     color_map = dict(zip(wm_unqiue, sns.color_palette("bright", len(wm_unqiue))))     
#     cols = list(df1['all_subnet_widthmul'].map(color_map))    
#     mm = list(df1['all_subnet_inputres'].map({32:'+', 28:'x'}))
       
    
#     x_data = df1['all_subnet_icostprop']
#     y_data = df1['all_subnet_acc']
#     #cc = df1['all_subnet_widthmul'].map(color_map)
#     #pprint(cc); sys.exit()
#     #tmp = df1['all_subnet_inputres'].map({32:'o', 28:'s'}); print(tmp); sys.exit()
    
#     fig, ax = plt.subplots()
    
#     for x,y,c,m in zip(x_data, y_data, cols, mm):        
#         ax.scatter(x,y,color=c, marker=m)
    
#     ax.set_xlabel("IMO (%)")
#     ax.set_ylabel("Accuracy (%)")        
        
#     custom = [
#             Line2D([], [], marker='+', color=color_map[0.2], linestyle='None'),   # <0.2, 32>            
#             Line2D([], [], marker='x', color=color_map[0.2], linestyle='None'),    # <0.2, 28>
#             Line2D([], [], marker='+', color=color_map[1.0], linestyle='None'),    # <1.0, 32>
#             Line2D([], [], marker='x', color=color_map[1.0], linestyle='None')    # <1.0, 28>
#             ]
#     plt.legend(handles = custom, labels=['<0.2, 32>', '<0.2, 28>', '<1.0, 32>', '<1.0, 28>'], bbox_to_anchor= (1.05, 0.5), loc= "lower left")
  






def plot_ipowe2elat_vs_acc(all_data):    
    # plot mem vs. acc    
    df1 = pd.DataFrame(data = all_data)  
    df1.info()    
    ax1 = plt_comm.plt_scatter(df1, 'all_subnet_inte2elat', 'all_subnet_acc', 
                              "Int. Power E2E Latency", "Accuracy (%)",
                              cmap='viridis', cmap_metric = 'all_subnet_icostprop', cmap_label = 'IMO (%)')
   

    

def plot_network_stats(all_data):
    # num_layers = 
    # # num layers
    # for each_subnet in all_data:
    #     nl = len(each_subnet['perf_exec_design_intpow'])
    #     num_layers.append(nl)
    
    plt.figure()
    plt.hist(all_data["all_subnet_numops"], bins=100)
    
        
    


    
    


if __name__ == '__main__':
    
    all_stats = get_data_and_stats()
    
    #plot_icostprop_vs_acc(all_stats)
    #plot_icostprop_vs_acc_colorsupnets(all_stats)
    plot_icostprop_vs_acc_colorsupnets_withhist(all_stats)
    #plot_ipowe2elat_vs_acc(all_stats)
    
    #plot_acc_icostprop_vs_avgch(all_stats)
    
    #plot_acc_icostprop_vs_numops(all_stats)
    
    #plot_acc_icostprop_vs_flops(all_stats)

    #plot_icostprop_vs_npc(all_stats)
    
    #plot_icostprop_vs_backup_recovery(all_stats)
    
    
    #plot_acc_icostprop_vs_ksizes(all_stats)
    
    #plot_ksizes_vs_backuprecovery_flops(all_stats)
    
    
    #plot_network_stats(all_data)
    
        
    plt.show()    
    
