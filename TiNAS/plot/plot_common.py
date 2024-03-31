import csv
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
import matplotlib.colors as mc
import colorsys

import mplcursors

import pandas as pd
import json
from collections import OrderedDict 

from scipy import stats

import re

import seaborn as sns

sys.path.append("..")
from NASBase.file_utils import json_load

from settings import Settings
from NASBase.hw_cost.Modules_inas_v1.CostModel.cnn import est_FLOPS_cost_layer
import NASBase.hw_cost.Modules_inas_v1.CostModel.common as common


######################################
# HELPERS
######################################
def _get_blkid_from_alias(alias):
    if "blocks." in alias:
        found = re.search('blocks.(.+?).', alias).group(1)
    return int(found)

def adjust_lightness(color, amount=0.5):    
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


######################################
# COMMON PLOTTING
######################################

def plt_regplot(df, x_metric, y_metric, x_label, y_label, marker='x',                  
                show_avg=False, col = '#377eb8', alpha=1.0, order=2,
                figsize=(5,4), fig_title=None, ax=None,
                ):
    
        
    scatter = sns.regplot(data=df, x=x_metric, y=y_metric,                              
                          order=order if order!=None else 1,
                          fit_reg=True if order!=None else False,
                          ax=ax,
                          marker=marker,
                          scatter_kws=dict(alpha=alpha, color=col, marker=marker),
                          line_kws=dict(color=adjust_lightness(col,0.9)))
                          
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)        
    ax.grid(True, linestyle='--')
    
    

def plt_scatter(df, x_metric, y_metric, x_label, y_label, marker='x',  
                cmap=None, cmap_metric = None, cmap_label = None, 
                show_avg=False, col = '#377eb8', alpha=1.0,
                savefig=False, figsize=(5,4), fig_title=None, ax=None, order=None,
                click_annotate_metric='all_subnet_choice_per_blk'):
    
    
    lines= None
    
    if cmap is None:        
        
        if ax:   
            #df.plot.scatter(ax = ax, x = x_metric, y = y_metric, grid=True, marker=marker, c=cmap_metric, colormap=cmap, figsize=figsize)
            lines = ax.scatter(df[x_metric], df[y_metric], marker=marker, color=col, alpha=alpha)
                 
        else:
            fig, ax = plt.subplots(figsize=figsize)
            if fig_title: fig.suptitle(fig_title)
            
            lines = ax.scatter(df[x_metric], df[y_metric], marker=marker, color=col, alpha=alpha)      
            #df.plot.scatter(x='date', y='value', c=df['category'].map({'Wholesale':'red','Retail':'blue'}))
            
        if show_avg:                
            round_by=2
            df.sort_values(by=x_metric, inplace=True)
            bins = df.groupby(df[x_metric].round(round_by)).mean()
            print("--- plt_common:")
            pprint(bins['all_subnet_icostprop'])
            #pprint(bins[x_metric])
            ax.plot(bins.index, bins[y_metric], 'g--', lw=3)
            
            #ax = df.plot.scatter(x = x_metric, y = y_metric, grid=True, marker=marker, figsize=figsize)    
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)        
        ax.grid(True)
        
    else:
        #lines = ax.scatter(df[x_metric], df[y_metric], marker=marker, c=df[cmap_metric], colormap=cmap)        
        if ax:        
            df.plot.scatter(ax = ax, x = x_metric, y = y_metric, grid=True, marker=marker, c=cmap_metric, colormap=cmap, figsize=figsize)
        else:
            ax = df.plot.scatter(x = x_metric, y = y_metric, grid=True, marker=marker, c=cmap_metric, colormap=cmap, figsize=figsize)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label) 
        if fig_title: ax.set_title(fig_title)       
        ax.grid(True)
        cax = plt.gcf().get_axes()[1]    
        cax.set_ylabel(cmap_label)
        
    plt.tight_layout()
    
    mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(df.iloc[sel.index][click_annotate_metric]))
    
    if (savefig):
        fname = "{}-{}.png".format(x_metric.replace(" ", "_"), y_metric.replace(" ", "_"))
        plt.savefig(fname, bbox_inches='tight')
        
    return ax, lines

    


######################################
# DATA EXTRACTION
######################################

def get_subnet_net_choices(subnet_data):    
    return subnet_data["net_choices"]


def get_subnet_num_skipsupp_blocks(subnet_data):
    # do a small varlidation check
    # according to block choices
    #num_skip_blocks_1 = np.sum([b[3] for b in subnet_data["subnet_choice_per_blk"]])    
    #num_skip_blocks_2 = len([op for op in subnet_data["perf_exec_design_intpow"] if "ADD" in op['layer']])    
    
    num_skip_blocks = len(set([_get_blkid_from_alias(op['alias']) for op in subnet_data["perf_exec_design_intpow"] if "ADD" in op['layer']]))
    
    #blkids = len(set([_get_blkid_from_alias(op['alias']) for op in subnet_data["perf_exec_design_intpow"] if "ADD" in op['layer']]))
    #print(blkids); sys.exit()
    #print(num_skip_blocks_1, len(blkids))
    
    #if(num_skip_blocks_1!=len(blkids)) and (subnet_data['test_class'] == "test_supskip"):
    #    print(num_skip_blocks_1, len(blkids))
    
    return num_skip_blocks
    

def get_subnet_num_ops(subnet_data):
    return len(subnet_data['subnet_obj'])

def get_subnet_avgch_per_op(subnet_data):
    nout_ch = [int(l['OFM']['CH']) for l in subnet_data['subnet_obj']]    
    return np.mean(nout_ch), np.min(nout_ch), np.max(nout_ch)


def get_subnet_avgkernelsize_per_op(subnet_data):
    ksizes = [int(l['K']['H']) for l in subnet_data['subnet_obj'] if ("CONV" in l['type']) and ("_dw" in l['alias'])] 
    #pprint(ksizes); input()
    return np.mean(ksizes)    

def get_subnet_modekernelsize_per_op(subnet_data):
    ksizes = [int(l['K']['H']) for l in subnet_data['subnet_obj'] if ("CONV" in l['type']) and ("_dw" in l['alias'])] 
    #pprint(ksizes); input()
    #return np.mean(ksizes)  
    #print(stats.mode(ksizes)[0][0]); sys.exit()
      
    return stats.mode(ksizes)[0][0]

# -- avg param choice per block < exp, ks, nl, sk >--

def get_subnet_avg_expf_per_blk(subnet_data):
    blk_choices = [b[0] for b in subnet_data['subnet_choice_per_blk']]
    return np.mean(blk_choices)

def get_subnet_avg_ksize_per_blk(subnet_data):
    blk_choices = [b[1] for b in subnet_data['subnet_choice_per_blk']]
    return np.mean(blk_choices)

def get_subnet_avg_numlay_per_blk(subnet_data):
    blk_choices = [b[2] for b in subnet_data['subnet_choice_per_blk']]
    return np.mean(blk_choices)

# num of skip enabled blocks
def get_subnet_num_skipen_blks(subnet_data):
    blk_choices = [b[3] for b in subnet_data['subnet_choice_per_blk']]
    return blk_choices.count(True)


def get_blk_choice_value(choice_per_blk, bix, k):
    # order : [EXP_FACTORS, KERNEL_SIZES, NUM_LAYERS, SUPPORT_SKIP]    
    if (k == 'EXP_FACTORS'):
        return choice_per_blk[bix][0]
    elif (k == 'KERNEL_SIZES'):
        return choice_per_blk[bix][1]
    elif (k == 'NUM_LAYERS'):
        return choice_per_blk[bix][2]
    elif (k == 'SUPPORT_SKIP'):
        return choice_per_blk[bix][3]
    else:
        sys.exit("_get_blk_choice_value::Error")



def get_subnet_icostprop(subnet_data):
    exec_design_intpow = subnet_data['perf_exec_design_intpow']
    #exec_design_contpow = subnet_data['perf_exec_design_contpow']
    exec_design_contpow_fp = subnet_data['perf_exec_design_contpow_fp']
    e2e_lat_intpow = subnet_data['perf_e2e_intpow_lat']
    #e2e_lat_contpow = subnet_data['perf_e2e_contpow_lat']    
    e2e_lat_contpow_fp = subnet_data['perf_e2e_contpow_fp_lat']    
    
    # INT power breakdown        
    ip_tot_npc = np.sum([l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1        # total power cycles
    ip_tot_rc = np.sum([l['L_rc_tot'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1    # total recharge time
    ip_tot_rb = np.sum([l['cost_brk']['rb'][1] * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
    ip_tot_backup = np.sum([ (l['cost_brk']['bd'][1] + l['cost_brk']['bl'][1]) * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
    ip_tot_recovery = np.sum([ (l['cost_brk']['fd'][1] + l['cost_brk']['fl'][1]) * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
    ip_tot_computation = np.sum([ l['cost_brk']['cp'][1] * l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
                       
    # CONT power breakdown - best design
    # cp_tot_nvmwr = np.sum([ (l['cost_brk']['bd'][1] + l['cost_brk']['bl'][1]) for l in exec_design_contpow]) if e2e_lat_contpow != -1 else -1
    # cp_tot_nvmrd = np.sum([ (l['cost_brk']['fd'][1] + l['cost_brk']['fl'][1]) for l in exec_design_contpow]) if e2e_lat_contpow != -1 else -1
    # cp_tot_computation = np.sum([ l['cost_brk']['cp'][1] for l in exec_design_contpow]) if e2e_lat_contpow != -1 else -1
    
    # CONT power breakdown - fixed design                              
    cpfp_tot_nvmwr = np.sum([ (l['cost_brk']['bd'][1] + l['cost_brk']['bl'][1]) for l in exec_design_contpow_fp]) if e2e_lat_contpow_fp != -1 else -1
    cpfp_tot_nvmrd = np.sum([ (l['cost_brk']['fd'][1] + l['cost_brk']['fl'][1]) for l in exec_design_contpow_fp]) if e2e_lat_contpow_fp != -1 else -1
    cpfp_tot_computation = np.sum([ l['cost_brk']['cp'][1] for l in exec_design_contpow_fp]) if e2e_lat_contpow_fp != -1 else -1
                
    
    # -- IMC - version 1 (a)  - A (cp: fixed design)  
    active_time = (e2e_lat_intpow - ip_tot_rc)
    int_mng_cost_proportion_v1_fp = ((active_time-e2e_lat_contpow_fp)/active_time)*100
    
    
    # -- IMC - version 1 (b) - A (cp: best desing)      
    # active_time = (e2e_lat_intpow - ip_tot_rc)
    # int_mng_cost_proportion_v1 = ((active_time-e2e_lat_contpow)/active_time)*100
        
    
    # -- IMC - version 2        
    # active_time = (e2e_lat_intpow - ip_tot_rc)
    # int_mng_cost_proportion_v2 = ((ip_tot_backup+ip_tot_recovery+ip_tot_rb)/active_time)*100
    
    
    return int_mng_cost_proportion_v1_fp



            
def get_subnet_flops(subnet_data):
    total_flops = 0
    total_macs = 0
    
    for each_layer, each_layer_data in zip(subnet_data['subnet_obj'], subnet_data['perf_exec_design_contpow_fp']):
        
        if (each_layer['K']['H'] is not None): Kh=int(each_layer['K']['H'])
        else: Kh=None
        if (each_layer['K']['W'] is not None): Kw=int(each_layer['K']['W']) 
        else: Kh=None
        
        #Kh = int(each_layer['K']['H']) if each_layer['K']['H']  None else None
        #Kw = int(each_layer['K']['W']) if type(each_layer['K']['W']) != None else None
        
        Tr, Tc, Tm, Tn, inter_lo, S = common.string_to_params_all(each_layer_data['params'])        
        params_exec = {'tile_size': [Kh, Kw, None, None, Tr, Tc, Tm, Tn], 'inter_lo': inter_lo}
        params_pres = {'backup_batch_size': S}
        
        layer_flops, layer_macs = est_FLOPS_cost_layer(each_layer, params_exec, params_pres, layer_based_cals=True)
         
        total_flops += layer_flops
        total_macs += layer_macs

    return total_flops, total_macs


def get_subnet_numpowcycles(subnet_data):
    exec_design_intpow = subnet_data['perf_exec_design_intpow']
    e2e_lat_intpow = subnet_data['perf_e2e_intpow_lat']
    tot_npc = np.sum([l['npc'] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1        # total power cycles
    return tot_npc
    

def get_subnet_backuprecovery_cost(subnet_data):
    exec_design_intpow = subnet_data['perf_exec_design_intpow']
    #exec_design_contpow = subnet_data['perf_exec_design_contpow']
    exec_design_contpow_fp = subnet_data['perf_exec_design_contpow_fp']    
    e2e_lat_intpow = subnet_data['perf_e2e_intpow_lat']
    #e2e_lat_contpow = subnet_data['perf_e2e_contpow_lat']                
    e2e_lat_contpow_fp = subnet_data['perf_e2e_contpow_fp_lat']   
    
    tot_intpow_backup_cost = np.sum([ l['npc']*(l['cost_brk']['bd'][1] + l['cost_brk']['bl'][1]) for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1  
    tot_intpow_recovery_cost = np.sum([ l['npc']*(l['cost_brk']['fd'][1] + l['cost_brk']['fl'][1]) for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1  
    tot_intpow_reboot_cost = np.sum([l['npc']*l['cost_brk']['rb'][1] for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1
    
    #tot_contpow_nvmwrite_cost = np.sum([ (l['cost_brk']['bd'][1]) for l in exec_design_contpow]) if e2e_lat_contpow != -1 else -1  
    #tot_contpow_nvmread_cost = np.sum([ (l['cost_brk']['fd'][1]) for l in exec_design_contpow]) if e2e_lat_contpow != -1 else -1  
    
    tot_contpow_fp_nvmwrite_cost = np.sum([ (l['cost_brk']['bd'][1]) for l in exec_design_contpow_fp]) if e2e_lat_contpow_fp != -1 else -1  
    tot_contpow_fp_nvmread_cost = np.sum([ (l['cost_brk']['fd'][1]) for l in exec_design_contpow_fp]) if e2e_lat_contpow_fp != -1 else -1  
        
    return tot_intpow_backup_cost, tot_intpow_recovery_cost+tot_intpow_reboot_cost, tot_contpow_fp_nvmwrite_cost, tot_contpow_fp_nvmread_cost
    #return tot_intpow_backup_cost, tot_intpow_recovery_cost+tot_intpow_reboot_cost, tot_contpow_nvmwrite_cost, tot_contpow_nvmread_cost


def get_subnet_compcost(subnet_data):
    exec_design_intpow = subnet_data['perf_exec_design_intpow']
    #exec_design_contpow = subnet_data['perf_exec_design_contpow']
    exec_design_contpow_fp = subnet_data['perf_exec_design_contpow_fp']    
    e2e_lat_intpow = subnet_data['perf_e2e_intpow_lat']
    #e2e_lat_contpow = subnet_data['perf_e2e_contpow_lat']                
    e2e_lat_contpow_fp = subnet_data['perf_e2e_contpow_fp_lat']   
    
    
    tot_intpow_compcost = np.sum([ l['npc']*(l['cost_brk']['cp'][1]) for l in exec_design_intpow]) if e2e_lat_intpow != -1 else -1  
    #tot_contpow_compcost = np.sum([ (l['cost_brk']['cp'][1]) for l in exec_design_contpow]) if e2e_lat_contpow != -1 else -1  
    tot_contpow_fp_compcost = np.sum([ (l['cost_brk']['cp'][1]) for l in exec_design_contpow_fp]) if e2e_lat_contpow_fp != -1 else -1  
    
    
    return tot_intpow_compcost, tot_contpow_fp_compcost


def process_evo_search_log(logfname, callback_func, **kwargs):

    logfile = open(logfname, 'r')

    reader = csv.reader(logfile)

    COLUMN_NAMES = ['time', 'iteration',
                    'best_score', 'worst_score',
                    'best_acc', 'worst_acc',
                    'best_imc', 'worst_imc',
                    'best_config', 'worst_config',
                    'uniq']

    ret_values = []

    for row_idx, row in enumerate(reader):        
        if row_idx == 0:    # Skip the header
            continue

        assert len(row) == len(COLUMN_NAMES)

        kwargs.update({
            column_name: row[col_idx]
            for col_idx, column_name in enumerate(COLUMN_NAMES)
        })

        kwargs['iteration'] = int(kwargs['iteration'])
        kwargs['best_score'] = float(kwargs['best_score'])
        kwargs['worst_score'] = float(kwargs['worst_score'])
        kwargs['best_acc'] = float(kwargs['best_acc'])
        kwargs['best_imc'] = float(kwargs['best_imc'])
        
        

        ret_values.append(callback_func(**kwargs))

    return ret_values
