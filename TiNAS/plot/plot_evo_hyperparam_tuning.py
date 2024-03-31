import math
import numpy as np
import pandas as pd
import csv
from os.path import dirname, realpath
import sys

from matplotlib import pyplot as plt

sys.path.append("..")
from NASBase.file_utils import json_load, file_exists

#from NASBase.evo_search.evo_hyperparam_tuning import SEARCH_PARENT_RATIO, SEARCH_MUT_PROB, SEARCH_MUT_RATIO


SEARCH_PARENT_RATIO = [0.1,  0.2, 0.25, 0.3] 
SEARCH_MUT_PROB = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
SEARCH_MUT_RATIO = [0.25, 0.5]


# SEARCH_PARENT_RATIO = [0.2] 
# SEARCH_MUT_PROB = [0.1, 0.2, 0.3, 0.5]
# SEARCH_MUT_RATIO = [0.25, 0.5]


SEARCH_SEEDS = [392, 521, 588]


#FNAME_SUFFIX = "evo_test_hyperparams_tune"
FNAME_SUFFIX = "TiNAS-U-HAR-IMO-perc100-LAT-perc25-"  

LOG_DIR = "../NASBase/train_log/evosearch_param_tuning/TiNAS-U-HAR-IMO-perc100-LAT-perc25/"

SCORE_TYPE = 'ACC_IMC'

def get_score(acc, imc):
    if SCORE_TYPE == 'ACC_IMC':
        return acc * (1/imc)
    elif SCORE_TYPE == 'ACC':
        return acc
    

def get_data():
    all_log_data = {}
    # evo param permutations    
    for evo_parent_ratio in SEARCH_PARENT_RATIO:
        for evo_mut_prob in SEARCH_MUT_PROB:
            for evo_mut_ratio in SEARCH_MUT_RATIO:                
                for evo_seed in SEARCH_SEEDS:
                    perm_k = "{}_{}_{}_{}".format(evo_parent_ratio, evo_mut_prob, evo_mut_ratio, evo_seed)            
                
                    csvlog_fname = LOG_DIR + FNAME_SUFFIX + "{}_{}_{}-seed{}-0_evo_search.csv".format(evo_parent_ratio, evo_mut_prob, evo_mut_ratio, evo_seed)
                    #print(csvlog_fname)
                    if file_exists(csvlog_fname):
                        print("getting data: ", csvlog_fname)
                        #sys.exit()
                    
                        file = open(csvlog_fname, "r")
                        row_data = list(csv.reader(file, delimiter=",")) 
                        row_data.pop(0) # delete header: time,iter,best_score,worst_score,best_acc,worst_acc,best_imc,worst_imc,best_efficiency,worst_efficiency,best_config,worst_config,uniq

                        file.close()                
                        
                        all_log_data[perm_k] = { 
                                                'best_score': [], 'worst_score': [], 
                                                'best_acc': [], 'worst_acc': [], 
                                                'best_imc': [], 'worst_imc': [],
                                                'uniq':[], 
                                                }
                        
                        for row in row_data:
                            #print(row); sys.exit()
                            #all_log_data[perm_k]['best_score'].append(get_score(float(row[2]), float(row[4])))
                            #all_log_data[perm_k]['worst_score'].append(get_score(float(row[3]), float(row[5])))                    
                            
                            all_log_data[perm_k]['best_score'].append(float(row[2]))
                            all_log_data[perm_k]['worst_score'].append(float(row[3]))                    
                            all_log_data[perm_k]['best_acc'].append(float(row[4]))
                            all_log_data[perm_k]['worst_acc'].append(float(row[5]))
                            all_log_data[perm_k]['best_imc'].append(float(row[6]))
                            all_log_data[perm_k]['worst_imc'].append(float(row[7]))   
                                 
                
    return all_log_data           







def plot_best_scores(all_log_data):
    
    x_data = {
        "best_score": [], #"worst_score": [],
        "perm_k": []
    }
        
    for i, evo_parent_ratio in enumerate(SEARCH_PARENT_RATIO):        
        for j, evo_mut_prob in enumerate(SEARCH_MUT_PROB):
            for k, evo_mut_ratio in enumerate(SEARCH_MUT_RATIO):
                
                perm_k_woseed = "{}_{}_{}".format(evo_parent_ratio, evo_mut_prob, evo_mut_ratio)
                
                # get seed data
                best_score_all_seeds = []
                for s, evo_seed in enumerate(SEARCH_SEEDS):                                    
                    perm_k_wseed = "{}_{}_{}_{}".format(evo_parent_ratio, evo_mut_prob, evo_mut_ratio, evo_seed)            
                                        
                    if (perm_k_wseed) in all_log_data:                            
                        best_score_all_seeds.append(all_log_data[perm_k_wseed]['best_score'][-1])
                        
                
                #if len(best_score_all_seeds)>0:                
                x_data['perm_k'].append(perm_k_woseed)          
                        
                x_data['best_score'].append([np.min(best_score_all_seeds), np.max(best_score_all_seeds), np.mean(best_score_all_seeds)]) # min, max, avg
    
    # plot
    fig, ax = plt.subplots()
    
    # min
    xx_data = [dmin for dmin, dmax, davg in x_data['best_score']]
    #ax.stem(xx_data, np.arange(len(xx_data)), markerfmt='or')
    ax.stem(xx_data, markerfmt='or')
    
    # max
    xx_data = [dmax for dmin, dmax, davg in x_data['best_score']]
    ax.stem(xx_data, markerfmt='og')
    
    # avg
    xx_data = [davg for dmin, dmax, davg in x_data['best_score']]
    ax.stem(xx_data, markerfmt='ob')
    
    #ax.set_xticks(np.arange(len(xx_data)))
    #ax.set_xticklabels(x_data['perm_k'], fontsize=8, rotation=45, ha='right')
    
    print(len(xx_data), len(x_data['perm_k']))
    
    plt.xticks(np.arange(len(xx_data)), x_data['perm_k'], fontsize=8, rotation=45, ha='right')



    
    
    # xx_data = []
    # df = pd.DataFrame(x_data, index=x_data["perm_k"])
    # #df.set_index('perm_k')    
    # ax = df.plot.stem(rot=45)
                    
    
    





def plot_all_evo_progress(all_log_data, metric='best_score'):
    
    tot_perms = len(all_log_data.keys())
    
    fig, axs = plt.subplots(len(SEARCH_MUT_PROB), len(SEARCH_MUT_RATIO) * len(SEARCH_PARENT_RATIO), sharex=True, sharey=True)   #r,c
        
    for r, evo_mut_prob in enumerate(SEARCH_MUT_PROB):
        
        c = 0
        for i, evo_mut_ratio in enumerate(SEARCH_MUT_RATIO):
            for j, evo_parent_ratio in enumerate(SEARCH_PARENT_RATIO):
                
                perm_k_woseed = "{}_{}_{}".format(evo_parent_ratio, evo_mut_prob, evo_mut_ratio)
                print(r, c, perm_k_woseed)
                axs[r,c].set_title(perm_k_woseed)
                
                cols = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
                
                for evo_seed, col in zip(SEARCH_SEEDS,cols):
                    perm_k = "{}_{}_{}_{}".format(evo_parent_ratio, evo_mut_prob, evo_mut_ratio, evo_seed)          
                    
                    if (perm_k) in all_log_data:
                    
                        data = np.array(all_log_data[perm_k][metric])  
                        gens = np.arange(len(data)) 
                        
                        axs[r,c].plot(gens, data, color=col) 
                        # show unique value indeces
                        #_, indices = np.unique(data, return_index=True)     
                        #indices = np.where(data[:-1] != data[1:])[0]               
                        indices = np.where(np.roll(data,1)!=data)[0]
                        axs[r,c].scatter(gens[indices], data[indices], marker='*', c=col)
                
                c+=1


def report_best_perm(all_log_data):
    pass #TODO
    


if __name__ == '__main__':
    all_log_data = get_data()
    plot_all_evo_progress(all_log_data, metric='best_imc')
    
    #plot_best_scores(all_log_data)
    
    
    plt.show()