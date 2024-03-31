import copy
import datetime
import itertools
from pprint import pprint
import statistics
import traceback
from typing import Dict, List, Tuple

import numpy as np

from settings import Settings
from NASBase.hw_cost.Modules_inas_v1.IEExplorer.plat_perf import PlatPerf
from NASBase.model.common_utils import get_network_dimension, get_network_obj, netobj_to_pyobj, get_supernet, get_dummy_net_input_tensor
from NASBase.model.mnas_arch import MNASSubNet
from NASBase.multiprocessing_helper import get_max_num_workers, run_multiprocessing_workers
from NASBase.ss_optimization.subnet_utils import sample_subnet_configs, check_constraints, merge_constraint_stats

NetChoice = Tuple[float, float]

def imc_worker(cpuid, global_settings: Settings, width_multiplier, input_resolution, subnet_config_list):
    print("CPUID [%d] :: Enter : has %d jobs " % (cpuid, len(subnet_config_list)))
    
    print("Starting processes for width_multiplier={}, input_resolution={}..".format(width_multiplier, input_resolution))

    # init
    subnet_name="UNKNOWN_SUBNET";subnet_cpb=[];subnet_obj=[]
    
    net_input = get_dummy_net_input_tensor(global_settings, input_resolution)

    subnet_results = []
    error_net_perf = False

    performance_model = PlatPerf(global_settings.NAS_SETTINGS_GENERAL, global_settings.PLATFORM_SETTINGS)

    constraint_stats = {}

    for i, each_subnet_config in enumerate(subnet_config_list):        
        
        each_subnet = MNASSubNet(**each_subnet_config)
        try:        
            subnet_name = each_subnet.name
            subnet_cpb = each_subnet.choice_per_block

            # -- get subnet costs
            subnet_dims = get_network_dimension(each_subnet, input_tensor = net_input)         
            subnet_obj = get_network_obj(subnet_dims)       

            checked_constraints = global_settings.NAS_SSOPTIMIZER_SETTINGS['SSOPT_CONSTRAINTS']
            if not check_constraints(performance_model, subnet_obj, subnet_name, subnet_cpb, checked_constraints, constraint_stats):
                continue
            
            network_imc = performance_model.get_imc(subnet_obj, each_subnet_config)
            
        except Exception as e:            
            error_net_perf = True
            pprint(e)
            tb = traceback.format_exc()
            print(tb)
            print("subnet_cpb: ", subnet_cpb)
        
                
        if (error_net_perf == False):
            print("[CPU-{}] Finished processing subnet: {}, imc: {}".format(cpuid, subnet_name, network_imc))
        else:
            print("[CPU-{}] ERROR processing subnet: {}, imc: {}".format(cpuid, subnet_name, network_imc))
        
        subnet_results.append({
            "subnet_name": subnet_name,
            "subnet_obj" : netobj_to_pyobj(subnet_obj),
            "subnet_choice_per_blk": subnet_cpb,
            "net_choices" : each_subnet.net_choices,

            "supernet_choice": (width_multiplier, input_resolution),

            "error_net_perf"  : error_net_perf,          
            
            # perf cont pow
            "perf_e2e_imc": network_imc,
        })

    return subnet_results, constraint_stats


def calc_imc_for_supernet(global_settings, dataset, supernet, width_multiplier, input_resolution, supernet_block_choices, n, first_block_hard_coded):

    # sampling in parent process, until there are n satisfying subnets
    all_subnet_configs_lst_generator = sample_subnet_configs(supernet, supernet_block_choices, first_block_hard_coded=first_block_hard_coded)

    available_cpus = min(16, get_max_num_workers(worker_type='CPU'))

    all_subnet_results = []
    while True:
        needs_subnets = n
        print(f'Sampling {needs_subnets} more subnet(s)')
        subnets_to_sample = max(available_cpus, needs_subnets)  # Process at least one subnet per core to speed up sample & check loop

        all_subnet_configs = list(itertools.islice(all_subnet_configs_lst_generator, subnets_to_sample))
        batched_subnet_configs = np.array_split(all_subnet_configs, available_cpus)

        results_all_cpus = run_multiprocessing_workers(
            num_workers=available_cpus,
            worker_func=imc_worker,
            worker_type='CPU',
            common_args=(global_settings, width_multiplier, input_resolution),
            worker_args=batched_subnet_configs,
        )
        all_constraint_stats = []
        for subnet_results_per_cpu, constraint_stats_per_cpu in results_all_cpus:
            all_subnet_results.extend(subnet_results_per_cpu)
            all_constraint_stats.append(constraint_stats_per_cpu)

        constraint_stats = merge_constraint_stats(all_constraint_stats)
        print(constraint_stats)

        # break when there are enough subnets
        if len(all_subnet_results) >= n:
            break

        if not global_settings.NAS_SSOPTIMIZER_SETTINGS['DO_RESAMPLING']:
            break

    print("\n------ All jobs complete ------ for width_multiplier={}, input_resolution={}, time={}..\n\n".format(
        width_multiplier, input_resolution, datetime.datetime.now()))

    return {
        'all_subnet_results': all_subnet_results[:n],
        'constraint_stats': constraint_stats,
    }


def reorganize_imc_data(imc_data) -> Dict[NetChoice, List[Dict]]:
    all_subnets_with_imc = {}

    for subnet_data in imc_data:
        supernet_choice = tuple(subnet_data['supernet_choice'])
        all_subnets_with_imc.setdefault(supernet_choice, []).append((subnet_data['perf_e2e_imc'], subnet_data))

    return all_subnets_with_imc

def sort_by_average_imc(data):
    supernet_choice, cur_subnets_with_imc, average_imc = data
    return average_imc

def ss_optimization_by_imc(global_settings, dataset, supernet_choices, supernet_block_choices):

    all_subnet_results = []
    per_supernet_stats = {}

    subnet_sample_size = global_settings.NAS_SSOPTIMIZER_SETTINGS['SUBNET_SAMPLE_SIZE']
    for width_multiplier, input_resolution in supernet_choices:
        supernet = get_supernet(global_settings=global_settings, dataset=dataset, width_multiplier=width_multiplier)
        ret = calc_imc_for_supernet(
            global_settings, dataset, supernet, width_multiplier, input_resolution, supernet_block_choices,
            n=subnet_sample_size,
            first_block_hard_coded=global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FIRST_BLOCK_HARD_CODED'])
        cur_subnet_results = ret['all_subnet_results']
        per_supernet_stats[f'({width_multiplier}, {input_resolution})'] = ret['constraint_stats']

        valid_subnets = len(cur_subnet_results)
        if valid_subnets >= global_settings.NAS_SSOPTIMIZER_SETTINGS['VALID_SUBNETS_THRESHOLD'] * subnet_sample_size:
            # drop the supernet if threshold is not reached
            all_subnet_results.extend(cur_subnet_results)
            print(f'There are {valid_subnets} valid subnets out of {subnet_sample_size} ones')
        else:
            print(f'Only {valid_subnets} valid subnets out of {subnet_sample_size} ones - dropping supernet')

    all_subnets_with_imc = reorganize_imc_data(all_subnet_results)

    supernet_average_imc = []

    # Put supernet choices and average imc for sorting
    for supernet_choice, cur_subnets_with_imc in all_subnets_with_imc.items():
        average_imc = statistics.mean(imc for imc, subnet_data in cur_subnets_with_imc)
        supernet_average_imc.append((
            supernet_choice,
            cur_subnets_with_imc,
            average_imc,
        ))
        print((supernet_choice, average_imc))

    supernet_average_imc.sort(key=sort_by_average_imc)

    supernet_choice, cur_subnets_with_imc, average_imc = supernet_average_imc[0]

    print('Stage 1 search space optimization done! Chosen supernet config: {}'.format(supernet_choice))

    supernet_properties = {
        'average_imc': average_imc,
        'num_subnets': len(cur_subnets_with_imc),
        'supernet_objtype': supernet.SUPERNET_OBJTYPE,
        'per_supernet_stats': per_supernet_stats,
    }

    return supernet_choice, supernet_properties
