import itertools
import operator
import numpy as np

import NASBase.utils as utils
from NASBase.model.mnas_arch import MNASSuperNet
from NASBase.model.mnas_ss import FIRST_BLOCK_EXP_FACTOR
from NASBase.hw_cost.Modules_inas_v1.IEExplorer.plat_perf import PlatPerf
from NASBase.hw_cost.Modules_inas_v1.CostModel import cnn

CONSTRAINT_STAT_KEYS = {
    'vm_feasible_subnets',
    'nvm_feasible_subnets',
    'imc_feasible_subnets',
    'latency_feasible_subnets',
}
CONSTRAINT_MAX_KEYS = {
    'flops_max',
    'npc_max',
    'latency_max',
    'imc_max',
}
CONSTRAINT_MIN_KEYS = {
    'flops_min',
    'npc_min',
    'latency_min',
    'imc_min',
}

# sample subnets from supernet
def sample_subnet_configs(net: MNASSuperNet, net_choices, first_block_hard_coded=False):

    num_choices_per_block = net_choices #model.choices #model.module.choices
    print("num_choices_per_block: ", len(num_choices_per_block))
    num_blocks = net.num_blocks
    
    all_choices_idxs_generator = utils.random_choices(len(num_choices_per_block), num_blocks)

    sidx = 0
    while True:
        # Make a copy of net_choices as single_choice_per_block will be modified if the first block is hard-coded
        single_choice_per_block = [list(net_choices[c]) for c in next(all_choices_idxs_generator)]
        if first_block_hard_coded:
            # Force expansion factor = 1
            single_choice_per_block[0][0] = FIRST_BLOCK_EXP_FACTOR
        #subnet_name = "subnet_ID" + str(sidx) + "_" + "_".join(list(map(','.join, single_choice_per_block)))
        subnet_name = "subnet_ID" + str(sidx)
        
        #pprint(single_choice_per_block); sys.exit()
        print("Creating Subnet config - ", sidx)

        subnet_config = {
            'name': subnet_name,
            'num_blocks': net.num_blocks,
            'num_classes': net.num_classes,
            'stem_c_out': net.stem_c_out,
            'input_channels': net.input_channels,
            'stride_first': net.stride_first,
            'downsample_blocks': net.downsample_blocks,
            'block_out_channels': net.output_channels,
            'subnet_id': sidx,
            'single_choice_per_block': single_choice_per_block,
            'use_1d_conv': net.use_1d_conv,
        }
        # if it passes the constraint
        yield subnet_config

        sidx += 1


def check_vm_constraint(performance_model: PlatPerf, subnet_obj, subnet_name, subnet_cpb, constraint_stats):
    if performance_model.PLAT_SETTINGS['VM_CAPACITY'] <= 0:
        # print if some constraint is skipped
        print('VM constraint is skipped!')
        return True

    all_layers_fit_vm, network_vm_usage, _, _ = performance_model.get_vm_usage(subnet_obj, fixed_params=None)

    if not all_layers_fit_vm:
        # print debug messages if some constraint is unsatisfied
        for layer_idx, layer_vm_usage in enumerate(network_vm_usage):
            layer_fit_vm, vm_capacity, total_vm_req = layer_vm_usage
            if total_vm_req > vm_capacity:
                print(f"Network {subnet_name}, layer {layer_idx} needs VM {total_vm_req}, which exceeds VM capacity {vm_capacity}")
        return False
    else:
        constraint_stats['vm_feasible_subnets'] += 1

    return True

def check_nvm_constraint(performance_model: PlatPerf, subnet_obj, subnet_name, subnet_cpb, constraint_stats):
    if performance_model.PLAT_SETTINGS['NVM_CAPACITY'] <= 0:
        # print if some constraint is skipped
        print('Memory constraints are skipped!')
        return True

    all_layers_fit_nvm, network_nvm_usage, _ = performance_model.get_nvm_usage(subnet_obj)

    if not all_layers_fit_nvm:
        return False
    else:
        constraint_stats['nvm_feasible_subnets'] += 1

    return True

def check_constraints(performance_model: PlatPerf, subnet_latency_info, subnet_obj, subnet_name, subnet_cpb, checked_constraints, constraint_stats):
    pass_constraints = True

    for key in CONSTRAINT_STAT_KEYS:
        constraint_stats.setdefault(key, 0)
    for key in CONSTRAINT_MAX_KEYS:
        constraint_stats.setdefault(key, float('-inf'))
    for key in CONSTRAINT_MIN_KEYS:
        constraint_stats.setdefault(key, float('inf'))

    # checking IMC or latency will also check others via find_best_solution
    basic_constraints_checked = 'CHK_PASS_IMC' in checked_constraints or 'CHK_PASS_RESPONSIVENESS' in checked_constraints

    if subnet_latency_info:
        int_mng_cost_proportion = subnet_latency_info['imc_prop']
        latency = subnet_latency_info['perf_e2e_intpow_lat']
        ip_tot_npc = subnet_latency_info['ip_tot_npc']

        print(f'{subnet_name} latency={latency} IMC={int_mng_cost_proportion} npc={ip_tot_npc}')

        def update_stats(kind, value):
            if value == -1:
                return
            max_value = constraint_stats[f'{kind}_max'] = max(value, constraint_stats[f'{kind}_max'])
            min_value = constraint_stats[f'{kind}_min'] = min(value, constraint_stats[f'{kind}_min'])
            print(f'{kind} min={min_value}, max={max_value}')

        update_stats('imc', int_mng_cost_proportion)
        update_stats('latency', latency)
        update_stats('npc', ip_tot_npc)

    else:
        int_mng_cost_proportion = latency = ip_tot_npc = -1

    checked_constraints = checked_constraints.split(',')
    if 'CHK_PASS_IMC' in checked_constraints:
        if performance_model.PLAT_SETTINGS['IMC_CONSTRAINT'] > 0:
            pass_imc_constraint, _, _ = cnn.pass_constraint_imc(int_mng_cost_proportion, performance_model.PLAT_SETTINGS)

            if pass_imc_constraint:
                constraint_stats['imc_feasible_subnets'] += 1

            pass_constraints = pass_constraints and pass_imc_constraint  # introduced in inas2
        else:
            print('IMC constraint is skipped!')

    if 'CHK_PASS_RESPONSIVENESS' in checked_constraints:
        if performance_model.PLAT_SETTINGS['LAT_E2E_REQ'] > 0:
            pass_latency_constraint, _, _ = cnn.pass_constraint_responsiveness(latency, performance_model.PLAT_SETTINGS)

            if pass_latency_constraint:
                constraint_stats['latency_feasible_subnets'] += 1

            pass_constraints = pass_constraints and pass_latency_constraint
        else:
            print('Latency constraint is skipped!')

    if not basic_constraints_checked:
        if 'CHK_PASS_SPATIAL' in checked_constraints:
            pass_constraints = pass_constraints and check_vm_constraint(performance_model, subnet_obj, subnet_name, subnet_cpb, constraint_stats)
        if 'CHK_PASS_STORAGE' in checked_constraints:
            pass_constraints = pass_constraints and check_nvm_constraint(performance_model, subnet_obj, subnet_name, subnet_cpb, constraint_stats)
        if 'CHK_PASS_ATOMICITY' in checked_constraints:
            # TODO
            pass
    else:
        if int_mng_cost_proportion != -1 and latency != -1 and ip_tot_npc != -1:
            constraint_stats['nvm_feasible_subnets'] += 1

    return pass_constraints

def merge_constraint_stats(all_constraint_stats):
    constraint_stats = {}
    for constraint_stats_per_cpu in all_constraint_stats:
        for key in constraint_stats_per_cpu.keys():
            if key not in constraint_stats:
                constraint_stats[key] = constraint_stats_per_cpu[key]
            else:
                if key in CONSTRAINT_MIN_KEYS:
                    constraint_stats[key] = min(constraint_stats[key], constraint_stats_per_cpu[key])
                elif key in CONSTRAINT_MAX_KEYS:
                    constraint_stats[key] = max(constraint_stats[key], constraint_stats_per_cpu[key])
                else:
                    constraint_stats[key] += constraint_stats_per_cpu[key]
    return constraint_stats
