from os.path import dirname, realpath
import sys

import numpy as np
import torch

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from settings import Settings
from NASBase import file_utils, multiprocessing_helper
from NASBase.model.common_utils import (
    blkchoices_to_blkchoices_ixs, get_network_dimension, get_network_obj, get_subnet, iter_blk_choices, netobj_to_pyobj,
)
from NASBase.model.mnas_ss import (
    EXP_FACTORS, KERNEL_SIZES, MOBILENET_NUM_LAYERS_EXPLICIT, SUPPORT_SKIP,
)
from NASBase.model.mnas_subnet_train import train
from NASBase.ss_optimization.ss_opt_flops import reorganize_flops_data

NUM_TOP_SUBNETS = 5


def sort_by_flops(data):
    flops, subnet_data = data
    return flops

def gpu_worker(gpuid, global_settings, dataset, supernet_blk_choices, batched_subnets_data):
    print(f'gpu_worker entered: worker {gpuid}')

    subnet_results = []

    for flops, subnet_data, subnet_idx, supernet_choice in batched_subnets_data:
        print(f'Processing subnet {subnet_idx}')

        width_multiplier, input_resolution = supernet_choice

        subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(supernet_blk_choices, subnet_data['subnet_choice_per_blk'])

        each_subnet = get_subnet(global_settings, dataset, supernet_blk_choices, subnet_choice_per_blk_ixs, subnet_idx, width_multiplier, input_resolution)
        # Disable workers for the data loader as multiprocessing pool cannot be nested
        max_val_acc = train(each_subnet, gpuid, global_settings, input_resolution=input_resolution, num_data_loader_workers=0)

        net_input = torch.rand(1, 3, input_resolution, input_resolution).to(torch.device(f'cuda:{gpuid}'))

        subnet_dims = get_network_dimension(each_subnet, input_tensor = net_input)
        subnet_obj = get_network_obj(subnet_dims)

        subnet_name = each_subnet.name
        subnet_obj = netobj_to_pyobj(subnet_obj)
        assert subnet_obj == subnet_data['subnet_obj']

        subnet_results.append({
            "subnet_obj" : subnet_obj,
            "subnet_name" : subnet_name,
            "subnet_choice_per_blk": subnet_data['subnet_choice_per_blk'],
            "net_choices" : each_subnet.net_choices,
            "width_multiplier": width_multiplier,
            "input_resolution": input_resolution,
            "flops": flops,

            # validation accuracy
            "max_val_acc": max_val_acc,
        })

    return subnet_results

def main():
    dataset = 'CIFAR10'

    global_settings = Settings()
    # override number of epochs
    # global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUBNET_EPOCHS'] = 1

    all_subnets_with_flops = reorganize_flops_data(file_utils.json_load(sys.argv[1]))

    supernet_blk_choices = iter_blk_choices(EXP_FACTORS, KERNEL_SIZES, MOBILENET_NUM_LAYERS_EXPLICIT, SUPPORT_SKIP)

    # dispatch workers (1 per gpu, each processing a batch of subnets)    
    all_subnets_data = []

    # Make it a function
    for supernet_choice in all_subnets_with_flops.keys():

        cur_subnets_with_flops = all_subnets_with_flops[supernet_choice]
        cur_subnets_with_flops.sort(key=sort_by_flops, reverse=True)
        num_top_subnets = NUM_TOP_SUBNETS
        cur_subnets_with_flops = cur_subnets_with_flops[:num_top_subnets]

        # use get_subnet
        for subnet_idx, (flops, subnet_data) in enumerate(cur_subnets_with_flops):
            # Re-index subnets as original subnet indices from different supernets may not be unique
            subnet_idx = len(all_subnets_data)
            print(f'Re-indexing subnet {subnet_idx}')
            all_subnets_data.append((flops, subnet_data, subnet_idx, supernet_choice))

    available_gpus = multiprocessing_helper.get_max_num_workers(worker_type='GPU')

    #pprint(len(all_subnets_lst)); sys.exit()
    batched_subnets_data = np.array_split(all_subnets_data, available_gpus)

    subnet_results = multiprocessing_helper.run_multiprocessing_workers(
        num_workers=available_gpus,
        worker_func=gpu_worker,
        worker_type='GPU',
        common_args=(global_settings, dataset, supernet_blk_choices),
        worker_args=batched_subnets_data,
    )

    logfname = global_settings.LOG_SETTINGS['TRAIN_LOG_DIR']  + 'results_subnet_accuracy.json'

    # overwrite json
    file_utils.delete_file(logfname)
    file_utils.json_dump(logfname, subnet_results)

if __name__ == '__main__':
    main()

