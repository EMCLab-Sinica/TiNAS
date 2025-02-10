from torch import nn

from settings import Settings
from NASBase.train_supernet import run_supernet_train
from NASBase.train_supernet_distributed import run_supernet_train_distributed

# Keep only selected block choices, effectively drop others
def strip_supernet(supernet, fine_tune_subnet_blkchoices_ixs):
    for block_idx in range(len(supernet.choice_blocks)):
        selected_block_choice = supernet.choice_blocks[block_idx][fine_tune_subnet_blkchoices_ixs[block_idx]]
        supernet.choice_blocks[block_idx] = nn.ModuleList(modules=[selected_block_choice])
    # torchsummary for supernet to check if blocks are correctly dropped or not

def fine_tune_best_solution(global_settings: Settings, dataset, supernet, supernet_chkpt_fname, best_supernet_config, best_solution):
    width_multiplier, input_resolution = best_supernet_config
    best_info = best_solution[1]

    # Disable workers for the data loader as multiprocessing pool cannot be nested
    fine_tune_subnet_blkchoices_ixs = best_info['subnet_choice_per_blk_ixs']
    if False:
        strip_supernet(supernet, fine_tune_subnet_blkchoices_ixs)
        fine_tune_subnet_blkchoices_ixs = [0] * len(supernet.choice_blocks)
    if not global_settings.NAS_SETTINGS_PER_DATASET[dataset].get('TRAIN_MULTI_GPU', False):
        fine_tune_result = run_supernet_train(global_settings, dataset, supernet_chkpt_fname.replace('.pth', '-fine-tuned.pth'), supernet,
                                              fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
    else:
        fine_tune_result = run_supernet_train_distributed(global_settings, dataset, supernet_chkpt_fname.replace('.pth', '-fine-tuned.pth'), supernet,
                                                          fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
    supernet_chkpt_fname, best_val_acc, best_val_loss = fine_tune_result

    best_info['fine_tune_result'] = {
        'supernet_chkpt_fname': supernet_chkpt_fname,
        'max_val_acc': best_val_acc,
        'min_val_loss': best_val_loss,
    }

    return best_solution
