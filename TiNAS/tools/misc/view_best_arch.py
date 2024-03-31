import argparse
import logging
import os
import sys
import time
import math
import numpy as np
import itertools

import torch
import torch.nn as nn
import torchvision

sys.path.append("../..")
from settings import Settings, arg_parser, load_settings

sys.path.append("..")
import utils as utils
from model.mnas_arch import MNASSuperNet
from model.mnas_ss import *





def get_arch_name(choice):
    name = "arch_"
    for each_blk_choice in choice:        
        #s = '-'.join(str(c) for c in each_blk_choice)
        name = name + "_" + str(each_blk_choice)
    return name
    

def evaluate_single_path(device, val_loader, model, criterion, choice):
    print("evaluate_single_path::Enter: ", str(choice))
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    #print(choice); sys.exit()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)            
            outputs = model(inputs, choice)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)
    return val_loss.avg, val_acc.avg


if __name__ == '__main__':


    # get experiment settings
    #device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3' #'0,1'
    
    global_settings = Settings() # default settings
    global_settings = arg_parser(global_settings)
    dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    test_out_ch_scale = 1.0    
    block_out_channels =  [math.ceil(test_out_ch_scale * c) for c in global_settings.NAS_SETTINGS_PER_DATASET[dataset]['OUT_CH_PER_BLK']]


    # Load Pretrained Supernet
    model = MNASSuperNet(global_settings, dataset, block_out_channels)
    
    #total_subnets = [len(choices) for choices in model.choice_blocks]
    #print(total_subnets); sys.exit()
        
    best_supernet_weights = '../checkpoints_v1_18choices/test_mnas_oneshot_train_best.pth'
    checkpoint = torch.load(best_supernet_weights, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    #print("here --"); sys.exit()
    model = model.to(device)
    
    logging.info('Finish loading checkpoint from %s', best_supernet_weights)
    criterion = nn.CrossEntropyLoss().to(device)
    
    
    
    # Dataset Definition
    _, valid_transform = utils.data_transforms(dataset)
    valset = torchvision.datasets.CIFAR10(#root= "../dataset/",
                                        root="."+global_settings.NAS_SETTINGS_PER_DATASET['CIFAR10']['TRAIN_DATADIR'], 
                                        train=False, download=False, transform=valid_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=global_settings.NAS_SETTINGS_PER_DATASET['CIFAR10']['VAL_BATCHSIZE'],
                                                shuffle=False, pin_memory=True, num_workers=8)

    # get accuracy for each subnet
    num_choices_per_block = len(model.choices)
    num_blocks = model.num_blocks
    all_subnet_acc = []
    
    choices_per_block = itertools.product(np.arange(num_choices_per_block), repeat=num_blocks)
    len_choices_per_block = len([x for x in itertools.product(np.arange(num_choices_per_block), repeat=num_blocks)])
    print("choices_per_block= ", len_choices_per_block)

    for cix, each_choice in enumerate(choices_per_block):    
        
        val_loss, val_acc = evaluate_single_path(device, val_loader, model, criterion, list(each_choice))
        subnet_result = {
            'arch_name': get_arch_name(each_choice),
            'arch_config': each_choice,
            'val_acc': val_acc,
            'val_loss': val_loss
        }
        all_subnet_acc.append(subnet_result)
        
        print("Finished evaluating subnet {}/{}: {} | val_acc={}".format(cix, len_choices_per_block, subnet_result['arch_name'], val_acc))


        

