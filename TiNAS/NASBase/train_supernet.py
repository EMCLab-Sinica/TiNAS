'''
Adapted from : https://github.com/ShunLu91/Single-Path-One-Shot-NAS
'''

import argparse
import logging
import os, sys
import time
import math
from pprint import pprint

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchinfo import summary

from NASBase import utils
from NASBase.model.mnas_arch import MNASSuperNet
#sys.path.append("..")
from settings import Settings, arg_parser, load_settings
from NASBase.model.common_utils import get_dataset
from logger.remote_logger import get_remote_logger_obj


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


trained_architectures = {}
    



# train loop per epoch
# fine_tune_subnet_blkchoices_ixs: for training only the specified subnet (if not None)
def train(device, global_settings : Settings, tot_epochs, cur_epoch, train_loader, model: MNASSuperNet, criterion, optimizer,
          mode_txt, fine_tune_subnet_blkchoices_ixs=None):
    
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)
    
    dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    num_choices_per_block = model.blk_choices #model.choices #model.module.choices
    print("num_choices_per_block: ", len(num_choices_per_block))
    # print("----------------")
    # print("network choices:")
    # pprint(num_choices_per_block)
    # print("----------------")
    
    num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']
    
    print_freq = global_settings.NAS_SETTINGS_GENERAL['TRAIN_PRINT_FREQ']
    
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if (fine_tune_subnet_blkchoices_ixs==None):
            choices = utils.random_choice(len(num_choices_per_block), num_blocks)
        else:
            choices = fine_tune_subnet_blkchoices_ixs
        
        #print(model._debug_get_tot_num_layers(choices))
        
        #print("-- choices: ", choices)
        outputs = model(inputs, choices)
        loss = criterion(outputs, targets)
        loss.backward()
        assert not torch.isnan(loss)
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        train_loss.update(loss.item(), n)
        train_acc.update(prec1.item(), n)
        if (step % print_freq == 0) or (step == (len(train_loader) - 1)):
            logging.info(
                '[%s Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
                'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
                % (mode_txt, lr, cur_epoch+1, tot_epochs, step+1, steps_per_epoch,
                   loss.item(), train_loss.avg, prec1, train_acc.avg)
            )
    return train_loss.avg, train_acc.avg
    

# validate loop per epoch
def validate(device, global_settings : Settings, val_loader, model, criterion,
             fine_tune_subnet_blkchoices_ixs=None):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    
    dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    num_choices_per_block = model.blk_choices
    num_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']
    
    max_prec1, min_prec1 = 0, 100

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if (fine_tune_subnet_blkchoices_ixs==None):
                choices = utils.random_choice(len(num_choices_per_block), num_blocks)
            else:
                choices = fine_tune_subnet_blkchoices_ixs
                
            #print("-- choices: ", choices)
            outputs = model(inputs, choices)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)

            max_prec1 = max(max_prec1, prec1)
            min_prec1 = min(min_prec1, prec1)
    
    # report min and max val_acc
    if (fine_tune_subnet_blkchoices_ixs==None):
        logging.info('[Supernet Validation] max prec1: %.3f, min prec1: %.3f' % (max_prec1, min_prec1))
    else:
        logging.info('[Subnet Fine-Tune Validation] max prec1: %.3f, min prec1: %.3f' % (max_prec1, min_prec1))

    return val_loss.avg, val_acc.avg







def run_supernet_train(global_settings: Settings, dataset=None, supernet_chkpt_fname=None, supernet=None,
                       fine_tune_subnet_blkchoices_ixs=None, train_epochs=None):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("run_supernet_train::Enter (fine_tune_subnet_blkchoices_ixs = {})".format(fine_tune_subnet_blkchoices_ixs))
        
    #logging.info(args)
    utils.set_seed(global_settings.NAS_SETTINGS_GENERAL['SEED'])
     
    # -- Check Checkpoints Directory
    ckpt_dir = global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR']
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    # -- Define Supernet 
    if dataset == None:
        dataset =  global_settings.NAS_SETTINGS_GENERAL['DATASET']
    
    
    # -- create supernet --
    if supernet == None:
        test_out_ch_scale = 1.0    
        block_out_channels =  [math.ceil(test_out_ch_scale * c) for c in global_settings.NAS_SETTINGS_PER_DATASET[dataset]['OUT_CH_PER_BLK']]
        print("--- Generating the SuperNet")
        #print("block_out_channels = ", block_out_channels)        
        model = MNASSuperNet(global_settings, dataset, block_out_channels)
    else:
        model = supernet
    
    #summary(model, depth=2, input_size=(1, 3, 32, 32))    
    #logging.info(model)
    
    if (fine_tune_subnet_blkchoices_ixs==None):
        train_epochs = train_epochs or global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUPERNET_EPOCHS']
        lr = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_OPT_LR']
        trainset_batchsize = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUBNET_BATCHSIZE']
        mode_txt = "Supernet"
    else:
        train_epochs = train_epochs or global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_SUBNET_EPOCHS']
        lr = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_OPT_LR']
        trainset_batchsize = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['FINETUNE_BATCHSIZE']
        mode_txt = "Subnet Fine-Tune"

    # -- get dataset
    _, input_resolution = model.net_choices
    train_loader, val_loader = get_dataset(global_settings, input_resolution=input_resolution, trainset_batchsize=trainset_batchsize)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr,
                                momentum=global_settings.NAS_SETTINGS_GENERAL['TRAIN_OPT_MOM'], 
                                weight_decay=global_settings.NAS_SETTINGS_GENERAL['TRAIN_OPT_WD']
                                )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs)
    print('\n')

    rlog = None
    if global_settings.GLOBAL_SETTINGS['USE_REMOTE_LOGGER']:
        rlog = get_remote_logger_obj(global_settings)

    if rlog and supernet_chkpt_fname:
        rlog.save(supernet_chkpt_fname)

    val_loss, val_acc = validate(device, global_settings, val_loader, model, criterion,
                                 fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
    logging.info(
        '[%s Validation] Before training val_loss: %.3f, val_acc: %.3f'
        % (mode_txt, val_loss, val_acc)
    )

    print("=== Starting Main Training Loop ===")

    # -- Training main loop
    start = time.time()
    best_val_acc = 0.0
    for epoch in range(train_epochs):
        
        # Supernet Training
        train_loss, train_acc = train(device, global_settings, train_epochs, epoch, train_loader, model, criterion, optimizer, mode_txt,
                                      fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
        scheduler.step()
        logging.info(
            '[%s Training] epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
            (mode_txt, epoch + 1, train_loss, train_acc)
        )
        
        # Supernet Validation
        val_loss, val_acc = validate(device, global_settings, val_loader, model, criterion,
                                     fine_tune_subnet_blkchoices_ixs=fine_tune_subnet_blkchoices_ixs)
        
        # Save Best Supernet Weights
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss            
            if supernet_chkpt_fname is not None:
                torch.save(model.state_dict(), supernet_chkpt_fname)
                logging.info('Save best checkpoints to %s' % supernet_chkpt_fname)
            else:
                logging.warning('Model checkpoint filename is not specified, so the best checkpoint cannot be saved')
        logging.info(
            '[%s Validation] epoch: %03d, val_loss: %.3f, val_acc: %.3f, best_acc: %.3f'
            % (mode_txt, epoch + 1, val_loss, val_acc, best_val_acc)
        )

        if rlog:
            rlog.log({
                'mode': mode_txt,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
            })

        print('\n')

    return supernet_chkpt_fname, best_val_acc, best_val_loss






if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #'0,1'

    test_settings = Settings() # default settings
    test_settings = arg_parser(test_settings)
    run_supernet_train(test_settings)
    
        
    

