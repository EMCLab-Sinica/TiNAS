import sys, os
from pprint import pprint
import numpy as np
import math
import random
import copy
from os.path import dirname, realpath, join

import torch
import torch.nn as nn
import torchvision
from torchinfo import summary
import inspect
import itertools

import pandas as pd
from torch.utils.data import TensorDataset
import torchvision.transforms.functional

from settings import Settings

from .common_types import LAYERTYPES, OPTYPES, Mat
from .. import utils

from .mnas_arch import MNASSubNet, MNASSuperNet


#from ..dataset.kws import *

# stride ?
def get_ofm_tile_dim(th, op_dim):
    tr = (((th - op_dim["K"]) + (2*op_dim["pad"]))/op_dim["stride"]) + 1
    return int(np.floor(tr))

def get_ofm_dim(op_dim):
    R = (((op_dim["H"] - op_dim["K"]) + (2*op_dim["pad"]))/op_dim["stride"]) + 1
    return int(np.floor(R))



def get_largest_feasible_tilesize():
    pass



def get_network_dimension_1d(model: nn.Module, input_tensor):   
    assert len(input_tensor.shape) == 3
    
    output = input_tensor
    model_dims = [] 
    prev_layer_dims =   {
                    "name": "input",
                    "H": None, "W": None,                     
                    "N": None,  # input channels
                    "M": input_tensor.shape[1], # output channels
                    "R": input_tensor.shape[2],
                    "K": None                    
                } 
    
    for name, m in model.named_modules():        
        
        # dirty hack to handle the residual aggregation
        include_residual_aggr = False
        residual_aggr_op = {}
        
        if not isinstance(m, nn.Sequential):
            
            cur_layer_dims = None
            #print(name, m, output.shape)    
            #print(name)    
            
            #======== CONV ==============
            if isinstance(m, nn.Conv1d):                
                conv_type = OPTYPES.get_conv1d_optype_by_name(name)
                if conv_type == OPTYPES.O_CONV1D_DW: # depthwise
                    output = m(output)
                    cur_layer_dims = {
                        "name" : name, "objtype": str(type(m)), 
                        "op_type": conv_type,
                        "H": prev_layer_dims['R'],
                        "N": 1, "M": output.shape[1], 
                        "R": output.shape[2],
                        "K": m.kernel_size[0],
                        "stride" : m.stride[0], "pad" : m.padding[0] # assuming uniform padding and uniform stride
                    }                    
                else:   # pointwise, std conv
                    
                    if "skip" in name:  # CONV part of the SKIP connection
                        # conv0_pw => the first conv operation inside mbconv
                        mbconv_conv0_pw_dims = get_op_by_name_from_netdims_obj(model_dims, "mbconv_conv0_pw", reverse=True)    
                        input_tensor = torch.rand(1, mbconv_conv0_pw_dims["N"], mbconv_conv0_pw_dims["H"])
                        output = m(input_tensor)
                        cur_layer_dims = {
                            "name" : name, "objtype": str(type(m)), 
                            "op_type": conv_type,
                            "H": mbconv_conv0_pw_dims['H'],
                            "N": mbconv_conv0_pw_dims["N"], "M": output.shape[1], 
                            "R": output.shape[2],
                            "K": m.kernel_size[0],
                            "stride" : m.stride[0], "pad" : m.padding[0] # assuming uniform padding and uniform stride
                        }
                    else:
                        output = m(output)
                        cur_layer_dims = {
                            "name" : name, "objtype": str(type(m)), 
                            "op_type": conv_type,
                            "H": prev_layer_dims['R'],
                            "N": prev_layer_dims['M'], "M": output.shape[1], 
                            "R": output.shape[2],
                            "K": m.kernel_size[0],
                            "stride" : m.stride[0], "pad" : m.padding[0] # assuming uniform padding and uniform stride
                        }
            
            #======== BN ==============                             
            elif isinstance(m, nn.BatchNorm1d):
                if "skip" in name:  # BN part of the SKIP connection
                    # skip_conv3_pw = the 1x1 conv operation belonging to the skip, inside mbconv
                    mbconv_skip_conv3_pw_dims = get_op_by_name_from_netdims_obj(model_dims, "skip_conv3_pw", reverse=True) 
                    input_tensor = torch.rand(1, mbconv_skip_conv3_pw_dims["M"], mbconv_skip_conv3_pw_dims["R"])
                    output = m(input_tensor)
                    cur_layer_dims = {
                        "name" : name, "objtype": str(type(m)),
                        "op_type": OPTYPES.O_BN,
                        "H": mbconv_skip_conv3_pw_dims['R'],
                        "N": mbconv_skip_conv3_pw_dims['M'], "M": output.shape[1], 
                        "R": mbconv_skip_conv3_pw_dims['R'],
                        "K": None                   
                    }   
                    
                    #-- residual aggregation --
                    include_residual_aggr = True
                    name_prefix = mbconv_skip_conv3_pw_dims['name'].replace("skip_conv3_pw", "")     
                    residual_aggr_op_dims = {
                        "name" : name_prefix+"skip_aggr", "objtype": "tensor.add",
                        "op_type": OPTYPES.O_ADD,
                        "H": mbconv_skip_conv3_pw_dims['R'],
                        "N": output.shape[1], "M": output.shape[1], 
                        "R": mbconv_skip_conv3_pw_dims['R'],
                        "K": None   
                    }                         
                else:                
                    output = m(output)
                    cur_layer_dims = {
                        "name" : name, "objtype": str(type(m)),
                        "op_type": OPTYPES.O_BN,
                        "H": prev_layer_dims['R'],
                        "N": prev_layer_dims['M'], "M": output.shape[1], 
                        "R": prev_layer_dims['R'],
                        "K": None                   
                    }                                
            
            #======== RELU ==============                             
            elif isinstance(m, nn.ReLU):
                output = m(output)
                cur_layer_dims = {
                    "name" : name, "objtype": str(type(m)),
                    "op_type": OPTYPES.O_RELU,
                    "H": prev_layer_dims['R'],
                    "N": prev_layer_dims['M'], "M": output.shape[1], 
                    "R": prev_layer_dims['R'],
                    "K": None                   
                }
                
            
            #======== IDENTITY ==============                               
            elif isinstance(m, nn.Identity):
                #-- residual aggregation --
                # conv0_pw = the first conv operation inside mbconv
                mbconv_conv0_pw_dims = get_op_by_name_from_netdims_obj(model_dims, "mbconv_conv0_pw", reverse=True)    
                input_tensor = torch.rand(1, mbconv_conv0_pw_dims["N"], mbconv_conv0_pw_dims["H"])
                include_residual_aggr = True    
                name_prefix = mbconv_conv0_pw_dims['name'].replace("op.mbconv_conv0_pw", "")     
                residual_aggr_op_dims = {
                    "name" : name_prefix+"shortcut.skip_aggr", "objtype": "tensor.add",
                    "op_type": OPTYPES.O_ADD,
                    "H": mbconv_conv0_pw_dims['H'],
                    "N": mbconv_conv0_pw_dims['N'], "M": mbconv_conv0_pw_dims['N'], # identity summation - inch==outch
                    "R": mbconv_conv0_pw_dims['H'],
                    "K": None   
                }         
            
            #======== AVG POOL ==============                                 
            elif isinstance(m, nn.AdaptiveAvgPool1d):
                output = m(output)
                cur_layer_dims = {
                    "name" : name, "objtype": str(type(m)),
                    "op_type": OPTYPES.O_AVGPOOL,
                    "H": prev_layer_dims['R'],
                    "N": prev_layer_dims['M'], "M": output.shape[1], 
                    "R": 1, "C": 1,
                    "K": None                   
                }
            
            #======== LINEAR ==============                                 
            # typically last layer
            elif isinstance(m, nn.Linear):
                output = m(output.view(output.size(0),-1))                
                cur_layer_dims = {
                    "name" : name, "objtype": str(type(m)),
                    "op_type": OPTYPES.O_FC,
                    "H": prev_layer_dims['R'],
                    "N": prev_layer_dims['M'], "M": m.out_features, 
                    "R": 1, "C": 1,
                    "K": None   # not sure ?                
                }
            
            else:                
                #raise BaseException("get_network_dimension::Error - unknown model type: {}, {}".format(name, m) )                                
                pass
                            
            if cur_layer_dims != None:
                prev_layer_dims = cur_layer_dims
                #pprint(cur_layer_dims)                
                model_dims.append(cur_layer_dims)
                
            if include_residual_aggr == True:
                model_dims.append(residual_aggr_op_dims)
        
    #print("------------------")
    return model_dims

def get_network_dimension(model: nn.Module, input_tensor):   
    if len(input_tensor.shape) == 3:
        return get_network_dimension_1d(model, input_tensor)
    
    output = input_tensor
    model_dims = [] 
    prev_layer_dims =   {
                    "name": "input",
                    "H": None, "W": None,                     
                    "N": None,  # input channels
                    "M": input_tensor.shape[1], # output channels
                    "R": input_tensor.shape[2], "C": input_tensor.shape[3], 
                    "K": None                    
                } 
    
    for name, m in model.named_modules():        
        
        # dirty hack to handle the residual aggregation
        include_residual_aggr = False
        residual_aggr_op = {}
        
        if not isinstance(m, nn.Sequential):
            
            cur_layer_dims = None
            #print(name, m, output.shape)    
            #print(name)    
            
            #======== CONV ==============
            if isinstance(m, nn.Conv2d):                
                conv_type = OPTYPES.get_conv_optype_by_name(name)
                if conv_type == OPTYPES.O_CONV2D_DW: # depthwise
                    output = m(output)
                    cur_layer_dims = {
                        "name" : name, "objtype": str(type(m)), 
                        "op_type": conv_type,
                        "H": prev_layer_dims['R'], "W": prev_layer_dims['C'], 
                        "N": 1, "M": output.shape[1], 
                        "R": output.shape[2], "C": output.shape[3], 
                        "K": m.kernel_size[0],
                        "stride" : m.stride[0], "pad" : m.padding[0] # assuming uniform padding and uniform stride
                    }                    
                else:   # pointwise, std conv
                    
                    if "skip" in name:  # CONV part of the SKIP connection
                        # conv0_pw => the first conv operation inside mbconv
                        mbconv_conv0_pw_dims = get_op_by_name_from_netdims_obj(model_dims, "mbconv_conv0_pw", reverse=True)    
                        input_tensor = torch.rand(1, mbconv_conv0_pw_dims["N"], mbconv_conv0_pw_dims["H"], mbconv_conv0_pw_dims["W"])
                        output = m(input_tensor)
                        cur_layer_dims = {
                            "name" : name, "objtype": str(type(m)), 
                            "op_type": conv_type,
                            "H": mbconv_conv0_pw_dims['H'], "W": mbconv_conv0_pw_dims['W'], 
                            "N": mbconv_conv0_pw_dims["N"], "M": output.shape[1], 
                            "R": output.shape[2], "C": output.shape[3], 
                            "K": m.kernel_size[0],
                            "stride" : m.stride[0], "pad" : m.padding[0] # assuming uniform padding and uniform stride
                        }
                    else:
                        output = m(output)
                        cur_layer_dims = {
                            "name" : name, "objtype": str(type(m)), 
                            "op_type": conv_type,
                            "H": prev_layer_dims['R'], "W": prev_layer_dims['C'], 
                            "N": prev_layer_dims['M'], "M": output.shape[1], 
                            "R": output.shape[2], "C": output.shape[3], 
                            "K": m.kernel_size[0],
                            "stride" : m.stride[0], "pad" : m.padding[0] # assuming uniform padding and uniform stride
                        }
            
            #======== BN ==============                             
            elif isinstance(m, nn.BatchNorm2d):
                if "skip" in name:  # BN part of the SKIP connection
                    # skip_conv3_pw = the 1x1 conv operation belonging to the skip, inside mbconv
                    mbconv_skip_conv3_pw_dims = get_op_by_name_from_netdims_obj(model_dims, "skip_conv3_pw", reverse=True) 
                    input_tensor = torch.rand(1, mbconv_skip_conv3_pw_dims["M"], mbconv_skip_conv3_pw_dims["R"], mbconv_skip_conv3_pw_dims["C"])
                    output = m(input_tensor)
                    cur_layer_dims = {
                        "name" : name, "objtype": str(type(m)),
                        "op_type": OPTYPES.O_BN,
                        "H": mbconv_skip_conv3_pw_dims['R'], "W": mbconv_skip_conv3_pw_dims['C'], 
                        "N": mbconv_skip_conv3_pw_dims['M'], "M": output.shape[1], 
                        "R": mbconv_skip_conv3_pw_dims['R'], "C": mbconv_skip_conv3_pw_dims['C'],
                        "K": None                   
                    }   
                    
                    #-- residual aggregation --
                    include_residual_aggr = True
                    name_prefix = mbconv_skip_conv3_pw_dims['name'].replace("skip_conv3_pw", "")     
                    residual_aggr_op_dims = {
                        "name" : name_prefix+"skip_aggr", "objtype": "tensor.add",
                        "op_type": OPTYPES.O_ADD,
                        "H": mbconv_skip_conv3_pw_dims['R'], "W": mbconv_skip_conv3_pw_dims['C'], 
                        "N": output.shape[1], "M": output.shape[1], 
                        "R": mbconv_skip_conv3_pw_dims['R'], "C": mbconv_skip_conv3_pw_dims['C'],
                        "K": None   
                    }                         
                else:                
                    output = m(output)
                    cur_layer_dims = {
                        "name" : name, "objtype": str(type(m)),
                        "op_type": OPTYPES.O_BN,
                        "H": prev_layer_dims['R'], "W": prev_layer_dims['C'], 
                        "N": prev_layer_dims['M'], "M": output.shape[1], 
                        "R": prev_layer_dims['R'], "C": prev_layer_dims['C'],
                        "K": None                   
                    }                                
            
            #======== RELU ==============                             
            elif isinstance(m, nn.ReLU):
                output = m(output)
                cur_layer_dims = {
                    "name" : name, "objtype": str(type(m)),
                    "op_type": OPTYPES.O_RELU,
                    "H": prev_layer_dims['R'], "W": prev_layer_dims['C'], 
                    "N": prev_layer_dims['M'], "M": output.shape[1], 
                    "R": prev_layer_dims['R'], "C": prev_layer_dims['C'],
                    "K": None                   
                }
                
            
            #======== IDENTITY ==============                               
            elif isinstance(m, nn.Identity):
                #-- residual aggregation --
                # conv0_pw = the first conv operation inside mbconv
                mbconv_conv0_pw_dims = get_op_by_name_from_netdims_obj(model_dims, "mbconv_conv0_pw", reverse=True)    
                input_tensor = torch.rand(1, mbconv_conv0_pw_dims["N"], mbconv_conv0_pw_dims["H"], mbconv_conv0_pw_dims["W"])
                include_residual_aggr = True    
                name_prefix = mbconv_conv0_pw_dims['name'].replace("op.mbconv_conv0_pw", "")     
                residual_aggr_op_dims = {
                    "name" : name_prefix+"shortcut.skip_aggr", "objtype": "tensor.add",
                    "op_type": OPTYPES.O_ADD,
                    "H": mbconv_conv0_pw_dims['H'], "W": mbconv_conv0_pw_dims['W'], 
                    "N": mbconv_conv0_pw_dims['N'], "M": mbconv_conv0_pw_dims['N'], # identity summation - inch==outch
                    "R": mbconv_conv0_pw_dims['H'], "C": mbconv_conv0_pw_dims['W'],
                    "K": None   
                }         
            
            #======== AVG POOL ==============                                 
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                output = m(output)
                cur_layer_dims = {
                    "name" : name, "objtype": str(type(m)),
                    "op_type": OPTYPES.O_AVGPOOL,
                    "H": prev_layer_dims['R'], "W": prev_layer_dims['C'], 
                    "N": prev_layer_dims['M'], "M": output.shape[1], 
                    "R": 1, "C": 1,
                    "K": None                   
                }
            
            #======== LINEAR ==============                                 
            # typically last layer
            elif isinstance(m, nn.Linear):
                output = m(output.view(output.size(0),-1))                
                cur_layer_dims = {
                    "name" : name, "objtype": str(type(m)),
                    "op_type": OPTYPES.O_FC,
                    "H": prev_layer_dims['R'], "W": prev_layer_dims['C'], 
                    "N": prev_layer_dims['M'], "M": m.out_features, 
                    "R": 1, "C": 1,
                    "K": None   # not sure ?                
                }
            
            else:                
                #raise BaseException("get_network_dimension::Error - unknown model type: {}, {}".format(name, m) )                                
                pass
                            
            if cur_layer_dims != None:
                prev_layer_dims = cur_layer_dims
                #pprint(cur_layer_dims)                
                model_dims.append(cur_layer_dims)
                
            if include_residual_aggr == True:
                model_dims.append(residual_aggr_op_dims)
        
    #print("------------------")
    return model_dims
    
    
def get_network_obj_1d(net_dims):    
    
    network = []
    
    for opix, each_op in enumerate(net_dims):        
        op_type = each_op['op_type']
        
        if (op_type == OPTYPES.O_CONV1D) or (op_type == OPTYPES.O_CONV1D_PW):
            item = {
                    'name' : "CONV_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "CONV", "objtype": each_op['objtype'], 'optype' : op_type,                                      
                    'stride' : each_op['stride'], 'pad' : each_op["pad"],                                     
                    'K' : Mat(None, each_op['M'], each_op['N'], each_op['K'], 1), # n, ch, h, w
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], 1), 
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], 1),
                }
        
        elif (op_type == OPTYPES.O_CONV1D_DW):
            item = {
                    'name' : "CONV_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "CONV", "objtype": each_op['objtype'], 'optype' : op_type,                    
                    'stride' : each_op['stride'], 'pad' : each_op["pad"],                    
                    'K' : Mat(None, each_op['M'], 1, each_op['K'], 1), # n, ch, h, w
                    'IFM' : Mat(None, 1, each_op['M'], each_op['H'], 1), # in_ch = out_ch
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], 1), # in_ch = out_ch
                }
            
        elif (op_type == OPTYPES.O_BN):  
            item = {
                    'name' : "BN_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "BN", "objtype": each_op['objtype'], 'optype' : op_type,                    
                    'stride' : None,                     
                    'K' : Mat(None, None, None, None, None), # (mu, sigma) per channel
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], 1),
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], 1), 
                } 
            
        elif (op_type == OPTYPES.O_ADD):
            item = {
                    'name' : "ADD_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "ADD", "objtype": each_op['objtype'], 'optype' : op_type,                   
                    'stride' : None,                     
                    'K' : Mat(None, None, None, None, None),
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], 1), # 2 of these IFMs
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], 1), 
                } 
                        
        elif (op_type == OPTYPES.O_RELU):   
            item = {
                    'name' : "RELU_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "RELU", "objtype": each_op['objtype'], 'optype' : op_type,                   
                    'stride' : None,                     
                    'K' : Mat(None, None, None, None, None), # (mu, sigma) per channel
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], 1),
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], 1), 
                }              
        
        elif (op_type == OPTYPES.O_AVGPOOL):     
            item = {
                    'name' : "GAVGPOOL_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "GAVGPOOL", "objtype": each_op['objtype'], 'optype' : op_type,                    
                    'stride' : None,                     
                    'K' : Mat(None, None, None, None, None),
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], 1), # n, ch, h, w
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], 1), 
                }              
        
        # typically last layer
        elif (op_type == OPTYPES.O_FC):   
            item = {
                    'name' : "FC_"+str(opix) if "classifier" not in each_op['name'] else "FC_END", 
                    'alias': each_op['name'], 
                    'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "FC", "objtype": each_op['objtype'], 'optype' : op_type,                    
                    'stride' : 1,                     
                    'K' : Mat(None, each_op['M'], each_op['N'], each_op['H'], 1),    # n, ch, h, w
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], 1),
                    'OFM' : Mat(None, 1, each_op['M'], 1, 1),
                }              
            
        else:
            raise ValueError('get_network_obj::Error - Wrong op type {}'.format(op_type))
        
        network.append(item)
    
    return network

def get_network_obj(net_dims):    

    if 'W' not in net_dims[0]:
        return get_network_obj_1d(net_dims)
    
    network = []
    
    for opix, each_op in enumerate(net_dims):        
        op_type = each_op['op_type']
        
        if (op_type == OPTYPES.O_CONV2D) or (op_type == OPTYPES.O_CONV2D_PW):
            item = {
                    'name' : "CONV_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "CONV", "objtype": each_op['objtype'], 'optype' : op_type,                                      
                    'stride' : each_op['stride'], 'pad' : each_op["pad"],                                     
                    'K' : Mat(None, each_op['M'], each_op['N'], each_op['K'], each_op['K']), # n, ch, h, w
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], each_op['W']), 
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], each_op['C']),
                }
        
        elif (op_type == OPTYPES.O_CONV2D_DW):
            item = {
                    'name' : "CONV_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "CONV", "objtype": each_op['objtype'], 'optype' : op_type,                    
                    'stride' : each_op['stride'], 'pad' : each_op["pad"],                    
                    'K' : Mat(None, each_op['M'], 1, each_op['K'], each_op['K']), # n, ch, h, w
                    'IFM' : Mat(None, 1, each_op['M'], each_op['H'], each_op['W']), # in_ch = out_ch
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], each_op['C']), # in_ch = out_ch
                }
            
        elif (op_type == OPTYPES.O_BN):  
            item = {
                    'name' : "BN_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "BN", "objtype": each_op['objtype'], 'optype' : op_type,                    
                    'stride' : None,                     
                    'K' : Mat(None, None, None, None, None), # (mu, sigma) per channel
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], each_op['W']),
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], each_op['C']), 
                } 
            
        elif (op_type == OPTYPES.O_ADD):
            item = {
                    'name' : "ADD_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "ADD", "objtype": each_op['objtype'], 'optype' : op_type,                   
                    'stride' : None,                     
                    'K' : Mat(None, None, None, None, None),
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], each_op['W']), # 2 of these IFMs
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], each_op['C']), 
                } 
                        
        elif (op_type == OPTYPES.O_RELU):   
            item = {
                    'name' : "RELU_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "RELU", "objtype": each_op['objtype'], 'optype' : op_type,                   
                    'stride' : None,                     
                    'K' : Mat(None, None, None, None, None), # (mu, sigma) per channel
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], each_op['W']),
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], each_op['C']), 
                }              
        
        elif (op_type == OPTYPES.O_AVGPOOL):     
            item = {
                    'name' : "GAVGPOOL_"+str(opix), 'alias': each_op['name'], 'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "GAVGPOOL", "objtype": each_op['objtype'], 'optype' : op_type,                    
                    'stride' : None,                     
                    'K' : Mat(None, None, None, None, None),
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], each_op['W']), # n, ch, h, w
                    'OFM' : Mat(None, 1, each_op['M'], each_op['R'], each_op['C']), 
                }              
        
        # typically last layer
        elif (op_type == OPTYPES.O_FC):   
            item = {
                    'name' : "FC_"+str(opix) if "classifier" not in each_op['name'] else "FC_END", 
                    'alias': each_op['name'], 
                    'lcnt': "{}/{}".format(opix, len(net_dims)),
                    'type' : "FC", "objtype": each_op['objtype'], 'optype' : op_type,                    
                    'stride' : 1,                     
                    'K' : Mat(None, each_op['M'], each_op['N'], each_op['H'], 1),    # n, ch, h, w
                    'IFM' : Mat(None, 1, each_op['N'], each_op['H'], each_op['W']),
                    'OFM' : Mat(None, 1, each_op['M'], 1, 1),
                }              
            
        else:
            raise ValueError('get_network_obj::Error - Wrong op type {}'.format(op_type))
        
        network.append(item)
    
    return network


# get an operation by name, from the given netowork obj list
def get_op_by_name_from_netdims_obj(net_op_list, op_name, reverse=False):
    if reverse == False:
        for opix, each_op in enumerate(net_op_list):   
            if op_name in each_op['name']:
                return each_op
    else:
        for opix, each_op in enumerate(reversed(net_op_list)):   
            if op_name in each_op['name']:
                return each_op
        
    return None


def netobj_to_string(net_obj):
    s = ""
    for layer in net_obj:
        s += str(layer)
        s += "\n"
    return s

def netobj_to_pyobj(net_obj):
    new_net=[]
    for each_layer in net_obj:
        new_layer = dict() 
        for k,v in each_layer.items():
            if k in ["IFM", "OFM", "K"]:
                new_layer[k] = {"N": v.n, "CH": v.ch, "H": v.h, "W": v.w}                
                # new_layer[k] = {"N": int(0 if v.n is None else v.n), 
                #                 "CH": int(0 if v.ch is None else v.ch), 
                #                 "H": int(0 if v.h is None else v.h), 
                #                 "W": int(0 if v.w is None else v.w)
                #                 }
            else:
                new_layer[k]=v
        new_net.append(new_layer)
    return new_net
            
# Inverse of netobj_to_pyobj
def pyobj_to_netobj(py_obj):
    new_net=[]
    for each_layer in py_obj:
        new_layer = dict() 
        for k,v in each_layer.items():
            if k in ["IFM", "OFM", "K"]:
                new_layer[k] = Mat(data=None, n=v["N"], ch=v["CH"], h=v["H"], w=v["W"])
            else:
                new_layer[k] = v
        new_net.append(new_layer)
    return new_net
        
    
          

def view_model(net):
    net_input = torch.rand(1, 3, 32, 32)
    net_input_size = (1, 3, 32, 32)
    stats = summary(net, row_settings=["var_names", "depth"], depth=8, input_size=net_input_size, verbose=1, 
                    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                    col_width =20
                    )
    #sys.exit()
    
    
    # --- debug    
    # print("----- subnet: start ----")
    # for each_sn in subnet_list:
    #     #for name, m in each_s.named_modules():
    #         #if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU):
    #         sn_dim = get_network_dimension(each_sn, input_tensor = net_input)
            
    #         pprint(sn_dim)
    
    # print("----- subnet: end  ----")
    
    #json_dump('test_net.json', sn_dim, indent=4)
    
    # torch.onnx.export(subnet_list[0],               # model being run
    #               net_input,                         # model input (or a tuple for multiple inputs)
    #               "test_net.onnx",   # where to save the model (can be a file or file-like object)
    #               export_params=False,        # store the trained parameter weights inside the model file
    #               #opset_version=10,          # the ONNX version to export the model to
    #               do_constant_folding=False,  # whether to execute constant folding for optimization
    #               input_names = ['input'],   # the model's input names
    #               output_names = ['output'], # the model's output names
    #               #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #               #              'output' : {0 : 'batch_size'}}
    #               )
    
    #writer=SummaryWriter('./logsdir')
    #writer.add_graph(subnet_list[0], net_input)
    
    #torch.save(subnet_list[0].state_dict(), 'test_net.pth')

# get supernet block choices
def iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip):
    choices = []
    for each_exf in k_expfactors:
        for each_ksz in k_kernelsizes:
                for each_nl in k_num_layers_explicit:
                    for each_sk in k_support_skip:
                        choices.append([each_exf, each_ksz, each_nl, each_sk])       # order matters: [CONV_TYPE, KSIZES, NUM_LAYERS, SUPPORT_SKIP]
    return choices

# get net choices
def iter_net_choices(lst_widthmult, lst_inputres):
    choices = []
    for each_wm in lst_widthmult:
        for each_ir in lst_inputres:
            choices.append([each_wm, each_ir])
    return choices

def drop_choices(choices, dropped_choices):
    assert set(choices) > set(dropped_choices)
    return sorted(set(choices) - set(dropped_choices))

def parametric_supernet_choices(global_settings: Settings):

    settings_per_dataset = global_settings.NAS_SETTINGS_PER_DATASET[global_settings.NAS_SETTINGS_GENERAL['DATASET']]

    # types of options
    net_search_options = {
        'WIDTH_MULTIPLIER' : settings_per_dataset['WIDTH_MULTIPLIER'],
        'INPUT_RESOLUTION' : settings_per_dataset['INPUT_RESOLUTION']
    }

    # -- different width multipliers
    lst_widthmult = net_search_options['WIDTH_MULTIPLIER']
    lst_inputres = net_search_options['INPUT_RESOLUTION']

    if global_settings.TINAS['STAGE1_SETTINGS']['DROPPING_ENABLED']:
        net_level_dropped_choices = global_settings.TINAS['STAGE1_SETTINGS']['DROPPING_NET_LEVEL']
        lst_widthmult = drop_choices(lst_widthmult, net_level_dropped_choices['WIDTH_MULTIPLIER'])
        lst_inputres = drop_choices(lst_inputres, net_level_dropped_choices['INPUT_RESOLUTION'])

    supernet_choices = iter_net_choices(lst_widthmult, lst_inputres)

    return supernet_choices, net_search_options

def parametric_supernet_blk_choices(global_settings: Settings, search_options=None):

    # permutations to test
    if search_options is None:
        settings_per_dataset    = global_settings.NAS_SETTINGS_PER_DATASET[global_settings.NAS_SETTINGS_GENERAL['DATASET']]
        k_expfactors            = settings_per_dataset['EXP_FACTORS']
        k_kernelsizes           = settings_per_dataset['KERNEL_SIZES']
        k_num_layers_explicit   = settings_per_dataset['MOBILENET_NUM_LAYERS_EXPLICIT']
        k_support_skip          = settings_per_dataset['SUPPORT_SKIP']
    else:
        k_expfactors            = search_options['EXP_FACTORS']
        k_kernelsizes           = search_options['KERNEL_SIZES']
        k_num_layers_explicit   = search_options['MOBILENET_NUM_LAYERS_EXPLICIT']
        k_support_skip          = search_options['SUPPORT_SKIP']

    if global_settings.TINAS['STAGE1_SETTINGS']['DROPPING_ENABLED']:
        block_level_dropped_choices = global_settings.TINAS['STAGE1_SETTINGS']['DROPPING_BLOCK_LEVEL']
        k_expfactors            = drop_choices(k_expfactors,            block_level_dropped_choices['EXP_FACTORS'])
        k_kernelsizes           = drop_choices(k_kernelsizes,           block_level_dropped_choices['KERNEL_SIZES'])
        k_num_layers_explicit   = drop_choices(k_num_layers_explicit,   block_level_dropped_choices['MOBILENET_NUM_LAYERS_EXPLICIT'])
        k_support_skip          = drop_choices(k_support_skip,          block_level_dropped_choices['SUPPORT_SKIP'])

    return iter_blk_choices(k_expfactors, k_kernelsizes, k_num_layers_explicit, k_support_skip)

def blkchoices_ixs_to_blkchoices(blk_choices_list_ixs, blk_choices_list):    
    cpb = []
    for cix in blk_choices_list_ixs:
        choice_per_blk = blk_choices_list[cix]
        cpb.append(choice_per_blk)    
    return cpb

def blkchoices_to_blkchoices_ixs(blk_choices_list, choices_per_blk):    
    blkchoices_ixs = []
    for each_blk_choice in choices_per_blk:
        ix = blk_choices_list.index(each_blk_choice)
        blkchoices_ixs.append(ix)        
    return blkchoices_ixs


# round number to nearest even number
def round_to_nearest_even_num(num):
    return round(num / 2) * 2

# round number UP to nearest even number
def round_up_to_nearest_even_num(num):
    return math.ceil(num / 2) * 2


def split_list_chunks(lst, num_chunks):
    chunk_size = int(len(lst) / num_chunks)
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def get_sampled_subnet_configs(global_settings, dataset, supernet_blk_choices, n_rnd_samples=None):    
    if dataset in ['CIFAR10', 'HAR', 'KWS']:
        num_blocks  = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']    
        choices_per_block = [list(x) for x in itertools.product(supernet_blk_choices, repeat=num_blocks)]        
        
        if n_rnd_samples != None:
            result = random.sample(choices_per_block, n_rnd_samples)
        else:
            result = choices_per_block
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown dataset, " + dataset)
        
    return result
        


    
    
    


def get_subnet(global_settings, dataset, blk_choices, subnet_choice_per_blk_ixs, sidx,                
               width_multiplier=1.0, input_resolution=32,
               subnet_name=None,
               ):    
    
    if dataset in ('CIFAR10', 'HAR', 'KWS'):
    
        num_blocks  = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_BLOCKS']
        num_classes = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['NUM_CLASSES']
        stem_c_out  = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['STEM_C_OUT']
        input_channels = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['INPUT_CHANNELS']            
        stride_first = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['STRIDE_FIRST']
        downsample_blocks = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['DOWNSAMPLE_BLOCKS']      
        block_out_channels = [round_to_nearest_even_num(width_multiplier*c) for c in global_settings.NAS_SETTINGS_PER_DATASET[dataset]['OUT_CH_PER_BLK']] 

        sb_blk_choice_key = "<" + ','.join([str(c) for c in subnet_choice_per_blk_ixs]) + ">"
        subnet_name = sb_blk_choice_key if (subnet_name == None) else subnet_name

        blk_choices_list = blk_choices    
        subnet_choice_per_blk = blkchoices_ixs_to_blkchoices(subnet_choice_per_blk_ixs, blk_choices_list)

        use_1d_conv = global_settings.NAS_SETTINGS_PER_DATASET[global_settings.NAS_SETTINGS_GENERAL['DATASET']]['USE_1D_CONV']
                
        net_choices = [width_multiplier, input_resolution]
        
        #print("get_subnet:: ", global_settings.NAS_SETTINGS_GENERAL['DATASET'])
        #sys.exit()

        #print("Creating Subnet - ", subnet_name)
        subnet = MNASSubNet(subnet_name, num_blocks, num_classes, stem_c_out, input_channels, stride_first, downsample_blocks, block_out_channels,
                            sidx, subnet_choice_per_blk, net_choices=net_choices, search_options=None, use_1d_conv=use_1d_conv,
                            )        
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown dataset, " + dataset)
            
    return subnet
    

def get_subnet_from_config(global_settings: Settings, dataset, net_config, supernet_config, subnet_idx=0, subnet_name=None):
    width_multiplier, input_resolution = supernet_config

    supernet_blk_choices = parametric_supernet_blk_choices(global_settings=global_settings)
    subnet_choice_per_blk_ixs = blkchoices_to_blkchoices_ixs(supernet_blk_choices, net_config)
    subnet_pyt = get_subnet(global_settings, dataset, supernet_blk_choices, subnet_choice_per_blk_ixs, subnet_idx, 
                            width_multiplier=width_multiplier, input_resolution=input_resolution,
                            subnet_name=subnet_name)

    net_input = get_dummy_net_input_tensor(global_settings, input_resolution)

    subnet_dims = get_network_dimension(subnet_pyt, input_tensor = net_input)
    subnet_obj = get_network_obj(subnet_dims)

    return subnet_obj, subnet_pyt

    
def get_supernet(global_settings, dataset, 
                 load_state=False, supernet_train_chkpnt_fname=None,
                 width_multiplier=1.0, input_resolution=32, blk_choices=None):    
    
    if dataset in ('CIFAR10', 'HAR', 'KWS'):
        block_out_channels =  [round_to_nearest_even_num(width_multiplier * c) for c in global_settings.NAS_SETTINGS_PER_DATASET[dataset]['OUT_CH_PER_BLK']]
        model = MNASSuperNet(global_settings, dataset, block_out_channels, blk_choices=blk_choices, net_choices=(width_multiplier, input_resolution))
        
        if (load_state):
            # load model from checkpoint        
            #ckpt_fname = global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR'] + supernet_train_chkpnt_fname            
            model.load_state_dict(torch.load(supernet_train_chkpnt_fname, map_location='cpu'))
    
    else:
        sys.exit(inspect.currentframe().f_code.co_name+"::Error - unknown dataset, " + dataset)
    
    return model


#################### Datasets ##########################

# From https://github.com/healthDataScience/deep-learning-HAR/blob/master/utils/utilities.py
def _har_read_data(data_path, split = "train"):
    """ Read data """

    # Fixed params
    n_class = 6
    n_steps = 128

    # Paths
    path_ = os.path.join(data_path, split)
    path_signals = os.path.join(path_, "Inertial Signals")

    # Read labels and one-hot encode
    label_path = os.path.join(path_, "y_" + split + ".txt")
    labels = pd.read_csv(label_path, header = None)

    # Read time-series data
    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = len(channel_files)
    posix = len(split) + 5

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))
    i_ch = 0
    for fil_ch in channel_files:
        channel_name = fil_ch[:-posix]
        dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
        X[:,:,i_ch] = dat_.to_numpy()

        # Record names
        list_of_channels.append(channel_name)

        # iterate
        i_ch += 1

    # Return 
    return np.transpose(X, axes=[0, 2, 1]), labels[0].values, list_of_channels

def _har_standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
    X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

    return X_train, X_test

def get_dataset(global_settings: Settings, dataset = None, input_resolution=None, num_workers=8, trainset_batchsize=None):
    if (dataset == None):
        dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']

    if trainset_batchsize is None:
        trainset_batchsize = global_settings.NAS_SETTINGS_PER_DATASET[dataset]['TRAIN_SUBNET_BATCHSIZE']

    train_transform, valid_transform = utils.data_transforms(dataset, input_resolution=input_resolution)
    
    if dataset == 'CIFAR10':
        
        # train_transform = transforms.Compose([
        #                                     transforms.RandomCrop(32, padding = 4),
        #                                     transforms.RandomHorizontalFlip(),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        #                                     ])        
        
        # valid_transform = transforms.Compose([
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        #                                     ])
                
        trainset = torchvision.datasets.CIFAR10(root=global_settings.NAS_SETTINGS_PER_DATASET['CIFAR10']['TRAIN_DATADIR'], 
                                                train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                   batch_size=trainset_batchsize,
                                                   shuffle=True, pin_memory=True, num_workers=num_workers)
        valset = torchvision.datasets.CIFAR10(root=global_settings.NAS_SETTINGS_PER_DATASET['CIFAR10']['TRAIN_DATADIR'],
                                              train=False, download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=global_settings.NAS_SETTINGS_PER_DATASET['CIFAR10']['VAL_BATCHSIZE'],
                                                 shuffle=False, pin_memory=True, num_workers=num_workers)        
        
        
    elif dataset == 'HAR':

        # Inspired by https://blog.csdn.net/bucan804228552/article/details/120143943
        archive_dir = join(dirname(dirname(dirname(dirname(dirname(realpath(__file__)))))), 'UCI HAR Dataset')

        X, labels_x, _ = _har_read_data(archive_dir, split='train')
        assert X[0].shape == (9, 128), str(X[0].shape)
        _, X = _har_standardize(np.random.rand(*X.shape), X)
        X = torch.from_numpy(X.astype(np.float32))

        # Need any other transforms?
        # RandomHorizontalFlip, RandomRotation, RandomAffine, RandomCrop, RandomResizedCrop and ColorJitter may not make sense for HAR dataset
        # Normalization is already done in _har_standardize function
        X = torchvision.transforms.functional.resize(X, (9, input_resolution))

        trainset = TensorDataset(X, torch.from_numpy(labels_x-1))
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=trainset_batchsize,
                                                   shuffle=True, pin_memory=True, num_workers=num_workers)

        Y, labels_y, _ = _har_read_data(archive_dir, split='test')
        _, Y = _har_standardize(np.random.rand(*Y.shape), Y)
        Y = torch.from_numpy(Y.astype(np.float32))

        Y = torchvision.transforms.functional.resize(Y, (9, input_resolution))

        valset = TensorDataset(Y, torch.from_numpy(labels_y-1))
        val_loader = torch.utils.data.DataLoader(valset, batch_size=global_settings.NAS_SETTINGS_PER_DATASET['HAR']['VAL_BATCHSIZE'],
                                                 shuffle=False, pin_memory=True, num_workers=num_workers)

    
    elif dataset == 'KWS':
        raise ValueError("common_utils:get_dataset:: Error - KWS not implemented yet")  # TODO_KWS
    
        
        # # Set data's parameters
        # kwargs = {'num_workers': 0, 'pin_memory': True} if self.device == 'cuda' else {}
        
        # # Load train data
        # train_loader = torch.utils.data.DataLoader(SpeechCommandsDataset(split='train', window_stride_ms=20), 
        #                                            batch_size=trainset_batchsize, 
        #                                            shuffle=True, **kwargs)
        
        # # Load test data
        # val_loader = torch.utils.data.DataLoader(SpeechCommandsDataset(split='test', 
        #                                                                window_stride_ms=20, background_frequency=0, background_volume_range=0), 
        #                                                                 batch_size=global_settings.NAS_SETTINGS_PER_DATASET['KWS']['VAL_BATCHSIZE'],
        #                                                                 shuffle=True, **kwargs)
        
    
    else:
        raise ValueError("common_utils:get_dataset:: Error - unknown dataset : " + str(dataset))
    
    return train_loader, val_loader

def get_dummy_net_input_tensor_size(global_settings: Settings, input_resolution):
    dataset = global_settings.NAS_SETTINGS_GENERAL['DATASET']
    settings_per_dataset = global_settings.NAS_SETTINGS_PER_DATASET[dataset]
    input_channels = settings_per_dataset['INPUT_CHANNELS']

    if settings_per_dataset['USE_1D_CONV']:
        net_input_size = (1, input_channels, input_resolution)
    else:
        net_input_size = (1, input_channels, input_resolution, input_resolution)

    return net_input_size

def get_dummy_net_input_tensor(global_settings: Settings, input_resolution):
    return torch.zeros(get_dummy_net_input_tensor_size(global_settings, input_resolution))

def get_dummy_net_input_tensor_exlicit(input_channels, input_resolution, use_1d_conv=False):
    if use_1d_conv:
        net_input_size = (1, input_channels, input_resolution)
    else:
        net_input_size = (1, input_channels, input_resolution, input_resolution)

    return torch.zeros(net_input_size)


# if __name__ == '__main__':
#     test_net = nn.Sequential()
#     test_net.add_module('conv1', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, groups=1, bias=False))
#     test_net.add_module('conv2', nn.Conv2d(24, 22, kernel_size=3, stride=1, padding=0, groups=1, bias=False))
    
#     get_network_dimension(test_net[0])
