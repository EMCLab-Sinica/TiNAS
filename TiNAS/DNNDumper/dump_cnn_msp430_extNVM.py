import sys
from pprint import pprint
from os.path import dirname, realpath

# local imports
sys.path.append('.')
import dump_headers_extNVM as dump
from dump_headers_extNVM import _get_text_mat
from cnn_types import CNNModel, CNNLayer

sys.path.append(dirname(dirname(realpath(__file__))))


########################################################
#   CONSTANTS
########################################################

Q15_SIZE = 2 # bytes per data element (q15)
BASE_ADDR = 100
# BUFF1_ADDRESS = 1000
# BUFF2_ADDRESS = 2000

# convolution layer
LAYER_FUNC_NAMES = {
    'CONV_DW' : "CNN_Intermittent_LayerConv_Tiled_Depthwise",
    'CONV_PW' : "CNN_Intermittent_LayerConvPW_Tiled_Std",
    'CONV' : "CNN_Intermittent_LayerConv_Tiled_Std",
    'GAVGPOOL' : "CNN_Intermittent_GlobalAveragePool",      # global avg pooling       
    'FC' : "CNN_Intermittent_LayerConv_Tiled_Std",
    'ADD' : "CNN_Intermittent_Add",
    'BN' : "CNN_Intermittent_BatchNormalization",
    'RELU' : "CNN_Intermittent_ReLU",
}


########################################################
#   DUMP HEADER FILE
########################################################
def dump_model_header(model_name, num_layers, layer_txt_objs_lst, output_path):
    model =  CNNModel(name = model_name, layers = layer_txt_objs_lst, num_layers=len(layer_txt_objs_lst))
    dump.write_model_header(model_name, model, output_path)

NUM_BUFFERS = 3 # three buffers to support skip

def find_layer_idx_from_alias(subnet_obj, alias):
    for layer_idx, layer in enumerate(subnet_obj):
        if layer['alias'] == alias:
            return layer_idx

def find_layer_from_alias(subnet_obj, alias):
    layer_idx = find_layer_idx_from_alias(subnet_obj, alias)
    return subnet_obj[layer_idx]


########################################################
#   PRIMARY HANDLER
########################################################
# convert tensorflow model to msp430 format
def dump_cnn_handler(subnet_obj, network_exec_design, network_exec_design_contpow, power_type, model_name='MODEL', output_path='model.h'):
    layer_funcs = LAYER_FUNC_NAMES

    ##################Calc buffer size###################
    # A simple memory allocation - all feature maps have the same size
    max_ofm_sz = 0
    for layer in subnet_obj:
        ofm_ch = layer['OFM']['CH']
        ofm_h = layer['OFM']['H']
        ofm_w = layer['OFM']['W']
        sz = ofm_ch * ofm_h * ofm_w * Q15_SIZE
        max_ofm_sz = max(max_ofm_sz, sz)

    model_input = subnet_obj[0]['IFM']
    model_ifm_h = model_input['H']
    model_ifm_w = model_input['W']
    model_ifm_ch = model_input['CH']
    model_ifm_sz = model_ifm_h * model_ifm_w * model_ifm_ch * Q15_SIZE
    max_ofm_sz = max(max_ofm_sz, model_ifm_sz)

    all_buffers = set(f'{BASE_ADDR} + {buffer_idx}*{max_ofm_sz}' for buffer_idx in range(NUM_BUFFERS))
    free_buffers = sorted(all_buffers)
    residual_buffer = None


    curr_buffer = free_buffers[0]

    
    layer_mat_objs = [] # keep track of layer mat objs

    input_layer_obj = CNNLayer(-1, 'Input', 'Input', '', 
                                None, None, None, 
                                _get_text_mat(data_loc=curr_buffer, n=1, ch=model_ifm_ch, h=model_ifm_h, w=model_ifm_w))

    layer_mat_objs.append(input_layer_obj)

    mat_bias_txtobj = _get_text_mat(data_loc=0, n=0, ch=0, h=0, w=0) # we assume no bias for now

    
    ################## Residual connections ###################
    # Identify the start layer of each residual connection, so that the input feature map can be preserved
    residual_start_layers = set()
    for layer in subnet_obj:
        if layer['objtype'] == 'tensor.add':
            block_first_layer_alias = layer['alias'].replace('.shortcut.skip_aggr', '.op.mbconv_conv0_pw')
            block_first_layer = find_layer_from_alias(subnet_obj, block_first_layer_alias)
            residual_start_layers.add(block_first_layer['alias'])

    ext_nvm_data_offset = BASE_ADDR + max_ofm_sz*NUM_BUFFERS # <<-- starts after feature map buffer address sizes
    lix = 0
    for layer in subnet_obj:
        layer_name = layer['name'].upper()
        #print("------------------ " + layer_name + " ------------------")
        
        ifm_h = layer['IFM']['H']
        ifm_w = layer['IFM']['W']
        ifm_ch = layer['IFM']['CH']

        ofm_h = layer['OFM']['H']
        ofm_w = layer['OFM']['W']
        ofm_ch = layer['OFM']['CH']

        knum = layer['K']['N']
        ksize_h = layer['K']['H']
        ksize_w = layer['K']['W']
        ksize_ch = layer['K']['CH']

        ifm_buffer = curr_buffer
        curr_buffer = free_buffers[1] if (curr_buffer == free_buffers[0]) else free_buffers[0]
        mat_ifm_txtobj = _get_text_mat(data_loc=ifm_buffer, n=1, ch=ifm_ch, h=ifm_h, w=ifm_w)
        mat_ofm_txtobj = _get_text_mat(data_loc=curr_buffer , n=1, ch=ofm_ch, h=ofm_h, w=ofm_w)

        if 'CONV' in layer_name:
            if layer['alias'].endswith('_dw'):
                layer_func_name = layer_funcs['CONV_DW']
            else:
                layer_func_name = layer_funcs['CONV']  

            mat_wei_txtobj = _get_text_mat(data_loc=ext_nvm_data_offset, n=knum, ch=ksize_ch, h=ksize_h, w=ksize_w)
                        
            ext_nvm_data_offset += (knum * ksize_ch * ksize_h * ksize_w) * Q15_SIZE # set the extrnal nvm data offset for next layer params

            if layer['alias'] in residual_start_layers:
                assert residual_buffer is None
                assert len(free_buffers) == NUM_BUFFERS
                # Preserve the IFM for the residual connection
                residual_buffer = ifm_buffer
                free_buffers = sorted(all_buffers - set([residual_buffer]))
                                       
        elif ('FC' in layer_name):
            layer_func_name = layer_funcs['FC']  

            mat_wei_txtobj = _get_text_mat(data_loc=ext_nvm_data_offset, n=knum, ch=ksize_ch, h=ksize_h, w=ksize_w)
            
            ext_nvm_data_offset += (knum * ksize_ch * ksize_h * ksize_w) * Q15_SIZE # set the extrnal nvm data offset for next layer params

                           
        elif ('GAVGPOOL' in layer_name):     # global avg pooling           
            layer_func_name = layer_funcs['GAVGPOOL']        
            
            mat_wei_txtobj = _get_text_mat(data_loc=0, n=0, ch=0, h=0, w=0)
            
            ext_nvm_data_offset += 0 # no weight params
        
        elif ('BN' in layer_name):  # BatchNorm
            layer_func_name = layer_funcs['BN']

            mat_wei_txtobj = _get_text_mat(data_loc=ext_nvm_data_offset, n=4, ch=ifm_ch, h=1, w=1)  # 4 vectors: mu, sigma, beta (weight), gamma (bias)

            ext_nvm_data_offset += (4 * ifm_ch) * Q15_SIZE

        elif ('RELU' in layer_name):  # ReLu
            layer_func_name = layer_funcs['RELU']

            mat_wei_txtobj = _get_text_mat(data_loc=0, n=0, ch=0, h=0, w=0)

            ext_nvm_data_offset += 0 # no weight params

        elif ('ADD' in layer_name):  # Add
            layer_func_name = layer_funcs['ADD']

            block_first_layer_alias = layer['alias'].replace('.shortcut.skip_aggr', '.op.mbconv_conv0_pw')
            block_first_layer_idx = find_layer_idx_from_alias(subnet_obj, block_first_layer_alias)
            residual_data = layer_mat_objs[1+block_first_layer_idx].ifm  # First layer is "Input"
            assert residual_data.data == residual_buffer

            mat_wei_txtobj = _get_text_mat(data_loc=residual_buffer, n=residual_data.n, ch=residual_data.ch, h=residual_data.h, w=residual_data.w)

            ext_nvm_data_offset += 0 # no weight params

            # All feature map buffers are available again
            residual_buffer = None
            free_buffers = sorted(all_buffers)

        else:            
            pprint(layer)
            sys.exit("ERROR: layer error")

        if power_type == 'contpow':
            layer_func_name = layer_func_name.replace('CNN_Intermittent_', 'CNN_')

        params_pres={'backup_batch_size' : 1}

        repeat = 1
        if network_exec_design_contpow:
            repeat = network_exec_design_contpow[lix].get('repeat') or 1

        layer_txtobj = CNNLayer(lix, layer_name, layer["alias"], layer_func_name, 
                            mat_wei_txtobj, mat_bias_txtobj, mat_ifm_txtobj, mat_ofm_txtobj,
                            s=layer['stride'] or 0,
                            params_exec= network_exec_design[lix]['params'],
                            params_pres= params_pres,
                            repeat=repeat)
        layer_mat_objs.append(layer_txtobj)
        
        lix+=1 # increment layer index
        
    num_layers = len(layer_mat_objs)

    # -- dump model definition --    
    #pprint(model_mat_objs_lst)    
    dump_model_header(model_name, num_layers, layer_mat_objs[1:], output_path)
    
    
  
def _check_mat_type(all_layers):
    for each_layer in all_layers:
        print([each_layer["name"], type(each_layer['ifm']), type(each_layer['ofm'])])

