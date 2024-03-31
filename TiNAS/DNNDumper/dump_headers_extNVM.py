import sys
from datetime import datetime
from os.path import basename, dirname, realpath


# local imports
sys.path.append('.')
from cnn_types import CNNModel

sys.path.append(dirname(dirname(realpath(__file__))))
from NASBase.model.common_types import Mat


#################################################################
#           HELPERS
#################################################################
def _get_text_mat(data_loc, n, ch, h, w, dtype='int'):
    if dtype == 'str':
        m = Mat( data = data_loc, n = str(n), ch = str(ch), h = str(h), w = str(w) )
    else:
        m = Mat( data = data_loc, n = n, ch = ch, h = h, w = w )
    return m


#################################################################
#           COMMON
#################################################################
def _write_file_top_datafile(fc, fname):
    fc.write('#ifndef %s_H_\n' % fname.upper())
    fc.write('#define %s_H_\n\n' % fname.upper())
    fc.write('#include "DSPLib.h"\n\n')       


#################################################################
#  OUTPUT MODEL ARCHITECTURE
#################################################################
def _get_layer_text(layer):

    s = "{\n"
    s += "\t// %s (%s)\n" % (layer.name, layer.alias)
    
    s += "\t.lix = %d,\n" % layer.lid
    s += "\t.fun = %s,\n" % layer.func
    s += "\t.repeat = %d,\n" % layer.repeat

    # -- weights --
    s += "\t.weights = (Mat_t){\n"
    s += "\t\t.data = %s,\n" % (layer.weights.data)
    s += "\t\t.n = %d,\n" % (layer.weights.n)
    s += "\t\t.ch = %d,\n" % (layer.weights.ch)
    s += "\t\t.h = %d,\n" % (layer.weights.h)
    s += "\t\t.w = %d\n" % (layer.weights.w)
    s += "\t},\n"

    # -- bias --
    s += "\t.bias = (Mat_t){\n"
    s += "\t\t.data = %s,\n" % (layer.bias.data)
    s += "\t\t.n = %d,\n" % (layer.bias.n)
    s += "\t\t.ch = %d,\n" % (layer.bias.ch)
    s += "\t\t.h = %d,\n" % (layer.bias.h)
    s += "\t\t.w = %d\n" % (layer.bias.w)
    s += "\t},\n"

    # -- ifm --
    s += "\t.ifm = (Mat_t){\n"
    s += "\t\t.data = %s,\n" % (layer.ifm.data)
    s += "\t\t.n = %d,\n" % (layer.ifm.n)
    # A special case for depthwise conv, where NAS gives N=Tn=1, but it's not the actual number of channels for an input tile
    if layer.alias.endswith('_dw'):
        s += "\t\t.ch = %d,\n" % (layer.ofm.ch)
    else:
        s += "\t\t.ch = %d,\n" % (layer.ifm.ch)
    s += "\t\t.h = %d,\n" % (layer.ifm.h)
    s += "\t\t.w = %d\n" % (layer.ifm.w)
    s += "\t},\n"

    # -- ofm --
    s += "\t.ofm = (Mat_t){\n"
    s += "\t\t.data = %s,\n" % (layer.ofm.data)
    s += "\t\t.n = %d,\n" % (layer.ofm.n)
    s += "\t\t.ch = %d,\n" % (layer.ofm.ch)
    s += "\t\t.h = %d,\n" % (layer.ofm.h)
    s += "\t\t.w = %d\n" % (layer.ofm.w)
    s += "\t},\n"


    # -- execution parameters --
    s += "\t.parE = (ExeParams_t){\n"    
    if layer.func.endswith('_LayerConv_Tiled_Depthwise') or layer.func.endswith('_BatchNormalization') or layer.func.endswith('_ReLU') or layer.func.endswith('_Add'):
        s += "\t\t.Tn = %d,\n" % (layer.Tm)
    else:
        s += "\t\t.Tn = %d,\n" % (layer.Tn)
    s += "\t\t.Tm = %d,\n" % (layer.Tm)
    s += "\t\t.Tr = %d,\n" % (layer.Tr)
    s += "\t\t.Tc = %d,\n" % (layer.Tc)
    s += "\t\t.str = %d,\n" % (layer.str)
    s += "\t\t.pad = %d,\n" % (layer.pad)
    s += "\t\t.lpOdr = %s\n" % (layer.lpOdr)
    s += "\t},\n"


    # -- preservation parameters --
    s += "\t.parP = (PreParams_t){\n"
    s += "\t\t.preSz = %d,\n" % (layer.preSz)    
    s += "\t},\n"


    # -- double buffer index --
    s += "\t.idxBuf = 0\n"

    s += "}"

    return s


def _write_file_top_modelfile(fc, model_name, output_path):
    dtstr = datetime.now().strftime("%m/%d/%y,%H:%M:%S")

    fc.write("/* \n" +
              "* %s \n" % basename(output_path) +
              "* (Auto-generated)\n" +               
              "* Created on: %s \n" % dtstr +
              "* Label : %s \n" % model_name +
              "*/\n\n")  
    
    fc.write('#ifndef %s_H_\n' % model_name.upper())
    fc.write('#define %s_H_\n\n' % model_name.upper())
        
    fc.write('#include "../cnn/cnn_types.h"\n')           
    fc.write('#include "../cnn/cnn_add.h"\n')
    fc.write('#include "../cnn/cnn_batchnorm.h"\n')
    fc.write('#include "../cnn/cnn_conv_tiled_dw.h"\n')   
    fc.write('#include "../cnn/cnn_conv_tiled_std.h"\n')   
    fc.write('#include "../cnn/cnn_pool.h"\n')       
    fc.write('#include "../cnn/cnn_relu.h"\n')
    fc.write('#include "../cnn/cnn_utils.h"\n')
    fc.write('\n\n')


def write_model_header(model_name, model: CNNModel, output_path):
    
    # Create and write file
    with open(output_path, 'w') as fc:

        _write_file_top_modelfile(fc, model_name, output_path)
                
        fc.write("\n")    

        # -- write out each layer in the model --
        fc.write("#pragma PERSISTENT(%s)\n" % model.name)
        fc.write("CNNLayer_t %s[%d] = {\n" % (model.name, model.num_layers))
        for lix, each_layer in enumerate(model.layers):
            s = _get_layer_text(each_layer)
            fc.write(s)
            if (lix < (model.num_layers-1)):
                fc.write(",\n")
        
        fc.write("\n};\n\n")
          
        # -- write out the network model instance --
        fc.write("#pragma PERSISTENT(network)\n" + 
                 "CNNModel_t network={\n" +             
                 "\t.Layers       = %s,\n" % model.name + 
                 "\t.numLayers = %d,\n" % model.num_layers + 
                 "\t.name = \"%s\"\n" % model.name + 
                 "};\n\n")

        # Generate some model parameters
        # Some layers does not work with zero weights
        fc.write("static void initializeData() {\n")

        for lix, each_layer in enumerate(model.layers):
            if each_layer.func.endswith('_BatchNormalization'):
                weights_data = [1] * (each_layer.weights.n * each_layer.weights.ch)
                weights_var_name = 'LAYER_%d_WEIGHTS' % (lix,)
                fc.write("\tstatic const _q15 %s[] = { %s };\n" % (weights_var_name, ",".join(map(str, weights_data))))
                fc.write("\tmemcpy_dma_ext(%d, %s, sizeof(%s), sizeof(%s), MEMCPY_WRITE);\n\n" % (
                    each_layer.weights.data, weights_var_name, weights_var_name, weights_var_name
                ))

        fc.write("}\n")

        fc.write('#endif /* %s_H_ */\n' % model_name.upper())
        

    print ("Written out - ", output_path)











