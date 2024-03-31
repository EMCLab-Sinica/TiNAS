#######################
# Data types
#######################
# Objects - primarily used for holding text labels (not values)
# @TODO: rename to differentiate text objects and value objects

class CNNModel:
  def __init__(self, name, layers, num_layers):
    self.name = name
    self.layers = layers
    self.num_layers = num_layers

class CNNLayer:
  def __init__(self, lid, name, alias, func, weights, bias, ifm, ofm, s=1, p=0, params_exec=None, params_pres=None, repeat=1):
    self.lid = lid
    self.name = name
    self.alias = alias
    self.func = func
    self.weights = weights
    self.bias = bias
    self.ifm = ifm
    self.ofm = ofm
    self.repeat = repeat

    # execution params
    if params_exec != None:
      self.Tn = params_exec['Tn']; self.Tm = params_exec['Tm']; self.Tr = params_exec['Tr']; self.Tc = params_exec['Tc']
      self.lpOdr = self._get_lpodr_lbl(params_exec['reuse_sch'])
    else:
      self.Tn = -1; self.Tm = -1; self.Tr = -1; self.Tc = -1      
      self.lpOdr = 'None'

    self.str= s; self.pad= p

    # preservation params
    if params_pres != None:
      self.preSz = params_pres['backup_batch_size']
    else:
      self.preSz = -1
    
    # buffer index (double buffering)
    self.idxBuf = 0

  
  # lpord as specified in msp430 code
  def _get_lpodr_lbl(self,inter_lo):
    lbl = {
      'reuse_I' : "IFM_ORIENTED",
      'reuse_W' : "WEIGHT_ORIENTED",
      'reuse_O' : "OFM_ORIENTED",
    }
    return lbl[inter_lo]
