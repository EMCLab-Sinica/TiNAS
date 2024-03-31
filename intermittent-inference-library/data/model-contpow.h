/* 
* model-contpow.h 
* (Auto-generated)
* Created on: 02/07/24,21:36:48 
* Label : MODEL_CONTPOW 
*/

#ifndef MODEL_CONTPOW_H_
#define MODEL_CONTPOW_H_

#include "../cnn/cnn_types.h"
#include "../cnn/cnn_add.h"
#include "../cnn/cnn_batchnorm.h"
#include "../cnn/cnn_conv_tiled_dw.h"
#include "../cnn/cnn_conv_tiled_std.h"
#include "../cnn/cnn_fc.h"
#include "../cnn/cnn_pool.h"
#include "../cnn/cnn_relu.h"
#include "../cnn/cnn_utils.h"



#pragma PERSISTENT(MODEL_CONTPOW)
CNNLayer_t MODEL_CONTPOW[72] = {
{
	// CONV_0 (stem.conv0)
	.lix = 0,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 1,
	.weights = (Mat_t){
		.data = 295012,
		.n = 16,
		.ch = 3,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 3,
		.h = 32,
		.w = 32
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 16,
		.h = 32,
		.w = 32
	},
	.parE = (ExeParams_t){
		.Tn = 1,
		.Tm = 8,
		.Tr = 8,
		.Tc = 8,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_1 (stem.bn0)
	.lix = 1,
	.fun = CNN_BatchNormalization,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 295876,
		.n = 4,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 16,
		.h = 32,
		.w = 32
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 16,
		.h = 32,
		.w = 32
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 4,
		.Tc = 12,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_2 (choice_blocks.0._layers.0.op.mbconv_conv0_pw)
	.lix = 2,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 4,
	.weights = (Mat_t){
		.data = 296004,
		.n = 48,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 16,
		.h = 32,
		.w = 32
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 32,
		.w = 32
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 24,
		.Tr = 1,
		.Tc = 16,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_3 (choice_blocks.0._layers.0.op.mbconv_bn0)
	.lix = 3,
	.fun = CNN_BatchNormalization,
	.repeat = 8,
	.weights = (Mat_t){
		.data = 297540,
		.n = 4,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 32,
		.w = 32
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 32,
		.w = 32
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 1,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_4 (choice_blocks.0._layers.0.op.mbconv_relu0)
	.lix = 4,
	.fun = CNN_ReLU,
	.repeat = 16,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 32,
		.w = 32
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 32,
		.w = 32
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 2,
		.Tc = 32,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_5 (choice_blocks.0._layers.0.op.mbconv_conv1_dw)
	.lix = 5,
	.fun = CNN_LayerConv_Tiled_Depthwise,
	.repeat = 2,
	.weights = (Mat_t){
		.data = 297924,
		.n = 48,
		.ch = 1,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 32,
		.w = 32
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 8,
		.Tm = 8,
		.Tr = 4,
		.Tc = 4,
		.str = 2,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_6 (choice_blocks.0._layers.0.op.mbconv_bn1)
	.lix = 6,
	.fun = CNN_BatchNormalization,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 298788,
		.n = 4,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 1,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_7 (choice_blocks.0._layers.0.op.mbconv_relu1)
	.lix = 7,
	.fun = CNN_ReLU,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 4,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_8 (choice_blocks.0._layers.0.op.mbconv_conv2_pw)
	.lix = 8,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 16,
	.weights = (Mat_t){
		.data = 299172,
		.n = 16,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 16,
		.Tr = 1,
		.Tc = 16,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_9 (choice_blocks.0._layers.0.op.mbconv_bn2)
	.lix = 9,
	.fun = CNN_BatchNormalization,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 300708,
		.n = 4,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 6,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_10 (choice_blocks.0._layers.1.op.mbconv_conv0_pw)
	.lix = 10,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 16,
	.weights = (Mat_t){
		.data = 300836,
		.n = 48,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 24,
		.Tr = 1,
		.Tc = 16,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_11 (choice_blocks.0._layers.1.op.mbconv_bn0)
	.lix = 11,
	.fun = CNN_BatchNormalization,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 302372,
		.n = 4,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 1,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_12 (choice_blocks.0._layers.1.op.mbconv_relu0)
	.lix = 12,
	.fun = CNN_ReLU,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 4,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_13 (choice_blocks.0._layers.1.op.mbconv_conv1_dw)
	.lix = 13,
	.fun = CNN_LayerConv_Tiled_Depthwise,
	.repeat = 2,
	.weights = (Mat_t){
		.data = 302756,
		.n = 48,
		.ch = 1,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 4,
		.Tm = 4,
		.Tr = 6,
		.Tc = 16,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_14 (choice_blocks.0._layers.1.op.mbconv_bn1)
	.lix = 14,
	.fun = CNN_BatchNormalization,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 303620,
		.n = 4,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 1,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_15 (choice_blocks.0._layers.1.op.mbconv_relu1)
	.lix = 15,
	.fun = CNN_ReLU,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 4,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_16 (choice_blocks.0._layers.1.op.mbconv_conv2_pw)
	.lix = 16,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 16,
	.weights = (Mat_t){
		.data = 304004,
		.n = 16,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 16,
		.Tr = 1,
		.Tc = 16,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_17 (choice_blocks.0._layers.1.op.mbconv_bn2)
	.lix = 17,
	.fun = CNN_BatchNormalization,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 305540,
		.n = 4,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 6,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// ADD_18 (choice_blocks.0._layers.1.shortcut.skip_aggr)
	.lix = 18,
	.fun = CNN_Add,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 1,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_19 (choice_blocks.1._layers.0.op.mbconv_conv0_pw)
	.lix = 19,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 16,
	.weights = (Mat_t){
		.data = 305668,
		.n = 48,
		.ch = 16,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 16,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 24,
		.Tr = 1,
		.Tc = 16,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_20 (choice_blocks.1._layers.0.op.mbconv_bn0)
	.lix = 20,
	.fun = CNN_BatchNormalization,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 307204,
		.n = 4,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 1,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_21 (choice_blocks.1._layers.0.op.mbconv_relu0)
	.lix = 21,
	.fun = CNN_ReLU,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 4,
		.Tc = 16,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_22 (choice_blocks.1._layers.0.op.mbconv_conv1_dw)
	.lix = 22,
	.fun = CNN_LayerConv_Tiled_Depthwise,
	.repeat = 8,
	.weights = (Mat_t){
		.data = 307588,
		.n = 48,
		.ch = 1,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 16,
		.w = 16
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 8,
		.Tm = 8,
		.Tr = 4,
		.Tc = 4,
		.str = 2,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_23 (choice_blocks.1._layers.0.op.mbconv_bn1)
	.lix = 23,
	.fun = CNN_BatchNormalization,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 308452,
		.n = 4,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 2,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_24 (choice_blocks.1._layers.0.op.mbconv_relu1)
	.lix = 24,
	.fun = CNN_ReLU,
	.repeat = 256,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 48,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 8,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_25 (choice_blocks.1._layers.0.op.mbconv_conv2_pw)
	.lix = 25,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 308836,
		.n = 24,
		.ch = 48,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 48,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 24,
		.Tr = 2,
		.Tc = 8,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_26 (choice_blocks.1._layers.0.op.mbconv_bn2)
	.lix = 26,
	.fun = CNN_BatchNormalization,
	.repeat = 256,
	.weights = (Mat_t){
		.data = 311140,
		.n = 4,
		.ch = 24,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_27 (choice_blocks.1._layers.1.op.mbconv_conv0_pw)
	.lix = 27,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 311332,
		.n = 72,
		.ch = 24,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 12,
		.Tr = 2,
		.Tc = 8,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_28 (choice_blocks.1._layers.1.op.mbconv_bn0)
	.lix = 28,
	.fun = CNN_BatchNormalization,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 314788,
		.n = 4,
		.ch = 72,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_29 (choice_blocks.1._layers.1.op.mbconv_relu0)
	.lix = 29,
	.fun = CNN_ReLU,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_30 (choice_blocks.1._layers.1.op.mbconv_conv1_dw)
	.lix = 30,
	.fun = CNN_LayerConv_Tiled_Depthwise,
	.repeat = 8,
	.weights = (Mat_t){
		.data = 315364,
		.n = 72,
		.ch = 1,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 8,
		.Tm = 8,
		.Tr = 4,
		.Tc = 8,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_31 (choice_blocks.1._layers.1.op.mbconv_bn1)
	.lix = 31,
	.fun = CNN_BatchNormalization,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 316660,
		.n = 4,
		.ch = 72,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_32 (choice_blocks.1._layers.1.op.mbconv_relu1)
	.lix = 32,
	.fun = CNN_ReLU,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_33 (choice_blocks.1._layers.1.op.mbconv_conv2_pw)
	.lix = 33,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 317236,
		.n = 24,
		.ch = 72,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 36,
		.Tm = 8,
		.Tr = 2,
		.Tc = 8,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_34 (choice_blocks.1._layers.1.op.mbconv_bn2)
	.lix = 34,
	.fun = CNN_BatchNormalization,
	.repeat = 256,
	.weights = (Mat_t){
		.data = 320692,
		.n = 4,
		.ch = 24,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// ADD_35 (choice_blocks.1._layers.1.shortcut.skip_aggr)
	.lix = 35,
	.fun = CNN_Add,
	.repeat = 256,
	.weights = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 8,
		.Tm = 8,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_36 (choice_blocks.2._layers.0.op.mbconv_conv0_pw)
	.lix = 36,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 320884,
		.n = 72,
		.ch = 24,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 24,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 12,
		.Tr = 2,
		.Tc = 8,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_37 (choice_blocks.2._layers.0.op.mbconv_bn0)
	.lix = 37,
	.fun = CNN_BatchNormalization,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 324340,
		.n = 4,
		.ch = 72,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_38 (choice_blocks.2._layers.0.op.mbconv_relu0)
	.lix = 38,
	.fun = CNN_ReLU,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 4,
		.Tc = 8,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_39 (choice_blocks.2._layers.0.op.mbconv_conv1_dw)
	.lix = 39,
	.fun = CNN_LayerConv_Tiled_Depthwise,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 324916,
		.n = 72,
		.ch = 1,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 72,
		.h = 8,
		.w = 8
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 8,
		.Tm = 8,
		.Tr = 4,
		.Tc = 4,
		.str = 2,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_40 (choice_blocks.2._layers.0.op.mbconv_bn1)
	.lix = 40,
	.fun = CNN_BatchNormalization,
	.repeat = 512,
	.weights = (Mat_t){
		.data = 326212,
		.n = 4,
		.ch = 72,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 72,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 72,
		.Tm = 72,
		.Tr = 2,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_41 (choice_blocks.2._layers.0.op.mbconv_relu1)
	.lix = 41,
	.fun = CNN_ReLU,
	.repeat = 512,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 72,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 72,
		.Tm = 72,
		.Tr = 2,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_42 (choice_blocks.2._layers.0.op.mbconv_conv2_pw)
	.lix = 42,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 326788,
		.n = 32,
		.ch = 72,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 72,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 16,
		.Tr = 4,
		.Tc = 4,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_43 (choice_blocks.2._layers.0.op.mbconv_bn2)
	.lix = 43,
	.fun = CNN_BatchNormalization,
	.repeat = 1024,
	.weights = (Mat_t){
		.data = 331396,
		.n = 4,
		.ch = 32,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 32,
		.Tm = 32,
		.Tr = 4,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_44 (choice_blocks.2._layers.1.op.mbconv_conv0_pw)
	.lix = 44,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 331652,
		.n = 96,
		.ch = 32,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 24,
		.Tr = 4,
		.Tc = 4,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_45 (choice_blocks.2._layers.1.op.mbconv_bn0)
	.lix = 45,
	.fun = CNN_BatchNormalization,
	.repeat = 256,
	.weights = (Mat_t){
		.data = 337796,
		.n = 4,
		.ch = 96,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 4,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_46 (choice_blocks.2._layers.1.op.mbconv_relu0)
	.lix = 46,
	.fun = CNN_ReLU,
	.repeat = 512,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 96,
		.Tm = 96,
		.Tr = 2,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_47 (choice_blocks.2._layers.1.op.mbconv_conv1_dw)
	.lix = 47,
	.fun = CNN_LayerConv_Tiled_Depthwise,
	.repeat = 16,
	.weights = (Mat_t){
		.data = 338564,
		.n = 96,
		.ch = 1,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 16,
		.Tr = 4,
		.Tc = 4,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_48 (choice_blocks.2._layers.1.op.mbconv_bn1)
	.lix = 48,
	.fun = CNN_BatchNormalization,
	.repeat = 256,
	.weights = (Mat_t){
		.data = 340292,
		.n = 4,
		.ch = 96,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 4,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_49 (choice_blocks.2._layers.1.op.mbconv_relu1)
	.lix = 49,
	.fun = CNN_ReLU,
	.repeat = 512,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 96,
		.Tm = 96,
		.Tr = 2,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_50 (choice_blocks.2._layers.1.op.mbconv_conv2_pw)
	.lix = 50,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 341060,
		.n = 32,
		.ch = 96,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 16,
		.Tr = 4,
		.Tc = 4,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_51 (choice_blocks.2._layers.1.op.mbconv_bn2)
	.lix = 51,
	.fun = CNN_BatchNormalization,
	.repeat = 1024,
	.weights = (Mat_t){
		.data = 347204,
		.n = 4,
		.ch = 32,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 32,
		.Tm = 32,
		.Tr = 4,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// ADD_52 (choice_blocks.2._layers.1.shortcut.skip_aggr)
	.lix = 52,
	.fun = CNN_Add,
	.repeat = 1024,
	.weights = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 32,
		.Tm = 32,
		.Tr = 2,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_53 (choice_blocks.3._layers.0.op.mbconv_conv0_pw)
	.lix = 53,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 347460,
		.n = 96,
		.ch = 32,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 32,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 16,
		.Tm = 24,
		.Tr = 4,
		.Tc = 4,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_54 (choice_blocks.3._layers.0.op.mbconv_bn0)
	.lix = 54,
	.fun = CNN_BatchNormalization,
	.repeat = 256,
	.weights = (Mat_t){
		.data = 353604,
		.n = 4,
		.ch = 96,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 48,
		.Tr = 4,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_55 (choice_blocks.3._layers.0.op.mbconv_relu0)
	.lix = 55,
	.fun = CNN_ReLU,
	.repeat = 512,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.parE = (ExeParams_t){
		.Tn = 96,
		.Tm = 96,
		.Tr = 2,
		.Tc = 4,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_56 (choice_blocks.3._layers.0.op.mbconv_conv1_dw)
	.lix = 56,
	.fun = CNN_LayerConv_Tiled_Depthwise,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 354372,
		.n = 96,
		.ch = 1,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 96,
		.h = 4,
		.w = 4
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 24,
		.Tm = 24,
		.Tr = 2,
		.Tc = 2,
		.str = 2,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_57 (choice_blocks.3._layers.0.op.mbconv_bn1)
	.lix = 57,
	.fun = CNN_BatchNormalization,
	.repeat = 1024,
	.weights = (Mat_t){
		.data = 356100,
		.n = 4,
		.ch = 96,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 96,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 96,
		.Tm = 96,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_58 (choice_blocks.3._layers.0.op.mbconv_relu1)
	.lix = 58,
	.fun = CNN_ReLU,
	.repeat = 2048,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 96,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 96,
		.Tm = 96,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_59 (choice_blocks.3._layers.0.op.mbconv_conv2_pw)
	.lix = 59,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 128,
	.weights = (Mat_t){
		.data = 356868,
		.n = 64,
		.ch = 96,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 96,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 16,
		.Tr = 2,
		.Tc = 2,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_60 (choice_blocks.3._layers.0.op.mbconv_bn2)
	.lix = 60,
	.fun = CNN_BatchNormalization,
	.repeat = 2048,
	.weights = (Mat_t){
		.data = 369156,
		.n = 4,
		.ch = 64,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 64,
		.Tm = 64,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_61 (choice_blocks.3._layers.1.op.mbconv_conv0_pw)
	.lix = 61,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 369668,
		.n = 192,
		.ch = 64,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 32,
		.Tm = 24,
		.Tr = 2,
		.Tc = 2,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_62 (choice_blocks.3._layers.1.op.mbconv_bn0)
	.lix = 62,
	.fun = CNN_BatchNormalization,
	.repeat = 512,
	.weights = (Mat_t){
		.data = 394244,
		.n = 4,
		.ch = 192,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 96,
		.Tm = 96,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_63 (choice_blocks.3._layers.1.op.mbconv_relu0)
	.lix = 63,
	.fun = CNN_ReLU,
	.repeat = 1024,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 192,
		.Tm = 192,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_64 (choice_blocks.3._layers.1.op.mbconv_conv1_dw)
	.lix = 64,
	.fun = CNN_LayerConv_Tiled_Depthwise,
	.repeat = 32,
	.weights = (Mat_t){
		.data = 395780,
		.n = 192,
		.ch = 1,
		.h = 3,
		.w = 3
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 32,
		.Tm = 32,
		.Tr = 2,
		.Tc = 2,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_65 (choice_blocks.3._layers.1.op.mbconv_bn1)
	.lix = 65,
	.fun = CNN_BatchNormalization,
	.repeat = 512,
	.weights = (Mat_t){
		.data = 399236,
		.n = 4,
		.ch = 192,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 96,
		.Tm = 96,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// RELU_66 (choice_blocks.3._layers.1.op.mbconv_relu1)
	.lix = 66,
	.fun = CNN_ReLU,
	.repeat = 1024,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 192,
		.Tm = 192,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// CONV_67 (choice_blocks.3._layers.1.op.mbconv_conv2_pw)
	.lix = 67,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 64,
	.weights = (Mat_t){
		.data = 400772,
		.n = 64,
		.ch = 192,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 192,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 48,
		.Tm = 16,
		.Tr = 2,
		.Tc = 2,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// BN_68 (choice_blocks.3._layers.1.op.mbconv_bn2)
	.lix = 68,
	.fun = CNN_BatchNormalization,
	.repeat = 2048,
	.weights = (Mat_t){
		.data = 425348,
		.n = 4,
		.ch = 64,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 64,
		.Tm = 64,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = WEIGHT_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// ADD_69 (choice_blocks.3._layers.1.shortcut.skip_aggr)
	.lix = 69,
	.fun = CNN_Add,
	.repeat = 2048,
	.weights = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 2*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.parE = (ExeParams_t){
		.Tn = 64,
		.Tm = 64,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// GAVGPOOL_70 (global_pooling)
	.lix = 70,
	.fun = CNN_GlobalAveragePool,
	.repeat = 2048,
	.weights = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 64,
		.h = 2,
		.w = 2
	},
	.ofm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 64,
		.h = 1,
		.w = 1
	},
	.parE = (ExeParams_t){
		.Tn = 1,
		.Tm = 64,
		.Tr = 2,
		.Tc = 2,
		.str = 0,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
},
{
	// FC_END (classifier)
	.lix = 71,
	.fun = CNN_LayerConv_Tiled_Std,
	.repeat = 4096,
	.weights = (Mat_t){
		.data = 425860,
		.n = 10,
		.ch = 64,
		.h = 1,
		.w = 1
	},
	.bias = (Mat_t){
		.data = 0,
		.n = 0,
		.ch = 0,
		.h = 0,
		.w = 0
	},
	.ifm = (Mat_t){
		.data = 100 + 1*98304,
		.n = 1,
		.ch = 64,
		.h = 1,
		.w = 1
	},
	.ofm = (Mat_t){
		.data = 100 + 0*98304,
		.n = 1,
		.ch = 10,
		.h = 1,
		.w = 1
	},
	.parE = (ExeParams_t){
		.Tn = 64,
		.Tm = 10,
		.Tr = 1,
		.Tc = 1,
		.str = 1,
		.pad = 0,
		.lpOdr = OFM_ORIENTED
	},
	.parP = (PreParams_t){
		.preSz = 1,
	},
	.idxBuf = 0
}
};

#pragma PERSISTENT(network)
CNNModel_t network={
	.Layers       = MODEL_CONTPOW,
	.numLayers = 72,
	.name = "MODEL_CONTPOW"
};

static void initializeData() {
	static const _q15 LAYER_1_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(295876, LAYER_1_WEIGHTS, sizeof(LAYER_1_WEIGHTS), sizeof(LAYER_1_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_3_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(297540, LAYER_3_WEIGHTS, sizeof(LAYER_3_WEIGHTS), sizeof(LAYER_3_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_6_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(298788, LAYER_6_WEIGHTS, sizeof(LAYER_6_WEIGHTS), sizeof(LAYER_6_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_9_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(300708, LAYER_9_WEIGHTS, sizeof(LAYER_9_WEIGHTS), sizeof(LAYER_9_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_11_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(302372, LAYER_11_WEIGHTS, sizeof(LAYER_11_WEIGHTS), sizeof(LAYER_11_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_14_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(303620, LAYER_14_WEIGHTS, sizeof(LAYER_14_WEIGHTS), sizeof(LAYER_14_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_17_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(305540, LAYER_17_WEIGHTS, sizeof(LAYER_17_WEIGHTS), sizeof(LAYER_17_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_20_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(307204, LAYER_20_WEIGHTS, sizeof(LAYER_20_WEIGHTS), sizeof(LAYER_20_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_23_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(308452, LAYER_23_WEIGHTS, sizeof(LAYER_23_WEIGHTS), sizeof(LAYER_23_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_26_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(311140, LAYER_26_WEIGHTS, sizeof(LAYER_26_WEIGHTS), sizeof(LAYER_26_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_28_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(314788, LAYER_28_WEIGHTS, sizeof(LAYER_28_WEIGHTS), sizeof(LAYER_28_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_31_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(316660, LAYER_31_WEIGHTS, sizeof(LAYER_31_WEIGHTS), sizeof(LAYER_31_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_34_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(320692, LAYER_34_WEIGHTS, sizeof(LAYER_34_WEIGHTS), sizeof(LAYER_34_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_37_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(324340, LAYER_37_WEIGHTS, sizeof(LAYER_37_WEIGHTS), sizeof(LAYER_37_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_40_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(326212, LAYER_40_WEIGHTS, sizeof(LAYER_40_WEIGHTS), sizeof(LAYER_40_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_43_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(331396, LAYER_43_WEIGHTS, sizeof(LAYER_43_WEIGHTS), sizeof(LAYER_43_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_45_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(337796, LAYER_45_WEIGHTS, sizeof(LAYER_45_WEIGHTS), sizeof(LAYER_45_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_48_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(340292, LAYER_48_WEIGHTS, sizeof(LAYER_48_WEIGHTS), sizeof(LAYER_48_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_51_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(347204, LAYER_51_WEIGHTS, sizeof(LAYER_51_WEIGHTS), sizeof(LAYER_51_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_54_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(353604, LAYER_54_WEIGHTS, sizeof(LAYER_54_WEIGHTS), sizeof(LAYER_54_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_57_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(356100, LAYER_57_WEIGHTS, sizeof(LAYER_57_WEIGHTS), sizeof(LAYER_57_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_60_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(369156, LAYER_60_WEIGHTS, sizeof(LAYER_60_WEIGHTS), sizeof(LAYER_60_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_62_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(394244, LAYER_62_WEIGHTS, sizeof(LAYER_62_WEIGHTS), sizeof(LAYER_62_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_65_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(399236, LAYER_65_WEIGHTS, sizeof(LAYER_65_WEIGHTS), sizeof(LAYER_65_WEIGHTS), MEMCPY_WRITE);

	static const _q15 LAYER_68_WEIGHTS[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	memcpy_dma_ext(425348, LAYER_68_WEIGHTS, sizeof(LAYER_68_WEIGHTS), sizeof(LAYER_68_WEIGHTS), MEMCPY_WRITE);

}
#endif /* MODEL_CONTPOW_H_ */
