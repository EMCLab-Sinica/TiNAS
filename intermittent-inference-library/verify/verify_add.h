#pragma once

#include "DSPLib.h"
#include "../cnn/cnn_add.h"
#include "../cnn/cnn_types.h"
#include "../cnn/cnn_utils.h"
#include "../utils/myuart.h"

CNNLayer_t Add1[1] = {
{
    .lix = 0,
    .fun = CNN_Add,  // also verified with CNN_Intermittent_Add
    .weights = (Mat_t){
        .data = 32868,
        .n = 1,
        .ch = 4,
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
        .data = 100,
        .n = 1,
        .ch = 4,
        .h = 4,
        .w = 4
    },
    .ofm = (Mat_t){
        .data = 16484,
        .n = 1,
        .ch = 4,
        .h = 4,
        .w = 4
    },
    .parE = (ExeParams_t){
        .Tn = 2,
        .Tm = 2,
        .Tr = 2,
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
};

CNNModel_t network={
    .Layers       = Add1,
    .numLayers = 1,
    .name = "Verify_Add"
};



_q15 X[] = {
     8130, 11430,  2882,  4558, 12586, 13107,  4420,  7896,  1449,  2638,
     2468, 13431,  2163,  4624,   519, 16335,  5036, 11167,  3409, 11443,
    10388, 14994, 15233,  9298,  8029,  6506, 11847, 13684, 14687, 14322,
    12162,  3368,  7465,  6871,  8622,  9718, 10359,  9058,  3992,  1840,
     5716, 15609,  9577,  2514,  6581,   592,   543,  3960,   365,  3034,
     2272, 11898,  2766,  6118,  3968, 11486,  4815,  4998, 13360,  3339,
     8495, 15269, 12995, 10666,
};

_q15 Y[] = {
    12689,   557, 12282,  3206,  7158, 15470,  9906, 15625,  8504, 14420,
     1801, 13805, 10090,    20,  3474,  1283, 13274,  9725, 15898,  6153,
    16057,  6811, 13711,  8561,  1879,  6843,  4620,  9387,  5189,  4442,
     6130, 10134, 11411, 11342,   388, 11406, 14979,  3339,  8044,  8682,
    15320, 11195,  2022,  4194, 15420, 12334,  1873, 12068,  9822, 14056,
     7740,   333,  1068, 11255,  9421,  3336,  8945,    84,  4837,  6141,
     3067,  2877, 13052,  4201,
};

static void initializeData() {
    memcpy_dma_ext(network.Layers[0].ifm.data, X, sizeof(X), sizeof(X), MEMCPY_WRITE);
    // In DNNDumper, the second input is passed as weight
    memcpy_dma_ext(network.Layers[0].weights.data + 0*sizeof(Y), Y, sizeof(Y), sizeof(Y), MEMCPY_WRITE);
}

static void dumpResults() {
    _DBGUART("Layer 0 outputs\r\n");
    CNN_printResult(&network.Layers[0].ofm);
}
