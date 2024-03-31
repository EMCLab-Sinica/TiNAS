#pragma once

#include "DSPLib.h"
#include "../cnn/cnn_pool.h"
#include "../cnn/cnn_types.h"
#include "../cnn/cnn_utils.h"
#include "../utils/myuart.h"

CNNLayer_t Pool1[1] = {
{
    .lix = 0,
    .fun = CNN_Intermittent_GlobalAveragePool, // CNN_GlobalAveragePool also verified
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
        .h = 1,
        .w = 1
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
    .Layers       = Pool1,
    .numLayers = 1,
    .name = "Verify_Pool"
};

_q15 INPUTS_DATA[] = {
     9707, 16307, 12317, 15671, 18619, 19661, 15395, 22346, -3654, -1276,
    11491, 32735, -2227,  2695,  7592, 32735,  3520, 15781, 13373, 29440,
    14223, 23435, 32735, 25150,  9505,  6458, 30248, 32735, 22821, 22090,
    30878, 13290,  8376,  7189, 23799, 25990, 14165, 11564, 14537, 10234,
     4878, 24665, 25709, 11582,  6609, -5368,  7639, 14473, -5822,  -483,
    11099, 30350, -1020,  5682, 14491, 29526,  3076,  3443, 32735, 13232,
    10437, 23986, 32543, 27887
};

static void initializeData() {
    memcpy_dma_ext(network.Layers[0].ifm.data, INPUTS_DATA, sizeof(INPUTS_DATA), sizeof(INPUTS_DATA), MEMCPY_WRITE);
}

static void dumpResults() {
    _DBGUART("Layer 0 outputs\r\n");
    CNN_printResult(&network.Layers[0].ofm);
}

