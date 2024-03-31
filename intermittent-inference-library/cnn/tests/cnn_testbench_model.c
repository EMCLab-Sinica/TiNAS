#include <stdint.h>

#include "../cnn_types.h"
#include "../../data/model-contpow.h"
#include "../../utils/sleep_timer.h"
#include "../../utils/util_macros.h"

#define CNN_TESTBENCH_MODEL_LATENCY 0
#define CNN_TESTBENCH_MODEL_ENERGY 1

#define CNN_TESTBENCH_MODEL_TYPE CNN_TESTBENCH_MODEL_ENERGY

void CNN_run(){
    eraseFRAM();
    initializeData();

#if CNN_TESTBENCH_MODEL_TYPE == CNN_TESTBENCH_MODEL_LATENCY
    // test CNN - run iteratively
    for(uint16_t lix=0; lix < network.numLayers; lix++){
        CNNLayer_t *layer = &network.Layers[lix];
        layer->fun(layer->lix, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm,&layer->parE,&layer->parP,layer->idxBuf);
    }
    while (1);
#else
    _lpm_sleep(20);

    // test CNN - run iteratively
    for(uint16_t lix=0; lix < network.numLayers; lix++){
        CNNLayer_t *layer = &network.Layers[lix];
        for (uint16_t idx=0; idx < layer->repeat; idx++) {
            layer->fun(layer->lix, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm,&layer->parE,&layer->parP,layer->idxBuf);
        }
        _lpm_sleep(3);
    }
    while (1) {
        _lpm_sleep(3);
    }
#endif
}
