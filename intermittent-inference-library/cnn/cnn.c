/*
 * cnn.c
 * Main CNN handler (entry)
 */

#include "driverlib.h"

#include "../utils/extfram.h"
#include "../utils/myuart.h"
#include "../utils/util_macros.h"

#include "cnn.h"
#include "cnn_common.h"
#include "cnn_utils.h"


#include "tests/cnn_testbench_conv.h"
//#include "tests/cnn_testbench_pool.h"
#include "tests/cnn_testbench_microbench.h"

#include "cnn_buffers.h"

#define TEST_CONTPOW_MODEL 0
#define TEST_MULTIPLE_INFERENCES 1

#if TEST_CONTPOW_MODEL
#include "../data/model-contpow.h"
#else
#include "../data/model-intpow.h"
#endif



/* for conv combo testing */








#pragma PERSISTENT(NVM_init)
uint8_t NVM_init=0;

// current execution CNN layer
#pragma PERSISTENT(fpLayer)
uint8_t fpLayer = 0;

#pragma PERSISTENT(printFlag)
uint8_t printFlag = 0;
// globals
#pragma PERSISTENT(CurrLayerBench)
CNNLayerBenchmark_t CurrLayerBench = {false, 0, 0, 0};

#pragma PERSISTENT(inferenceDone)
uint8_t inferenceDone = 0;














/*******************************************************************
 * CNN runner
 *******************************************************************/



void CNN_run(){

    if(!NVM_init){
        eraseFRAM();
        NVM_init=1;
        initializeData();
        while(1);
     }


#if     CNN_TB_TYPE == TB_CNN

    if (inferenceDone) {
        _STOP();
    }

    uint16_t lix=0;
    uint8_t  fpL=0;
    CNNLayer_t *layer;
    SPI_ADDR A;
    A.L = LOC_LAYER_ID;
    SPI_READ(&A,(uint8_t*)&fpL,sizeof(uint8_t));

#if ENABLE_FRAM_COUNTERS
    counters.pulseCount++;
#endif
    // test CNN - run iteratively
    for(;;) { // each channel in ifm
    	if( (lix==0)  && (printFlag==0)){_DBGUART("\r\nL-ST\r\n");printFlag=1;_SHUTDOWN();_STOP();}

        for(lix=fpL; lix < network.numLayers; lix++){ // each channel in ifm
            CNN_Benchmark_Set_LID(lix);

            layer = &network.Layers[lix];
            layer->fun(layer->lix, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm,&layer->parE,&layer->parP,layer->idxBuf);

            fpL++;
            A.L = LOC_LAYER_ID;
            SPI_WRITE(&A,(uint8_t*)&fpL,sizeof(uint8_t));
#if ENABLE_FRAM_COUNTERS
            _DBGUART("L : %d, P: %d\r\n",counters.fpCount,counters.pulseCount);
#endif
        }

#if ENABLE_FRAM_COUNTERS
        counters.fpCount++;
        _DBGUART("L-DN Cnt: %d, P: %d\r\n",counters.fpCount,counters.pulseCount);
        counters.pulseCount=0;
#endif
        fpL=0;printFlag=0;
        A.L = LOC_LAYER_ID;
        SPI_WRITE(&A,(uint8_t*)&fpL,sizeof(uint8_t));
#ifdef __MSP430__
        GPIO_setOutputHighOnPin( GPIO_PORT_P4, GPIO_PIN1 ); // indicate inference done
#endif

#if TEST_MULTIPLE_INFERENCES
        _DBGUART(".\r\n");
#else
        inferenceDone = 1;
        _SHUTDOWN_AFTER_TILE();
#endif

    }




#elif (CNN_TB_TYPE == TB_COMBOLAYER_CONV)


    //test_LayerConv_Combo();


#elif (CNN_TB_TYPE == TB_CNN_MICROBENCH)

    /* LEA */
    

    /* DMA */
    //test_iterative_dma_fram_to_sram();
    //test_iterative_dma_sram_to_fram();
    //test_iterative_dma_fram_to_fram();
    //test_iterative_dma_sram_to_sram();

    /* MATH */
    //test_iterative_add();
    //test_iterative_multiply();
    //test_iterative_division();
    //test_iterative_modulo();

    /* CPU DATA MOVE */
    //test_iterative_cpu_fram_to_sram();
    //test_iterative_cpu_sram_to_fram();
    //test_iterative_cpu_fram_to_fram();
    //test_iterative_cpu_sram_to_sram();
    //test_iterative_cpu_fram_to_reg();

#else


    // test layer - run iteratively
    CNNLayer_t *layer = &network.Layers[0];

    for (;;){
        CNN_Benchmark_Set_LID(0);
    #if   (CNN_TB_TYPE == TB_SINGLELAYER_CONV)
        test_LayerConv(layer);
    #elif (CNN_TB_TYPE == TB_SINGLELAYER_FC)
        test_LayerFC(layer);
    #elif (CNN_TB_TYPE == TB_SINGLELAYER_POOL)
        test_LayerPool(layer);
    #else
        _DBGUART("Error - no valid TB type specified \r\n");
    #endif

        counters.fpCount++;

        // manually trigger EHM power switch close (try to start the new layer with a fresh power pulse)
        __delay_cycles(100);
        P1OUT = 0x1;
    }

    CNN_ClearFootprints_LayerConv_Tiled_Std(0);
	counters.fpCount = 0;
    _STOP();

#endif


}


/*******************************************************************
 * Print Info
 *******************************************************************/
void CNN_printLayerInfo(uint16_t lid, uint16_t ifm_t, uint16_t k_t, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm){
    _DBGUART("L[%d] - IFM:[%d, %d, %d, %d], WEIGHTS:[%d, %d, %d, %d], OFM:[%d, %d, %d] \r\n",
             lid,
             ifm_t, ifm->ch, ifm->h, ifm->w,
             k_t, weights->n, weights->h, weights->w,
             ofm->ch, ofm->h, ofm->w
    );
}


/* related to the model */
uint16_t CNN_GetModel_numLayers(){
    return network.numLayers;
}

/*******************************************************************
 * Benchmark related
 *******************************************************************/
void CNN_Benchmark_Set_TID(uint32_t tid){
    CurrLayerBench.tid = tid;
}
void CNN_Benchmark_Set_LID(uint16_t lid){
    CurrLayerBench.lid = lid;
}
void CNN_Benchmark_Set_CycleCount(uint32_t cc){
    CurrLayerBench.cycleCount = cc;
}
void CNN_Benchmark_Set_Active(bool a){
    CurrLayerBench.active = a;
}
void CNN_Benchmark_ClearData(){
    CurrLayerBench.active = false;
    CurrLayerBench.tid = 0;
    CurrLayerBench.cycleCount = 0;
}
CNNLayerBenchmark_t CNN_Benchmark_GetData(){
    return CurrLayerBench;
}
uint32_t CNN_Benchmark_Get_TID(){
    return CurrLayerBench.tid;
}
uint16_t CNN_Benchmark_Get_LID(){
    return CurrLayerBench.lid;
}








