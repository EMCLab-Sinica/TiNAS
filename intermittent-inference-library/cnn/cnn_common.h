/*
 * cnn_common.h
 * Common defines, constants and header includes related to the CNN module
 */

#ifndef CNN_COMMON_H_
#define CNN_COMMON_H_

#define Q15_SIZE    2           // how many bytes is a q15 ?

#define LEA_STACK       200
#define LEA_MEM_SIZE    (2048-LEA_STACK)


/* Benchmarking related */

// enable/disable benchmarking
#define CNN_BENCHMARKING_LEA_CONV1D    0
#define CNN_BENCHMARKING_LEA_MATADD    0
#define CNN_BENCHMARKING_DMA           0

#define CNN_BENCHMARKING_IM_LAY_CONV_TASKS      0
#define CNN_BENCHMARKING_IM_LAY_FC_TASKS        0
#define CNN_BENCHMARKING_IM_LAY_MAXPOOL_TASKS   0



#define CNN_TASKCOMPLETE_SYSSHUTDOWN        1



/* testbench selection */
#define TB_CNN                  0
#define TB_SINGLELAYER_CONV     1
#define TB_SINGLELAYER_FC       2
#define TB_SINGLELAYER_POOL     3

#define TB_COMBOLAYER_CONV                 100


#define TB_CNN_MICROBENCH                  150

#define CNN_TB_TYPE   TB_CNN







//#pragma PERSISTENT(DummyOFMBuff)
//_q15 DummyOFMBuff[1][1][1] = {{{0}}};

#endif /* CNN_COMMON_H_ */
