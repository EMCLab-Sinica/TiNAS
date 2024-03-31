/*
 * cnn.h
 * 
 */

#ifndef CNN_H_
#define CNN_H_

#include "DSPLib.h"

#include "cnn_common.h"
#include "cnn_types.h"

void CNN_run(void);

void CNN_printLayerInfo(uint16_t lid, uint16_t ifm_t, uint16_t k_t, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm);

/* getters/setters */

// model related
uint16_t CNN_GetModel_numLayers(void);







// benchmark related
void CNN_Benchmark_Set_TID(uint32_t tid);
void CNN_Benchmark_Set_LID(uint16_t lid);
void CNN_Benchmark_Set_CycleCount(uint32_t cc);
void CNN_Benchmark_Set_Active(bool a);
void CNN_Benchmark_ClearData(void);
CNNLayerBenchmark_t CNN_Benchmark_GetData(void);
uint32_t CNN_Benchmark_Get_TID(void);
uint16_t CNN_Benchmark_Get_LID(void);


#endif /* CNN_H_ */
