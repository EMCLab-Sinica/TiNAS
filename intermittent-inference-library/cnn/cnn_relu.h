/*
 * cnn_relu.h
 */

#ifndef CNN_RELU_H_
#define CNN_RELU_H_

#include <stdint.h>

#include "cnn_types.h"

void CNN_Intermittent_ReLU(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);
void CNN_ReLU(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);
void _intra_tile_relu(ExeParams_t* parE, uint16_t ifm_t_w, _q15* pIFM_t_buff);

#endif /* CNN_RELU_H_ */

