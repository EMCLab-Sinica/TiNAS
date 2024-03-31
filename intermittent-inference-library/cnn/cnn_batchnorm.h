/*
 * cnn_batchnorm.h
 */

#ifndef CNN_BATCHNORM_H_
#define CNN_BATCHNORM_H_

#include <stdint.h>

#include "cnn_types.h"

void _intra_tile_batchnorm(ExeParams_t* parE, Mat_t* weights, uint16_t ifm_t_w, uint16_t ti, _q15* pIFM_t_buff,_q15* pScale_t_buff, _q15* pB_t_buff, _q15* pMean_t_buff, _q15* pVar_t_buff, uint16_t preSz, LOOPORDER_t lpOdr);
void CNN_BatchNormalization(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);
void CNN_Intermittent_BatchNormalization(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);

#endif /* CNN_BATCHNORM_H_ */

