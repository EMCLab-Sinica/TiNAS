/*
 * cnn_add.h
 */

#ifndef CNN_ADD_H_
#define CNN_ADD_H_

#include <stdint.h>

#include "cnn_types.h"

void _intra_tile_add(ExeParams_t* parE, Mat_t* weights, uint16_t ifm_t_w, uint16_t ti, _q15* pIFM_t_buff,_q15* pWeight_t_buff, uint16_t preSz, LOOPORDER_t lpOdr);
void CNN_Add(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);
void CNN_Intermittent_Add(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);

#endif /* CNN_ADD_H_ */
