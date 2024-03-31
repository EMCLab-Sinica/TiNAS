/*
 * cnn_buffers.h
 *
 * Only used for internal FRAM buffers
 */

#ifndef CNN_BUFFERS_H_
#define CNN_BUFFERS_H_

#include "DSPLib.h"

#include "cnn_buffer_sizes.h"
#include "cnn_common.h"


extern _q15 *pLEAMemory;
extern _q15 *pLayerTempBuff1;
extern _q15 *pLayerTempBuff2;
extern _q15 *pOFMBuff1;
extern _q15 *pOFMBuff2;
extern _q15 *pSRAMBuff;

#endif /* CNN_BUFFERS_H_ */
