#include "cnn_buffers.h"

DSPLIB_DATA(LEA_MEMORY,4)
_q15 LEA_MEMORY[LEA_MEM_SIZE];
_q15 *pLEAMemory = LEA_MEMORY;


// temporary buffer used during LEA based convolution
#pragma location = LOC_TEMPBUFF1
#pragma PERSISTENT(LAYER_TEMP_BUFF1)
_q15 LAYER_TEMP_BUFF1[LAY_TMPBUFF1_SIZE] = {[0 ... ((LAY_TMPBUFF1_SIZE)-1)] = 0x0000};   // !! @TODO CHANGE IF MODEL IS LARGER (rationale: Ow * Iw * Kh)
_q15 *pLayerTempBuff1   = (_q15*)LAYER_TEMP_BUFF1;

#pragma location = LOC_TEMPBUFF2
#pragma PERSISTENT(LAYER_TEMP_BUFF2)
_q15 LAYER_TEMP_BUFF2[LAY_TMPBUFF2_SIZE] = {[0 ... ((LAY_TMPBUFF2_SIZE)-1)] = 0x0000};   // !! @TODO CHANGE IF MODEL IS LARGER (rationale: Ow * Ow * C)
_q15 *pLayerTempBuff2   = (_q15*)LAYER_TEMP_BUFF2;


// buffer to store output feature map results
#pragma location = LOC_OFMBUFF1
#pragma PERSISTENT(OFM_BUFF1)
_q15 OFM_BUFF1[LAY_OFMBUFF1_SIZE] = {[0 ... ((LAY_OFMBUFF1_SIZE)-1)] = 0x0000};   // !! @TODO CHANGE IF MODEL IS LARGER (rationale: Ow * Ow * F)
_q15 *pOFMBuff1   = (_q15*)OFM_BUFF1;

#pragma location = LOC_OFMBUFF2
#pragma PERSISTENT(OFM_BUFF2)
_q15 OFM_BUFF2[LAY_OFMBUFF2_SIZE] = {[0 ... ((LAY_OFMBUFF2_SIZE)-1)] = 0x0000};   // !! @TODO CHANGE IF MODEL IS LARGER (rationale: Ow * Ow * F)
_q15 *pOFMBuff2   = (_q15*)OFM_BUFF2;


// buffer used for storing fc values before transfer to obuff
_q15 SRAM_BUFF[SRAM_BUFF_SIZE] = {[0 ... ((SRAM_BUFF_SIZE)-1)] = 0x0000};
_q15 *pSRAMBuff   = (_q15*)SRAM_BUFF;
