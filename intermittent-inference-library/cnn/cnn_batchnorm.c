/*
 * cnn_batchnorm.c
 * BatchNormalization layer
 */

#include <stdint.h>

#include "cnn.h"
#include "cnn_batchnorm.h"
#include "cnn_common.h"
#include "cnn_types.h"
#include "cnn_utils.h"

// ============== BEGIN _intra_tile_batchnorm COST MODEL ==============
// Number of loop iterations: (Tr*Tc*Tn)
// Addressing: 2*MUL + 7*ADD
// Computation: SUB + 2*MPY + ADD + Shift (assume similar to ADD)
// =============== END _intra_tile_batchnorm COST MODEL ===============

void _intra_tile_batchnorm(ExeParams_t* parE, Mat_t* weights, uint16_t ifm_t_w, uint16_t ti, _q15* pIFM_t_buff,_q15* pScale_t_buff, _q15* pB_t_buff, _q15* pMean_t_buff, _q15* pVar_t_buff, uint16_t preSz, LOOPORDER_t lpOdr) {
    for (uint16_t trr = 0; trr < parE->Tr; trr++) {
        for (uint16_t tcc = 0; tcc < parE->Tc; tcc++) {
            for (uint16_t tii = 0; tii < parE->Tn; tii++) {
                // HWC memory layout for IFM, HWNC memory layout for weights
                uint32_t IFM_offset = (trr * ifm_t_w + tcc) * parE->Tn + tii;
                uint32_t weight_offset = tii;
                _q15 *pIFM = pIFM_t_buff + IFM_offset,
                     *pMean = pMean_t_buff + weight_offset,
                     *pScale = pScale_t_buff + weight_offset,
                     *pVar = pVar_t_buff + weight_offset,
                     *pB = pB_t_buff + weight_offset;

                // x - mean
                *pIFM -= (*pMean);

                // (x - mean)*scale
                *pIFM = my_q15mpy(*pIFM, *pScale);

                // (x - mean)*scale/sqrt(var+epsilon)
                *pIFM = my_q15mpy(*pIFM, *pVar);  // pVar is inverted offline -> 1/pVar

                // (x - mean)*scale/sqrt(var+epsilon)+B
                *pIFM += (*pB);
            }
        }
    }
}

// =============== BEGIN CNN_BatchNormalization COST MODEL ===============
// Fetch IFM: (N/Tn * R/Tr * C/Tc) * _fetch_ifm_tile_RCN_ext
// Fetch weights: (N/Tn) * SPI_READ(Tn*Q15_SIZE) * 4
// Fetch OFM: 0 (no partial sums)
// Addressing (for weights): (N/Tn) * (7*ADD)
// Computation: (N/Tn * R/Tr * C/Tc) * _intra_tile_batchnorm
// Tile output backup: (N/Tn * R/Tr * C/Tc / preSz) * _save_ofm_tile_RCN_ext
// ================ END CNN_BatchNormalization COST MODEL ================
void CNN_BatchNormalization(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf) {

    const uint16_t CHANNEL = ifm->ch;

    BNTileIndices idx;
    uint16_t row = 0, col = 0, ti = 0;
    uint16_t iteration = 0;  // number of completed tiles

    _q15 *pLEAMEM = CNN_GetLEAMemoryLocation();
    // For BatchNorm, an IFM tile has the same size as an OFM tile, and OFM
    // does not need additional buffer as the operator uses in-place update
    _q15 *pIFM_t_buff, *pScale_t_buff, *pB_t_buff, *pMean_t_buff, *pVar_t_buff;

    uint16_t ifm_t_h = parE->Tr;
    uint16_t ifm_t_w = parE->Tc;

    uint32_t ifm_t_sz = parE->Tn * ifm_t_h * ifm_t_w;
    uint32_t weight_t_sz = parE->Tn * 4; // 4 vectors: scale, B, mean, var

    pIFM_t_buff   = pLEAMEM;
    pScale_t_buff = pIFM_t_buff   + ifm_t_sz + (ifm_t_sz & 0x1);
    pB_t_buff     = pScale_t_buff + parE->Tn + (parE->Tn & 0x1);
    pMean_t_buff  = pB_t_buff     + parE->Tn + (parE->Tn & 0x1);
    pVar_t_buff   = pMean_t_buff  + parE->Tn + (parE->Tn & 0x1);

    const uint32_t Scale_base_addr = weights->data,
                   B_base_addr     = Scale_base_addr + CHANNEL*Q15_SIZE,
                   Mean_base_addr  = B_base_addr     + CHANNEL*Q15_SIZE,
                   Var_base_addr   = Mean_base_addr  + CHANNEL*Q15_SIZE;

    // Always use weight reuse loop order, as it's the only helpful one for BatchNorm
    // =============== Inter-Tile =========================
    for (; ti < ifm->ch; ti += parE->Tn){
        // load weight tiles
        // Assume weights are scale, B, mean, var in order
        const uint32_t Scale_addr = Scale_base_addr + ti*Q15_SIZE;
        const uint32_t B_addr     = B_base_addr     + ti*Q15_SIZE;
        const uint32_t Mean_addr  = Mean_base_addr  + ti*Q15_SIZE;
        const uint32_t Var_addr   = Var_base_addr   + ti*Q15_SIZE;

        /* Fetching weights */
        memcpy_dma_ext(Scale_addr, pScale_t_buff, parE->Tn*Q15_SIZE, parE->Tn*Q15_SIZE, MEMCPY_READ);
        memcpy_dma_ext(B_addr,     pB_t_buff,     parE->Tn*Q15_SIZE, parE->Tn*Q15_SIZE, MEMCPY_READ);
        memcpy_dma_ext(Mean_addr,  pMean_t_buff,  parE->Tn*Q15_SIZE, parE->Tn*Q15_SIZE, MEMCPY_READ);
        memcpy_dma_ext(Var_addr,   pVar_t_buff,   parE->Tn*Q15_SIZE, parE->Tn*Q15_SIZE, MEMCPY_READ); // var is already offseted by epsilon and square-rooted offline

        for (; row < ofm->h; row += parE->Tr) {
            for (; col < ofm->w; col += parE->Tc) {
                if(iteration == parP->preSz ){
                    _save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pIFM_t_buff, idx.r, idx.c, idx.n, parE->Tn, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
                    iteration = 0;  // reset the counter to start the next parP->preSz tiles
                }

                _fetch_ifm_tile_RCN_ext(ifm, pIFM_t_buff, ifm_t_h, ifm_t_w, row, col, ti, parE->Tn);

                // =============== Intra-Tile =========================
                _intra_tile_batchnorm(parE, weights, ifm_t_w, ti, pIFM_t_buff, pScale_t_buff, pB_t_buff, pMean_t_buff, pVar_t_buff, parP->preSz, parE->lpOdr);
                iteration++;
                idx.r = row;
                idx.c = col;
                idx.n = ti;
            }
            col = 0;
        }
        row = 0;
    }

    if(iteration == parP->preSz) {
        _save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pIFM_t_buff, idx.r, idx.c, idx.n, parE->Tn, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
        iteration = 0;
    }
}
