/*
 * cnn_batchnorm.c
 * BatchNormalization layer
 */

#include <stdint.h>

#include "cnn.h"
#include "cnn_batchnorm.h"
#include "cnn_buffer_sizes.h"
#include "cnn_common.h"
#include "cnn_types.h"
#include "cnn_utils.h"
#include "../utils/extfram.h"
#include "../utils/myuart.h"
#include "../utils/util_macros.h"

static void CNN_ClearFootprints_BatchNormalization(uint8_t idxBuf) {
    SPI_ADDR A;
    A.L = LOC_LOOP_INDICES;
    BNTileIndices idx = { 0, 0, 0 };
    SPI_WRITE(&A, (uint8_t*)&idx, sizeof(BNTileIndices));
}

// =============== BEGIN CNN_BatchNormalization COST MODEL ===============
// Number of power cycles: M/Tm * R/Tr * C/Tc / preSz
// Fetch tile idx: DMA READ 4 Q15
// Fetch IFM: _fetch_ifm_tile_RCN_ext
// Fetch weights: 4*Tn (for every tile)
// Fetch OFM: 0 (no partial sums)
// Addressing: (7*ADD) (same as stable power for one tile)
// Computation: _intra_tile_batchnorm (same as stable power for one tile)
// Tile output backup: _save_ofm_tile_RCN_ext (same as stable power for one tile)
// Tile idx backup: DMA WRITE 4 Q15
// ================ END CNN_BatchNormalization COST MODEL ================
void CNN_Intermittent_BatchNormalization(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf) {

    const uint16_t CHANNEL = ifm->ch;

    BNTileIndices idx;

    SPI_ADDR A;
    A.L = LOC_LOOP_INDICES;
    SPI_READ(&A, (uint8_t*)&idx, sizeof(BNTileIndices)); //fetch loop indices

    uint16_t row = idx.r, col = idx.c, ti = idx.n;
    uint16_t iteration = 0;  // number of completed tiles

    // report progress
    // loop order: NHW
    uint32_t num_tiles_W = (ofm->w + parE->Tc - 1) / parE->Tc,
             num_tiles_HW = (ofm->h + parE->Tr - 1) / parE->Tr * num_tiles_W;
    uint32_t tile_idx = ti / parE->Tn * num_tiles_HW + row / parE->Tr * num_tiles_W + col / parE->Tc;
    _DBGUART("P,%d,%d\r\n", lid, tile_idx);
    // _DBGUART("_,%d,%d,%d\r\n", row, col, ti);

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

                    idx.r = row;
                    idx.c = col;
                    idx.n = ti;
                    A.L = LOC_LOOP_INDICES;
                    SPI_WRITE(&A, (uint8_t*)&idx, sizeof(BNTileIndices));

                    iteration = 0;  // reset the counter to start the next parP->preSz tiles

                    _SHUTDOWN_AFTER_TILE();
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

        idx.r = row;
        idx.c = col;
        idx.n = ti;
        A.L = LOC_LOOP_INDICES;
        SPI_WRITE(&A, (uint8_t*)&idx, sizeof(BNTileIndices));

        iteration = 0;

        _SHUTDOWN_AFTER_TILE();
    }

    CNN_ClearFootprints_BatchNormalization(idxBuf);
}
