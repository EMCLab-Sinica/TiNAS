/*
 * cnn_add.c
 * Add layer
 */

#include <stdint.h>

#include "cnn.h"
#include "cnn_add.h"
#include "cnn_buffer_sizes.h"
#include "cnn_common.h"
#include "cnn_types.h"
#include "cnn_utils.h"
#include "../utils/myuart.h"
#include "../utils/extfram.h"
#include "../utils/myuart.h"
#include "../utils/util_macros.h"

static void CNN_ClearFootprints_Add(uint8_t idxBuf) {
    SPI_ADDR A;
    A.L = LOC_LOOP_INDICES;
    AddTileIndices idx = { 0, 0, 0 };
    SPI_WRITE(&A, (uint8_t*)&idx, sizeof(AddTileIndices));
}

// =============== BEGIN CNN_Intermittent_Add COST MODEL ===============
// Number of power cycles: N/Tn * R/Tr * C/Tc / preSz
// Fetch tile idx: DMA READ 4 Q15
// Fetch IFM: _fetch_ifm_tile_RCN_ext
// Fetch weights: _fetch_ifm_tile_RCN_ext (for every tile)
// Fetch OFM: 0 (no partial sums)
// Addressing: (2*MUL + 2*ADD) (same as stable power for one tile)
// Computation: _intra_tile_add (same as stable power for one tile)
// Tile output backup: _save_ofm_tile_RCN_ext (same as stable power for one tile)
// Tile idx backup: DMA WRITE 4 Q15
// ================ END CNN_Intermittent_Add COST MODEL ================

void CNN_Intermittent_Add(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf) {

    AddTileIndices idx;

    SPI_ADDR A;
    A.L = LOC_LOOP_INDICES;
    SPI_READ(&A, (uint8_t*)&idx, sizeof(AddTileIndices)); //fetch loop indices

    uint16_t row = idx.r, col = idx.c, ti = idx.n;
    uint16_t iteration = 0;  // number of completed tiles

    // report progress
    // loop order: NHW
    uint32_t num_tiles_W = ROUND(ofm->w, parE->Tc),
             num_tiles_HW = ROUND(ofm->h, parE->Tr) * num_tiles_W;
    uint32_t tile_idx = ti / parE->Tn * num_tiles_HW + row / parE->Tr * num_tiles_W + col / parE->Tc;
    _DBGUART("P,%d,%d\r\n", lid, tile_idx);
    // _DBGUART("_,%d,%d,%d\r\n", row, col, ti);

    _q15 *pLEAMEM = CNN_GetLEAMemoryLocation();
    // For Add, an IFM tile has the same size as an OFM tile, and OFM
    // does not need additional buffer as the operator uses in-place update
    _q15 *pIFM_t_buff, *pWeight_t_buff;

    uint16_t ifm_t_h = parE->Tr;
    uint16_t ifm_t_w = parE->Tc;

    uint32_t ifm_t_sz = parE->Tn * ifm_t_h * ifm_t_w;
    uint32_t weight_t_sz = ifm_t_sz;

    pIFM_t_buff   = pLEAMEM;
    pWeight_t_buff = pIFM_t_buff + ifm_t_sz + (ifm_t_sz & 0x1);

    // No data reuse for Add, just pick a random loop order
    // =============== Inter-Tile =========================
    for (; ti < ifm->ch; ti += parE->Tn){
        for (; row < ofm->h; row += parE->Tr) {
            for (; col < ofm->w; col += parE->Tc) {
                if(iteration == parP->preSz ){
                    _save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pIFM_t_buff, idx.r, idx.c, idx.n, parE->Tn, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);

                    idx.r = row;
                    idx.c = col;
                    idx.n = ti;
                    A.L = LOC_LOOP_INDICES;
                    SPI_WRITE(&A, (uint8_t*)&idx, sizeof(AddTileIndices));

                    iteration = 0;  // reset the counter to start the next parP->preSz tiles

                    _SHUTDOWN_AFTER_TILE();
                }

                _fetch_ifm_tile_RCN_ext(ifm, pIFM_t_buff, ifm_t_h, ifm_t_w, row, col, ti, parE->Tn);
                // load weight tiles via _fetch_ifm_tile_RCN_ext, as weight is actually another input
                _fetch_ifm_tile_RCN_ext(weights, pWeight_t_buff, ifm_t_h, ifm_t_w, row, col, ti, parE->Tn);

                // =============== Intra-Tile =========================
                _intra_tile_add(parE, weights, ifm_t_w, ti, pIFM_t_buff, pWeight_t_buff, parP->preSz, parE->lpOdr);
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
        SPI_WRITE(&A, (uint8_t*)&idx, sizeof(AddTileIndices));

        iteration = 0;

        _SHUTDOWN_AFTER_TILE();
    }

    CNN_ClearFootprints_Add(idxBuf);
}

