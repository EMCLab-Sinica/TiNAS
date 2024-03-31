/*
 * cnn_relu.c
 * ReLU layer
 */

#include <stdint.h>

#include "cnn.h"
#include "cnn_common.h"
#include "cnn_relu.h"
#include "cnn_types.h"
#include "cnn_utils.h"

// ============== BEGIN _intra_tile_relu COST MODEL ==============
// Number of loop iterations: Tr*Tc
// Addressing: 2*MPY + 3*ADD
// Computation: 1*COMPARE
// =============== END _intra_tile_relu COST MODEL ===============

void _intra_tile_relu(ExeParams_t* parE, uint16_t ifm_t_w, _q15* pIFM_t_buff) {
    for (uint16_t trr = 0; trr < parE->Tr; trr++) {
        for (uint16_t tcc = 0; tcc < parE->Tc; tcc++) {
            for (uint16_t tii = 0; tii < parE->Tn; tii++) {
                uint32_t valIFM_offset = trr * ifm_t_w * parE->Tn + tcc * parE->Tn + tii;
                _q15* pValIFM = pIFM_t_buff + valIFM_offset;

                if (*pValIFM < 0) {
                    *pValIFM = 0;
                }
            }
        }
    }
}

// =============== BEGIN CNN_ReLU COST MODEL ===============
// Fetch IFM: N/Tn * R/Tr * C/Tc * _fetch_ifm_tile_RCN_ext
// Fetch weights: 0
// Fetch OFM: 0
// Addressing: 0
// Computation: N/Tn * R/Tr * C/Tc * _intra_tile_relu
// Tile output backup: N/Tn * R/Tr * C/Tc / preSz * _save_ofm_tile_RCN_ext
// ================ END CNN_ReLU COST MODEL ================

void CNN_ReLU(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf) {

    const uint16_t CHANNEL = ifm->ch;

    ReLUTileIndices idx;
    uint16_t row = 0, col = 0, ti = 0;
    uint16_t iteration = 0;  // number of completed tiles

    _q15 *pLEAMEM = CNN_GetLEAMemoryLocation();
    // For ReLU, an IFM tile has the same size as an OFM tile, and OFM
    // does not need additional buffer as the operator uses in-place update
    _q15 *pIFM_t_buff;

    uint16_t ifm_t_h = parE->Tr;
    uint16_t ifm_t_w = parE->Tc;

    pIFM_t_buff   = pLEAMEM;

    // The loop order doesn't matter for ReLU. Here we use weight reuse loop order
    // =============== Inter-Tile =========================
    for (; ti < ifm->ch; ti += parE->Tn){

        for (; row < ofm->h; row += parE->Tr) {
            for (; col < ofm->w; col += parE->Tc) {
                if(iteration == parP->preSz ){
                    _save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pIFM_t_buff, idx.r, idx.c, idx.n, parE->Tn, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
                    iteration = 0;  // reset the counter to start the next parP->preSz tiles
                }

                _fetch_ifm_tile_RCN_ext(ifm, pIFM_t_buff, ifm_t_h, ifm_t_w, row, col, ti, parE->Tn);

                // =============== Intra-Tile =========================
                _intra_tile_relu(parE, ifm_t_w, pIFM_t_buff);
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
