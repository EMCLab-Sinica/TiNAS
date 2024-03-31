/*
 * cnn_add.c
 * Add layer
 */

#include <stdint.h>

#include "cnn.h"
#include "cnn_add.h"
#include "cnn_common.h"
#include "cnn_types.h"
#include "cnn_utils.h"
#include "../utils/myuart.h"

// ============== BEGIN _intra_tile_add COST MODEL ==============
// Number of loop iterations: Tr*Tc*Tn
// Addressing: 2*MUL + 4*ADD
// Computation: 1*ADD
// =============== END _intra_tile_add COST MODEL ===============

void _intra_tile_add(ExeParams_t* parE, Mat_t* weights, uint16_t ifm_t_w, uint16_t ti, _q15* pIFM_t_buff,_q15* pWeight_t_buff, uint16_t preSz, LOOPORDER_t lpOdr) {
    for (uint16_t trr = 0; trr < parE->Tr; trr++) {
        for (uint16_t tcc = 0; tcc < parE->Tc; tcc++) {
            for (uint16_t tii = 0; tii < parE->Tn; tii++) {
                // HWC memory layout for IFM, HWNC memory layout for weights
                uint32_t IFM_offset = trr * ifm_t_w * parE->Tn + tcc * parE->Tn + tii;
                uint32_t Weight_offset = IFM_offset;
                _q15* pIFM = pIFM_t_buff + IFM_offset;
                _q15* pWeight = pWeight_t_buff + Weight_offset;

                *pIFM = (*pIFM) + (*pWeight);
            }
        }
    }
}

// =============== BEGIN CNN_Add COST MODEL ===============
// Fetch IFM: N/Tn * R/Tr * C/Tc * _fetch_ifm_tile_RCN_ext
// Fetch weights: N/Tn * R/Tr * C/Tc * _fetch_ifm_tile_RCN_ext
// Fetch OFM: 0
// Addressing: 2*MUL + 2*ADD
// Computation: N/Tn * R/Tr * C/Tc * _intra_tile_add
// Tile output backup: N/Tn * R/Tr * C/Tc / preSz * _save_ofm_tile_RCN_ext
// ================ END CNN_Add COST MODEL ================
void CNN_Add(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf) {

    AddTileIndices idx;
    uint16_t row = 0, col = 0, ti = 0;
    uint16_t iteration = 0;  // number of completed tiles

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
                    iteration = 0;  // reset the counter to start the next parP->preSz tiles
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
        iteration = 0;
    }
}

