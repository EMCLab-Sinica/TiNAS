/*
 * cnn_pool.c
 * Pooling layer
 * Currently supports: Global Average Pooling
 */

#include "driverlib.h"


#include "../utils/extfram.h"
#include "../utils/myuart.h"
#include "../utils/util_macros.h"
#include "cnn_buffer_sizes.h"

#include "cnn.h"

#include "cnn_common.h"
#include "cnn_matops.h"

#include "cnn_pool.h"
#include "cnn_utils.h"

#pragma PERSISTENT(FP_LayerPool)
FootprintsLayerPool_t FP_LayerPool= {0};

// =============== BEGIN CNN_Intermittent_GlobalAveragePool COST MODEL ===============
// Number of power cycles: (M/Tm * R/Tr * C/Tc / preSz)
// Fetch tile idx: SPI_READ(10)
// Fetch IFM: preSz * _fetch_ifm_tile_RCN_ext
// Fetch weights: 0
// Fetch OFM: _fetch_ofm_tile_RCN_ext, except for row==0 && col==0
// Addressing: 2*MUL + 1*ADD
// Computation: preSz * (Tr*Tc*Tm * ADD)
// Tile output backup: _save_ofm_tile_RCN_ext
// Tile idx backup: SPI_WRITE(10)
// ================ END CNN_Intermittent_GlobalAveragePool COST MODEL ================
void CNN_Intermittent_GlobalAveragePool(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){

	SPI_ADDR A;
	ConvTileIndices idx;
	A.L = LOC_LOOP_INDICES;
	SPI_READ(&A,(uint8_t*)&idx,sizeof(ConvTileIndices)); //fetch loop indices
	uint16_t row=idx.r, col=idx.c, to=idx.m;
	uint16_t iteration = 0;  // number of completed tiles

    // report progress
    // loop order: MHW
    uint32_t num_tiles_W = ROUND(ofm->w, parE->Tc),
             num_tiles_HW = ROUND(ofm->h, parE->Tr) * num_tiles_W;
    uint32_t tile_idx = to / parE->Tm * num_tiles_HW + row / parE->Tr * num_tiles_W + col / parE->Tc;
    _DBGUART("P,%d,%d\r\n", lid, tile_idx);
    // _DBGUART("_,%d,%d,%d\r\n", row, col, to);

	_q15 *pLEAMEM = CNN_GetLEAMemoryLocation();
	_q15 *pIFM_t_buff, *pOFM_t_buff;

	volatile _iq31 resiq31 = 0;
	volatile _q15 resq15 = 0;
	uint16_t ifm_t_sz = (parE->Tm * parE->Tc * parE->Tr);

	if( (  ( (ifm->h / parE->Tr) * (ifm->w / parE->Tc) ) % parP->preSz) !=0 ){_DBGUART("CNN_Intermittent_GlobalAveragePool:: Error - wrong preSz sizes %d\r\n",parP->preSz);_STOP();}
	pIFM_t_buff = pLEAMEM;
	pOFM_t_buff = pIFM_t_buff + ifm_t_sz;
    
    // ===== Inter tile =========
	for (; to<ofm->ch; to+=parE->Tm){

        if( (row ==0 ) && (col == 0) ){
        	for (uint16_t too=0; too < parE->Tm; too++)pOFM_t_buff[to+too] = 0;
        }else{
        	_fetch_ofm_tile_RCN_ext( ofm , pOFM_t_buff+idx.m, 0, 0, to, parE->Tm, 1, 1, parP->preSz, OFM_ORIENTED);
        }

		for (; row<ifm->h; row+=parE->Tr){
			for (; col<ifm->w; col+=parE->Tc){
                if(iteration==parP->preSz ){
                	_save_ofm_tile_RCN_ext(ofm, 0 , pOFM_t_buff+idx.m, 0, 0, idx.m, parE->Tm, 1, 1, parP->preSz, OFM_ORIENTED);
                	idx.r = row;
                	idx.c = col;
                	idx.m = to;
                	A.L = LOC_LOOP_INDICES;
                	SPI_WRITE(&A,(uint8_t*)&idx, sizeof(ConvTileIndices));
                    iteration=0;  //reset the
                    _SHUTDOWN_AFTER_TILE();
                }
                // load ifm tile *
                _fetch_ifm_tile_RCN_ext(ifm, pIFM_t_buff, parE->Tr, parE->Tc, row, col, to, parE->Tm);

                // ===== Intra tile =========
                uint16_t offset = 0;
                for(uint16_t trr=0; trr < parE->Tr ; trr++){  // Tr => Tri   partition IFM
                	for(uint16_t tcc=0; tcc < parE->Tc ; tcc++){ // Tc => Tci
                		for (uint16_t too=0; too < parE->Tm; too++){
#ifdef __MSP430__
                			pOFM_t_buff[to+too] = pOFM_t_buff[to+too] + pIFM_t_buff[ offset] / (ifm->h * ifm->w );
#else
                			pOFM_t_buff[to+too] = pOFM_t_buff[to+too] + pIFM_t_buff[ offset] / (ifm->h * ifm->w );
#endif
                			offset++;
                		}
                	}
                }
                iteration++;
               	idx.r = row;
                idx.c = col;
                idx.m = to;
			}col=0;
		}row=0;
	}to=0;

	if(iteration==parP->preSz ){
		_save_ofm_tile_RCN_ext(ofm, 0 , pOFM_t_buff+idx.m, 0, 0, idx.m, parE->Tm, 1, 1, parP->preSz, parE->lpOdr);
		idx.r = ifm->h;
		idx.c = ifm->w;
		idx.m = ofm->ch;
		A.L = LOC_LOOP_INDICES;
		SPI_WRITE(&A,(uint8_t*)&idx, sizeof(ConvTileIndices));
		iteration=0;  //reset the
		_SHUTDOWN_AFTER_TILE();
	}
	A.L = LOC_LOOP_INDICES;
	ConvTileIndices idx_clr={0,0,0,0};
	SPI_WRITE(&A,(uint8_t*)&idx_clr,sizeof(ConvTileIndices));
}

void CNN_ClearFootprints_LayerMaxPool(){
    FP_LayerPool.ch=0;
}
