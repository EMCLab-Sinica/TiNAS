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

// =============== BEGIN CNN_GlobalAveragePool COST MODEL ===============
// Fetch IFM: M/Tm * R/Tr * C/Tc * _fetch_ifm_tile_RCN_ext
// Fetch weights: 0
// Fetch OFM: 0
// Addressing & Computation: M/Tm * R/Tr * C/Tc * (Tr*Tc*Tm * (1*MUL + 1*DIV + 4*ADD + 2*Shift (assumed similar to ADD)))
// Tile output backup: M/Tm * R/Tr * C/Tc / preSz * _save_ofm_tile_RCN_ext
// ================ END CNN_GlobalAveragePool COST MODEL ================

void CNN_GlobalAveragePool(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){

	ConvTileIndices idx;
	uint16_t iteration = 0;  // number of completed tiles
	_q15 *pLEAMEM = CNN_GetLEAMemoryLocation();
	_q15 *pIFM_t_buff, *pOFM_t_buff;
	uint16_t row=0, col=0, to=0;
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
        }

		for (; row<ifm->h; row+=parE->Tr){
			for (; col<ifm->w; col+=parE->Tc){
                if(iteration==parP->preSz ){
                	_save_ofm_tile_RCN_ext(ofm, 0 , pOFM_t_buff+idx.m, 0, 0, idx.m, parE->Tm, 1, 1, parP->preSz, OFM_ORIENTED);
                	idx.r = row;
                	idx.c = col;
                	idx.m = to;
                    iteration=0;  //reset the

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
		iteration=0;  //reset the
	}
}
