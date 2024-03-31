/*
 * cnn_conv_tiled_std.c
 * Depthwise Tiled Convolution
 */

#include "driverlib.h"

#include "../utils/extfram.h"
#include "../utils/myuart.h"
#include "../utils/util_macros.h"

#include "cnn.h"
#include "cnn_matops.h"

#include "cnn_conv_tiled_dw.h"

#include "cnn_buffer_sizes.h"
#include "cnn_utils.h"


static void _CNN_LayerConvDW_Tiled_OFM(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);
static void _CNN_LayerConvDW_Tiled_IFM(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);
static void _CNN_LayerConvDW_Tiled_WEI(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);

// ============== BEGIN _intra_tile_conv_dw COST MODEL ==============
// Number of loop iterations: (Tr*Tc*Kh*Kw*Tm)
// Addressing (IFM_ORIENTED): 12*MUL + 11*ADD
// Addressing (others):       10*MUL + 11*ADD
// Computation: 1*MUL
// =============== END _intra_tile_conv_dw COST MODEL ===============
// Intra tile convolution loops
void _intra_tile_conv_dw(ExeParams_t *parE, Mat_t* weights, uint16_t ifm_t_w, uint16_t ti, _q15 * pIFM_t_buff,_q15 * pWeight_t_buff,_q15* pOFM_t_buff,_iq31* pVMAC_result, uint16_t preSz, LOOPORDER_t lpOdr){
	_q15* pVec1;
	_q15* pVec2;
	for (uint16_t trr=0; trr<parE->Tr; trr++){
		for (uint16_t tcc=0; tcc<parE->Tc; tcc++){
			for (uint16_t i=0; i<weights->h; i++){
				for (uint16_t j=0; j<weights->w; j++){
					for (uint16_t too=0; too<parE->Tm; too++){
						// calculate input locations
						// HWC memory layout for IFM, HWNC memory layout for weights
						uint16_t tii = too;
						uint16_t v1_offset = ((((((parE->str*trr)+i)) * ifm_t_w) + ((parE->str*tcc)+j))*parE->Tn) + tii;
						uint16_t v2_offset =  ( (parE->Tm * weights->h * i) + (parE->Tm * j) + tii );
						pVec1 = pIFM_t_buff + v1_offset;
						pVec2 = pWeight_t_buff + v2_offset;
						_q15 *tmp = 0;
						if(lpOdr == IFM_ORIENTED){
							tmp = (pOFM_t_buff + (trr * parE->Tc * parE->Tm * preSz) + (tcc * parE->Tm * preSz) + too);
						}else{
							tmp = (pOFM_t_buff + (trr * parE->Tc * parE->Tm) + (tcc * parE->Tm) + too);
						}

						*tmp = my_q15mpy(*pVec1, *pVec2);
					}

				}
			}
		}
	}
}

/*******************************************************************************************************************
 Tiled Convolution using LEA vector MAC command
 exec design params : Tr, Tc, Tm, Tn, loop order, preservation batch size
 Tn MACs performed in one command
 [UNDER CONTINUOUS POWER]
*******************************************************************************************************************/
void CNN_LayerConv_Tiled_Depthwise(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){
    uint16_t ifm_t_h = (weights->h + (parE->str * parE->Tr) - parE->str);
    uint16_t ifm_t_w = (weights->w + (parE->str * parE->Tc) - parE->str);

    // 1D local buffer sizes and locations (directly inside LEA memory)
    uint16_t ifm_t_sz = (parE->Tn * ifm_t_h * ifm_t_w);
    uint16_t weight_t_sz = (parE->Tm * weights->h * weights->w);
    uint16_t ofm_t_sz = (parE->Tm * parE->Tr * parE->Tc); // TODO: fix for presSz !

    if(OFM_ORIENTED == parE->lpOdr){
		if ((ifm_t_sz + weight_t_sz + ofm_t_sz) > LEA_MEM_SIZE){
			_DBGUART("CNN_LayerConv_Tiled_Depthwise:: Error - tile sizes too large %d, %d, %d\r\n", ifm_t_sz, weight_t_sz, ofm_t_sz); _STOP();
		}
    }else{
		if ((ifm_t_sz + weight_t_sz + ofm_t_sz*parP->preSz) > LEA_MEM_SIZE){
			_DBGUART("CNN_LayerConv_Tiled_Depthwise:: Error - tile sizes too large %d, %d, %d\r\n", ifm_t_sz, weight_t_sz, ofm_t_sz); _STOP();
		}
    }

    /*
    if (
         ((ofm->h     % parE->Tr) !=0) ||
         ((ofm->w     % parE->Tc) !=0) ||
         ((weights->n % parE->Tm) !=0) ||
         ((ifm->ch    % parE->Tn) !=0) ){
        _DBGUART("CNN_LayerConv_Tiled_Depthwise:: Error - wrong tile sizes %d|%d, %d|%d, %d|%d, %d|%d\r\n",
            ofm->h,     parE->Tr,
            ofm->w,     parE->Tc,
            weights->n, parE->Tm,
            ifm->ch,    parE->Tn);
        _STOP();
    }
    */

    // main handler
	switch(parE->lpOdr){
		case OFM_ORIENTED:
			if( ((ifm->ch / parE->Tn) % parP->preSz) !=0 ){_DBGUART("CNN_LayerConv_Tiled_OFM:: Error - wrong preSz sizes %d\r\n",parP->preSz);_STOP();}
			_CNN_LayerConvDW_Tiled_OFM( lid, weights, bias, ifm, ofm, parE, parP,idxBuf);break;
		case IFM_ORIENTED:
			if( ((ofm->ch / parE->Tm) % parP->preSz) !=0 ){_DBGUART("CNN_LayerConv_Tiled_IFM:: Error - wrong preSz sizes %d\r\n",parP->preSz);_STOP();}
			_CNN_LayerConvDW_Tiled_IFM( lid, weights, bias, ifm, ofm, parE, parP,idxBuf);break;
		case WEIGHT_ORIENTED:
			if( ( ((ofm->w / parE->Tc)*(ofm->h / parE->Tr)) % parP->preSz) !=0 ){_DBGUART("CNN_LayerConv_Tiled_WEI:: Error - wrong preSz sizes %d\r\n",parP->preSz);_STOP();}
			_CNN_LayerConvDW_Tiled_WEI( lid, weights, bias, ifm, ofm, parE, parP,idxBuf);break;
		case NONE_ORIENTED:
			break;
	}
}

// =============== BEGIN _CNN_LayerConvDW_Tiled_OFM COST MODEL ===============
// Fetch IFM: (R/Tr * C/Tc* N/Tn) * _fetch_ifm_tile_RCN_ext
// Fetch weights: (R/Tr * C/Tc * N/Tn) * _fetch_weights_tile_RCMN_ext
// Fetch OFM: 0
// Computation: (R/Tr * C/Tc * N/Tn) * _intra_tile_conv_dw
// Tile output backup: (R/Tr * C/Tc * N/Tn / preSz) * _save_ofm_tile_RCN_ext
// ================ END _CNN_LayerConvDW_Tiled_OFM COST MODEL ================
void _CNN_LayerConvDW_Tiled_OFM(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){

    ConvTileIndices idx;
    uint16_t row=0, col=0, ti=0;
    uint16_t iteration = 0;  // number of completed tiles

    _q15 *pLEAMEM = CNN_GetLEAMemoryLocation();
    _q15 *pIFM_t_buff, *pWeight_t_buff, *pOFM_t_buff;
    _iq31 *pVMAC_result;

    volatile _iq31 resiq31 = 0;
    volatile _q15 resq15 = 0;
    uint16_t ifm_t_h = (weights->h + (parE->str * parE->Tr) - parE->str);
    uint16_t ifm_t_w = (weights->w + (parE->str * parE->Tc) - parE->str);

    // 1D local buffer sizes and locations (directly inside LEA memory)
    uint16_t ifm_t_sz = (parE->Tn * ifm_t_h * ifm_t_w);
    uint16_t weight_t_sz = (parE->Tm * weights->h * weights->w);
    uint16_t ofm_t_sz = (parE->Tm * parE->Tr * parE->Tc);

    pIFM_t_buff = pLEAMEM;
    pWeight_t_buff = pIFM_t_buff + ifm_t_sz + (ifm_t_sz & 0x1) ;
    pOFM_t_buff = pWeight_t_buff + weight_t_sz + (weight_t_sz & 0x1);
    pVMAC_result = (_iq31*)( (pOFM_t_buff + ofm_t_sz) + (ofm_t_sz & 0x1)  ); // scalar vec mac result

    // =============== Inter-Tile =========================
    for (; row<ofm->h; row+=parE->Tr){
        for (; col<ofm->w; col+=parE->Tc){
                for (; ti<ifm->ch; ti+=parE->Tn){
                    if(iteration==parP->preSz ){
                    	_save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pOFM_t_buff, idx.r, idx.c, idx.m, parE->Tm, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
                        iteration=0;  //reset the
                    }
                    // load ifm tile *
                    _fetch_ifm_tile_RCN_ext(ifm, pIFM_t_buff, ifm_t_h, ifm_t_w, row, col, ti, parE->Tn);
                    // load weight tile
                    _fetch_weights_tile_RCMN_ext(weights, pWeight_t_buff, /*to=*/0, ti, parE->Tm, /*t_n=*/1);

                   // =============== Intra-Tile =========================
                    _intra_tile_conv_dw(parE,weights,ifm_t_w,ti,pIFM_t_buff,pWeight_t_buff,pOFM_t_buff,pVMAC_result, parP->preSz, parE->lpOdr);
                    iteration++;
                   	idx.r = row;
                    idx.c = col;
                    idx.m = ti;
                    idx.n = ti;
                }ti=0;
        }col=0;
    }
    if(iteration==parP->preSz ){
    	_save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pOFM_t_buff, idx.r, idx.c, idx.m, parE->Tm, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
        iteration=0;  //reset the
    }
}

// =============== BEGIN _CNN_LayerConvDW_Tiled_IFM COST MODEL ===============
// Fetch IFM: (R/Tr * C/Tc * N/Tn) * _fetch_ifm_tile_RCN_ext
// Fetch weights: (R/Tr * C/Tc * N/Tn) * _fetch_weights_tile_RCMN_ext
// Fetch OFM: 0
// Addressing (for OFM buffer): (R/Tr * C/Tc * N/Tn) * ADD +                     /* for every tile */
//                              (R/Tr * C/Tc * N/Tn / preSz) * (1*MPY + 2*ADD)   /* reset after each preservation */
// Computation: (R/Tr * C/Tc * N/Tn) * _intra_tile_conv_dw
// Tile output backup: (R/Tr * C/Tc * N/Tn / preSz) * _save_ofm_tile_RCN_ext
// ================ END _CNN_LayerConvDW_Tiled_IFM COST MODEL ================
void _CNN_LayerConvDW_Tiled_IFM(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){
    ConvTileIndices idx;
    uint16_t row=0, col=0, ti=0;
    uint16_t iteration = 0;  // number of completed tiles

    _q15 *pLEAMEM = CNN_GetLEAMemoryLocation();
    _q15 *pIFM_t_buff, *pWeight_t_buff, *pOFM_t_buff;
    _iq31 *pVMAC_result;

    volatile _iq31 resiq31 = 0;
    volatile _q15 resq15 = 0;

    uint16_t ifm_t_h = (weights->h + (parE->str * parE->Tr) - parE->str);
    uint16_t ifm_t_w = (weights->w + (parE->str * parE->Tc) - parE->str);

    // 1D local buffer sizes and locations (directly inside LEA memory)
    uint16_t ifm_t_sz = (parE->Tn * ifm_t_h * ifm_t_w);
    uint16_t weight_t_sz = (parE->Tm * weights->h * weights->w);
    uint16_t ofm_t_sz = (parE->Tm * parE->Tr * parE->Tc)* parP->preSz;

    pIFM_t_buff = pLEAMEM;
    pWeight_t_buff = pIFM_t_buff + ifm_t_sz + (ifm_t_sz & 0x1);
    pOFM_t_buff = pWeight_t_buff + weight_t_sz * parP->preSz + (weight_t_sz & 0x1);
    pVMAC_result = (_iq31*)( (pOFM_t_buff + ofm_t_sz) + (ofm_t_sz & 0x1)  ); // scalar vec mac result

    // =============== Inter-Tile =========================    
    for (; row<ofm->h; row+=parE->Tr){
        for (; col<ofm->w; col+=parE->Tc){
            for (; ti<ifm->ch; ti+=parE->Tn){
            	_fetch_ifm_tile_RCN_ext(ifm, pIFM_t_buff, ifm_t_h, ifm_t_w, row, col, ti, parE->Tn); // load ifm tile
                    if(iteration==parP->preSz ){
                    	_save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pOFM_t_buff- ofm_t_sz, idx.r, idx.c, idx.m, parE->Tm, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
                        iteration=0;
                        pOFM_t_buff = pWeight_t_buff + weight_t_sz * parP->preSz + (weight_t_sz & 0x1);
                    }
                    //
                    // load weight tile
                    _fetch_weights_tile_RCMN_ext(weights, pWeight_t_buff, /*to=*/0, ti, parE->Tm, /*t_n=*/1);
                    //
                   // =============== Intra-Tile =========================
                    _intra_tile_conv_dw(parE,weights,ifm_t_w,ti,pIFM_t_buff,pWeight_t_buff,pOFM_t_buff,pVMAC_result, parP->preSz, parE->lpOdr);
                   	idx.r = row;
                    idx.c = col;
                    idx.m = ti;
                    idx.n = ti;
                    iteration++;
                    pOFM_t_buff += ofm_t_sz;
            }ti=0;
        }col=0;
    }
    if(iteration==parP->preSz ){
    	_save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pOFM_t_buff - ofm_t_sz, idx.r, idx.c, idx.m, parE->Tm, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
        iteration=0;  //reset the
    }
}

// =============== BEGIN _CNN_LayerConvDW_Tiled_WEI COST MODEL ===============
// Fetch IFM: (N/Tn * R/Tr * C/Tc) * _fetch_ifm_tile_RCN_ext
// Fetch weights: (N/Tn) * _fetch_weights_tile_RCMN_ext
// Fetch OFM: 0
// Addressing (for OFM buffer): (N/Tn * R/Tr * C/Tc) * ADD +                     /* for every tile */
//                              (N/Tn * R/Tr * C/Tc / preSz) * (1*MPY + 2*ADD)   /* reset after each preservation */
// Computation: (N/Tn * R/Tr * C/Tc) * _intra_tile_conv_dw
// Tile output backup: (N/Tn * R/Tr * C/Tc / preSz) * _save_ofm_tile_RCN_ext
// ================ END _CNN_LayerConvDW_Tiled_WEI COST MODEL ================
void _CNN_LayerConvDW_Tiled_WEI(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){
	ConvTileIndices idx;
    uint16_t row=0, col=0, ti=0;
    uint16_t iteration = 0;  // number of completed tiles

    _q15 *pLEAMEM = CNN_GetLEAMemoryLocation();
    _q15 *pIFM_t_buff, *pWeight_t_buff, *pOFM_t_buff;
    _iq31 *pVMAC_result;
    
    volatile _iq31 resiq31 = 0;
    volatile _q15 resq15 = 0;
    uint16_t ifm_t_h = (weights->h + (parE->str * parE->Tr) - parE->str);
    uint16_t ifm_t_w = (weights->w + (parE->str * parE->Tc) - parE->str);

    // 1D local buffer sizes and locations (directly inside LEA memory)
    uint16_t ifm_t_sz = (parE->Tn * ifm_t_h * ifm_t_w);
    uint16_t weight_t_sz = (parE->Tm * weights->h * weights->w);
    uint16_t ofm_t_sz = (parE->Tm * parE->Tr * parE->Tc)* parP->preSz;

    pIFM_t_buff = pLEAMEM;
    pWeight_t_buff = pIFM_t_buff + ifm_t_sz + (ifm_t_sz & 0x1) ;
    pOFM_t_buff = pWeight_t_buff + weight_t_sz * parP->preSz + (weight_t_sz & 0x1);
    pVMAC_result = (_iq31*)( (pOFM_t_buff + ofm_t_sz) + (ofm_t_sz & 0x1)  ); // scalar vec mac result

    // =============== Inter-Tile =========================    
    	for (; ti<ifm->ch; ti+=parE->Tn){
    		 // load weight tile
    		_fetch_weights_tile_RCMN_ext(weights, pWeight_t_buff, /*to=*/0, ti, parE->Tm, /*t_n=*/1);
    		for (; row<ofm->h; row+=parE->Tr){
    			for (; col<ofm->w; col+=parE->Tc){
                    // tiles in the middle
                    if(iteration==parP->preSz ){
                    	_save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pOFM_t_buff - ofm_t_sz, idx.r, idx.c, idx.m, parE->Tm, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
                        iteration=0;  //reset the
                        pOFM_t_buff = pWeight_t_buff + weight_t_sz * parP->preSz + (weight_t_sz & 0x1);
                    }
                	// load ifm tile *
                	_fetch_ifm_tile_RCN_ext(ifm, pIFM_t_buff, ifm_t_h, ifm_t_w, row, col, ti, parE->Tn);

                   // =============== Intra-Tile =========================
                    _intra_tile_conv_dw(parE,weights,ifm_t_w,ti,pIFM_t_buff,pWeight_t_buff,pOFM_t_buff,pVMAC_result, parP->preSz, parE->lpOdr);
                    iteration++;
                    pOFM_t_buff +=ofm_t_sz;
                   	idx.r = row;
                    idx.c = col;
                    idx.m = ti;
                    idx.n = ti;

                }col=0;
            }row=0;
        }ti=0;
    // for last S tiles
    if(iteration==parP->preSz ){
    	_save_ofm_tile_RCN_ext(ofm, /*buffer_idx=*/0, pOFM_t_buff - ofm_t_sz, idx.r, idx.c, idx.m, parE->Tm, parE->Tr, parE->Tc, parP->preSz, parE->lpOdr);
        iteration=0;  //reset the
    }
}
