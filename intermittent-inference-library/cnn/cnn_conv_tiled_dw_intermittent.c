/*
 * cnn_conv_tiled_std.c
 * Depthwise Tiled Convolution
 */

#include "cnn.h"

#include "cnn_conv_tiled_dw.h"

#include "cnn_buffer_sizes.h"
#include "cnn_utils.h"
#include "../utils/util_macros.h"
#include "../utils/myuart.h"
#include "../utils/extfram.h"


static void _CNN_Intermittent_LayerConvDW_Tiled_OFM(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);
static void _CNN_Intermittent_LayerConvDW_Tiled_IFM(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);
static void _CNN_Intermittent_LayerConvDW_Tiled_WEI(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf);


/*******************************************************************************************************************
 Tiled Convolution using LEA vector MAC command
 exec design params : Tr, Tc, Tm, Tn, loop order, preservation batch size
 Tn MACs performed in one command
 [UNDER INTERMITTENT POWER]
*******************************************************************************************************************/
void CNN_Intermittent_LayerConv_Tiled_Depthwise(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){
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
			_CNN_Intermittent_LayerConvDW_Tiled_OFM( lid, weights, bias, ifm, ofm, parE, parP,idxBuf);break;
		case IFM_ORIENTED:
			if( ((ofm->ch / parE->Tm) % parP->preSz) !=0 ){_DBGUART("CNN_LayerConv_Tiled_IFM:: Error - wrong preSz sizes %d\r\n",parP->preSz);_STOP();}
			_CNN_Intermittent_LayerConvDW_Tiled_IFM( lid, weights, bias, ifm, ofm, parE, parP,idxBuf);break;
		case WEIGHT_ORIENTED:
			if( ( ((ofm->w / parE->Tc)*(ofm->h / parE->Tr)) % parP->preSz) !=0 ){_DBGUART("CNN_LayerConv_Tiled_WEI:: Error - wrong preSz sizes %d\r\n",parP->preSz);_STOP();}
			_CNN_Intermittent_LayerConvDW_Tiled_WEI( lid, weights, bias, ifm, ofm, parE, parP,idxBuf);break;
		case NONE_ORIENTED:
			break;
	}
}

// ============== BEGIN _CNN_Intermittent_LayerConvDW_Tiled_OFM COST MODEL ==============
// Number of power cycles: (R/Tr * C/Tc * N/Tn / preSz)
// Fetch tile idx: SPI_READ(10)
// Fetch IFM: preSz * _fetch_ifm_tile_RCN_ext
// Fetch weights: preSz * _fetch_weights_tile_RCMN_ext
// Fetch OFM: 0
// Addressing: 9*MUL + 8*ADD + 2*SUB + (preSz-1)*ADD
// Computation: preSz * _intra_tile_conv_dw
// Tile output backup: _save_ofm_tile_RCN_ext
// Tile idx backup: SPI_WRITE(10)
// =============== END _CNN_Intermittent_LayerConvDW_Tiled_OFM COST MODEL ===============

void _CNN_Intermittent_LayerConvDW_Tiled_OFM(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){
    SPI_ADDR A;
    ConvTileIndices idx;
    A.L = LOC_LOOP_INDICES;
    SPI_READ(&A,(uint8_t*)&idx,sizeof(ConvTileIndices)); //fetch loop indices
    uint16_t row=idx.r, col=idx.c, ti=idx.n;
    uint16_t iteration = 0;  // number of completed tiles

    // report progress
    if (row < ofm->h && col < ofm->w && ti < ifm->ch) {
        uint32_t num_tiles_N = ROUND(ifm->ch, parE->Tn),
                 num_tiles_CN = ROUND(ofm->w, parE->Tc) * num_tiles_N;
        uint32_t tile_idx = row / parE->Tr * num_tiles_CN + col / parE->Tc * num_tiles_N + ti / parE->Tn;
        _DBGUART("P,%d,%d\r\n", lid, tile_idx);
        // _DBGUART("_,%d,%d,%d\r\n", row, col, ti);
    }

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
                    	idx.r = row;
                    	idx.c = col;
                    	idx.m = ti;
                    	idx.n = ti;
                    	A.L = LOC_LOOP_INDICES;
                    	SPI_WRITE(&A,(uint8_t*)&idx, sizeof(ConvTileIndices));
                        iteration=0;  //reset the
                        _SHUTDOWN_AFTER_TILE();
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
    	idx.r = ofm->h;
    	idx.c = ofm->w;
    	idx.m = ofm->ch;
    	idx.n = ifm->ch;
    	A.L = LOC_LOOP_INDICES;
    	SPI_WRITE(&A,(uint8_t*)&idx, sizeof(ConvTileIndices));
        iteration=0;  //reset the
        _SHUTDOWN_AFTER_TILE();
    }
    CNN_ClearFootprints_LayerConv_Tiled_Depthwise(0);
}

// ============== BEGIN _CNN_Intermittent_LayerConvDW_Tiled_IFM COST MODEL ==============
// Number of power cycles: (R/Tr * C/Tc * N/Tn / preSz)
// Fetch tile idx: SPI_READ(10)
// Fetch IFM: preSz * _fetch_ifm_tile_RCN_ext
// Fetch weights: preSz * _fetch_weights_tile_RCMN_ext
// Fetch OFM: 0
// Addressing: 11*MUL + 8*ADD + 2*SUB + (preSz-1)*ADD
// Computation: preSz * _intra_tile_conv_dw
// Tile output backup: _save_ofm_tile_RCN_ext
// Tile idx backup: SPI_WRITE(10)
// =============== END _CNN_Intermittent_LayerConvDW_Tiled_IFM COST MODEL ===============

void _CNN_Intermittent_LayerConvDW_Tiled_IFM(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){
    SPI_ADDR A;
    ConvTileIndices idx;
    A.L = LOC_LOOP_INDICES;
    SPI_READ(&A,(uint8_t*)&idx,sizeof(ConvTileIndices)); //fetch loop indices
    uint16_t row=idx.r, col=idx.c, ti=idx.n;

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
                       	idx.r = row;
                        idx.c = col;
                        idx.m = ti;
                        idx.n = ti;
                    	A.L = LOC_LOOP_INDICES;
                    	SPI_WRITE(&A,(uint8_t*)&idx, sizeof(ConvTileIndices));//

                        iteration=0;
                        pOFM_t_buff = pWeight_t_buff + weight_t_sz * parP->preSz + (weight_t_sz & 0x1);

                        _SHUTDOWN_AFTER_TILE();
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
    	idx.r = ofm->h;
    	idx.c = ofm->w;
    	idx.m = ofm->ch;
    	idx.n = ifm->ch;

    	A.L = LOC_LOOP_INDICES;
    	SPI_WRITE(&A,(uint8_t*)&idx, sizeof(ConvTileIndices));
        iteration=0;  //reset the
        _SHUTDOWN_AFTER_TILE();
    }
    CNN_ClearFootprints_LayerConv_Tiled_Depthwise(idxBuf);
}

// ============== BEGIN _CNN_Intermittent_LayerConvDW_Tiled_WEI COST MODEL ==============
// Number of power cycles: (N/Tn * R/Tr * C/Tc / preSz)
// Fetch tile idx: SPI_READ(10)
// Fetch IFM: preSz * _fetch_ifm_tile_RCN_ext
// Fetch weights: _fetch_weights_tile_RCMN_ext
// Fetch OFM: 0
// Addressing: 11*MUL + 8*ADD + 2*SUB + (preSz-1)*ADD
// Computation: preSz * _intra_tile_conv_dw
// Tile output backup: _save_ofm_tile_RCN_ext
// Tile idx backup: SPI_WRITE(10)
// =============== END _CNN_Intermittent_LayerConvDW_Tiled_WEI COST MODEL ===============

void _CNN_Intermittent_LayerConvDW_Tiled_WEI(uint16_t lid, Mat_t* weights, Mat_t* bias, Mat_t* ifm, Mat_t* ofm, ExeParams_t *parE, PreParams_t *parP, uint8_t idxBuf){
    SPI_ADDR A;
    ConvTileIndices idx;
    A.L = LOC_LOOP_INDICES;
    SPI_READ(&A,(uint8_t*)&idx,sizeof(ConvTileIndices)); //fetch loop indices
    uint16_t row=idx.r, col=idx.c, ti=idx.n;
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
                       	idx.r = row;
                        idx.c = col;
                        idx.m = ti;
                        idx.n = ti;
                    	A.L =  LOC_LOOP_INDICES;
                    	SPI_WRITE(&A,(uint8_t*)&idx, sizeof(ConvTileIndices));

                        iteration=0;  //reset the
                        pOFM_t_buff = pWeight_t_buff + weight_t_sz * parP->preSz + (weight_t_sz & 0x1);

                        _SHUTDOWN_AFTER_TILE();
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
    	idx.r = ofm->h;
    	idx.c = ofm->w;
    	idx.m = ofm->ch;
    	idx.n = ifm->ch;
    	A.L = LOC_LOOP_INDICES;
    	SPI_WRITE(&A,(uint8_t*)&idx, sizeof(ConvTileIndices));
        iteration=0;  //reset the
        _SHUTDOWN_AFTER_TILE();
    }
    CNN_ClearFootprints_LayerConv_Tiled_Depthwise(idxBuf);
}

/*************************************************************************
 * FOOTPRINTING
 *************************************************************************/
void CNN_ClearFootprints_LayerConv_Tiled_Depthwise(uint8_t idxBuf){
    SPI_ADDR A;
    A.L = LOC_LOOP_INDICES;
    ConvTileIndices idx={0,0,0,0};
    SPI_WRITE(&A,(uint8_t*)&idx,sizeof(ConvTileIndices));

}

void CNN_PrintFootprints_LayerConv_Tiled_Depthwise(){
    //_DBGUART("FP(conv-matmul):: %d, %d, %d, %d \r\n", FP_LayerConv.n, FP_LayerConv.ch, FP_LayerConv.chloop_task, FP_LayerConv.nloop_task);
}
