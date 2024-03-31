/*
 * cnn_utils.c
 * Misc utility functions
 * (data management, debugging etc.)
 */


#include "msp430.h"
#include "driverlib.h"

#include "../utils/myuart.h"
#include "../utils/util_macros.h"
#include "../utils/extfram.h"

#include "cnn_buffers.h"
#include "cnn_utils.h"
#include "cnn_common.h"
#include <string.h>
#include <math.h>



#if ENABLE_FRAM_COUNTERS
#pragma PERSISTENT(counters)
struct Counters counters = {
    .bootCount = 0,
    .fpCount = 0,
    .pulseCount = 0,
    .tileCount = 0,
    .shutdownCount = 0,
    .errorShutdownCount = 0,
};
#endif



/*****************************************************************************************
 * MATRIX/VECTOR DATA MANAGEMENT RELATED
 *****************************************************************************************/
 uint32_t offset2D(uint16_t V2,uint16_t V1,uint16_t D1){
    return (uint32_t) (uint32_t)(V2 * D1) + (uint32_t)(V1);
}

 uint32_t offset3D(uint16_t V3,uint16_t V2,uint16_t V1,uint16_t D2,uint16_t D1){
    return (uint32_t)(V3 * D2 * D1) + (uint32_t)(V2 * D1) + (uint32_t)(V1);
}
 uint32_t offset4D(uint16_t V4,uint16_t V3,uint16_t V2,uint16_t V1,uint16_t D3,uint16_t D2,uint16_t D1){
    return (uint32_t)(V4 * D3 * D2 * D1) + (uint32_t)(V3 * D2 * D1) + (uint32_t)(V2 * D1) + (uint32_t)(V1);
}

// get element from Input Feature Map matrix (3D)
_q15 getIFMElement_ext(Mat_t* m, uint16_t h, uint16_t w, uint16_t c){
    _q15 temp;
    SPI_ADDR A;
    A.L = (uint32_t)m->data + offset3D(h,w,c,m->w,m->ch) * Q15_SIZE;
    SPI_READ(&A,(uint8_t*)&temp,2);
    return temp;
}

// get element from Weight matrix (4D)
_q15 getWeightElement_ext(Mat_t* m, uint16_t h, uint16_t w, uint16_t n, uint16_t c){
    _q15 temp;
    SPI_ADDR A;
    A.L = (uint32_t)m->data + offset4D(h,w,n,c,m->w,m->n,m->ch) * Q15_SIZE;
    SPI_READ(&A,(uint8_t*)&temp,2);
    return temp;
    // return (_q15) *(m->data + (m->h * m->w * m->ch * n) + (m->h * m->w * c) + (m->w * h) + m->w);
}

// get element from Weight matrix (4D)
_q15 getMatrixElement_ext(Mat_t* m, uint16_t h, uint16_t w, uint16_t n, uint16_t c){
    _q15 temp;
    SPI_ADDR A;
    A.L = (uint32_t)m->data + offset4D(h,w,n,c,m->w,m->n,m->ch) * Q15_SIZE;
    SPI_READ(&A,(uint8_t*)&temp,2);
    return temp;

    // return (_q15) *(m->data + (m->h * m->w * m->ch * n) + (m->h * m->w * c) + (m->w * h) + m->w);
}



#ifndef EXTFRAM
// get element from Input Feature Map matrix (3D)
_q15 getIFMElement(Mat_t* m, uint16_t c, uint16_t h, uint16_t w){
    return (_q15) *(m->data + (m->h * m->w * c) + (m->w * h) + w);
}

// get element from Weight matrix (4D)
_q15 getWeightElement(Mat_t* m, uint16_t n, uint16_t c, uint16_t h, uint16_t w){
    return (_q15) *(m->data + (m->h * m->w * m->ch * n) + (m->h * m->w * c) + (m->w * h) + m->w);
}

// get element from Weight matrix (4D)
_q15 getMatrixElement(Mat_t* m, uint16_t n, uint16_t c, uint16_t h, uint16_t w){
    return (_q15) *(m->data + (m->h * m->w * m->ch * n) + (m->h * m->w * c) + (m->w * h) + m->w);
}
#endif

// print out feature map in table format
static void dumpFeatureMap(Mat_t *mat){
    uint16_t c, h, w;
    _DBGUART("\t[\r\n");
    for (c = 0; c < mat->ch; c++){
        _DBGUART("\t\t[\r\n");
        for (h = 0; h < mat->h; h++){
            _DBGUART("\t\t\t[");
            for (w = 0; w < mat->w; w++){
#ifdef EXTFRAM
                _DBGUART("%d, ", getIFMElement_ext(mat, h, w, c));
#else
                _DBGUART("%d, ", getIFMElement(mat, h, w, c));
#endif
                if (w && (w % 16 == 0)) { _DBGUART("\r\n"); }
            }
            _DBGUART("],\r\n");
        }
        _DBGUART("\r\n\t\t],\r\n");
    }
    _DBGUART("\r\n\t],\r\n");
}

void CNN_printInput(Mat_t *ifm){
    dumpFeatureMap(ifm);
}

void CNN_printResult(Mat_t *ofm){
    dumpFeatureMap(ofm);
}

// print out matrix in table format
void dumpMatrix(Mat_t *mat) {
    uint16_t n, c, h, w;
    _DBGUART("[\r\n");
    for (n = 0; n < mat->n; n++){
        _DBGUART("\t[\r\n");
        for (c = 0; c < mat->ch; c++){
            _DBGUART("\t\t[\r\n");
            for (h = 0; h < mat->h; h++){
                _DBGUART("\t\t\t[");
                for (w = 0; w < mat->w; w++){
#ifdef EXTFRAM
                    _DBGUART("%d, ", getMatrixElement_ext(mat, h, w, n, c));
#else
                    _DBGUART("%d, ", getMatrixElement(mat, h, w, n, c));
#endif
                    if (w && (w % 16 == 0)) { _DBGUART("\r\n"); }
                }
                _DBGUART("],\r\n");
            }
            _DBGUART("\r\n\t\t],\r\n");
        }
        _DBGUART("\r\n\t],\r\n");
    }
    _DBGUART("\r\n];\n");
}


// print out vector/array in python format
void printQ15Vector(_q15 *buff, uint16_t n){
    uint16_t i;
    _DBGUART("[");
    for(i=0;i<n;i++){
        //_DBGUART("%d [0x%x] \r\n", *(buff + i), (buff+i));
        _DBGUART("%d, ", *(buff + i));
        if (i && (i % 16 == 0)) { _DBGUART("\r\n"); }
    }
    _DBGUART("]\r\n");
}

void printQ15Matrix(_q15 *buff, uint16_t h, uint16_t w){
    uint16_t i, j;
    _DBGUART("[\r\n");

    for(i=0;i<h;i++){
        _DBGUART("\t[");
        for(j=0;j<w;j++){
            _DBGUART("%d, ", *(buff + (w*i) + j));
        }
        _DBGUART("],\r\n");
    }
    _DBGUART("]\r\n");
}


// set/clear all values in buffer
void setQ15VectorBuff(_q15 *buff, uint16_t n, _q15 val){
    uint16_t i;
    for(i=0;i<n;i++){
        *(buff+i) = val;
    }
}


void printSumOfVector(_q15 *buff, uint16_t size){
    uint16_t i;
    _iq31 sum=0;
    for(i=0; i<size; i++){
        sum += *(buff + i);
    }
    if(sum!=0){
        _DBGUART("nop\r\n");
    }
}

/****************************************************************************************
 * DATA TYPE CONVERSIONS
 ****************************************************************************************/
_q15 iq31_to_q15(_iq31 *iq31_val_ptr) {
    return *(_q15*)iq31_val_ptr;
}

/*****************************************************************************************
 * DMA RELATED DATA TRANSFER
 *****************************************************************************************/
void memcpy_dma_ext(uint32_t nvm_addr, const void* vm_addr, size_t n, size_t dstSize, uint16_t RW) {

#if CNN_BENCHMARKING_DMA
    /* Benchmark related */
    volatile uint32_t cycleCount = 0;
#endif

#if CNN_BENCHMARKING_DMA
    msp_benchmarkStart(TIMER_A1_BASE, 1); //  set res 1
#endif

    if (dstSize < n){
        _DBGUART("memcpy_dma_ext:: Error - dst size is too small %d, %d\r\n", (uint16_t)n, (uint16_t)dstSize); _STOP();
    }
    SPI_ADDR A;
    A.L = nvm_addr;
    if(RW==MEMCPY_READ){
        SPI_READ( &A, (uint8_t*)vm_addr, n );
    }
    else{
        SPI_WRITE( &A, (uint8_t*)vm_addr, n );
    }

#if CNN_BENCHMARKING_DMA
    cycleCount += msp_benchmarkStop(TIMER_A1_BASE);
    _DBGUART("memcpy_dma_ext (bench):: [lid=%d] [tid=%l] [sz=%d] [cc=%l] [dir=%d]\r\n",
             CNN_Benchmark_Get_LID(), CNN_Benchmark_Get_TID(),
             (uint16_t)n, (uint32_t)cycleCount, (uint16_t)RW);
    cycleCount = 0;
#endif

}

void memcpy_dma(void* dest, const void* src, size_t n, size_t dstSize, uint16_t srcDir, uint16_t dstDir) {

#ifdef __MSP430__

#if CNN_BENCHMARKING_DMA
    /* Benchmark related */
    volatile uint32_t cycleCount = 0;
    uint16_t sMemType = addr_loc_type((uint32_t)(src));
    uint16_t dMemType = addr_loc_type((uint32_t)(dest));
#endif

#if CNN_BENCHMARKING_DMA
    msp_benchmarkStart(TIMER_A1_BASE, 1); //  set res 1
#endif

    if (dstSize < n){
        _DBGUART("memcpy_dma:: Error - dst size is too small %d, %d\r\n", (uint16_t)n, (uint16_t)dstSize); _STOP();
    }

    DMA_initParam dma_params = {
        .channelSelect = DMA_CH,
        .transferModeSelect = DMA_TRANSFER_BLOCK
    };
    DMA_init(&dma_params);
    DMA_setSrcAddress(DMA_CH, (uint32_t)(src), srcDir);
    DMA_setDstAddress(DMA_CH, (uint32_t)(dest), dstDir);

    /* transfer size is in words (2 bytes) */
    DMA_setTransferSize(DMA_CH, (n) >> 1);
    //_DBGUART("DMA_getTransferSize = %d \r\n", DMA_getTransferSize(DMA_CH));

    DMA_enableTransfers(DMA_CH);
    DMA_startTransfer(DMA_CH);

#if CNN_BENCHMARKING_DMA
    cycleCount += msp_benchmarkStop(TIMER_A1_BASE);
    _DBGUART("memcpy_dma (bench):: [lid=%d] [tid=%l] [sz=%d] [cc=%l] [dir=%d->%d]\r\n",
             CNN_Benchmark_Get_LID(), CNN_Benchmark_Get_TID(),
             (uint16_t)n, (uint32_t)cycleCount, (uint16_t)sMemType, (uint16_t)dMemType);
    cycleCount = 0;
#endif

#else /* __MSP430__ */

    memcpy(dest, src, n);

#endif

}


// =============== BEGIN FETCH IFM _fetch_ifm_tile_RCN_ext COST MODEL ===============
// DMA Blk size: Tn
// N (DMA operations): Th * Tw
// OVH (addressing): 4*MUL + 8*ADD
// ================ END _fetch_ifm_tile_RCN_ext COST MODEL ================

// =============== BEGIN FETCH WEIGHTS _fetch_weights_tile_RCMN_ext COST MODEL ===============
// DMA Blk size: Tn
// N (DMA operations): Kh * Kw * Tm 
// OVH (addressing): 9*MUL + 9*ADD
// ================ END _fetch_weights_tile_RCMN_ext COST MODEL ================

// =============== BEGIN FETCH OFM _fetch_ofm_tile_RCN_ext COST MODEL ===============
// DMA Blk size: Tm
// N (DMA operations): Tr * Tc
// OVH (addressing): 5*MUL + 8*ADD (IFM reuse), 4*MUL + 8*ADD (WEIGHT, OFM reuse)
// ================ END _fetch_ofm_tile_RCN_ext COST MODEL ================

// =============== BEGIN WRITE OFM _save_ofm_tile_RCN_ext COST MODEL ===============
// DMA Blk size: Tm if (WEIGHT, OFM reuse), S*Tm if (IFM reuse)
// N (DMA operations): [Tr*Tc] if (IFM, OFM reuse), [S*Tr*Tc] if (WEIGHT reuse)
// OVH (addressing): 5*MUL + 8*ADD (OFM reuse), 9*MUL + 8*ADD (WEIGHT reuse), 8*MUL + 8*ADD (IFM reuse)
// ================ END _save_ofm_tile_RCN_ext COST MODEL ================

/***************************************************************************
 OFF CHIP TILE FETCHING/SAVING - into 1d array
 RCN (HWC) layout
****************************************************************************/
// Assume storage layout order in main memory : RCN (HWC)
// Transfers a tile of the IFM to LEAMEM (starting from the given LEAMEM buffer location)
void _fetch_ifm_tile_RCN_ext(Mat_t* ifm, _q15 *pBuffLoc, uint16_t ifm_t_h, uint16_t ifm_t_w, uint16_t row, uint16_t col, uint16_t ti, uint16_t t_n){
#ifdef EXTFRAM
    uint16_t r=0, c=0;
    _q15 *pDst;
    uint32_t pSrc;

    for (r=0; r<ifm_t_h; r++){
        for (c=0; c<ifm_t_w; c++){
            // fetch whole N dim (Tn elements) <-- 1 DMA (Tn block size)
            pSrc = ifm->data + ((ifm->w * ifm->ch * (row+r)) + (ifm->ch * (col+c)) + ti) * Q15_SIZE;
            pDst = pBuffLoc + (((r * ifm_t_w) + c) * t_n);
            // void memcpy_dma_ext(uint32_t nvm_addr, const void* vm_addr, size_t n, size_t dstSize, uint16_t RW)
            memcpy_dma_ext(pSrc, pDst, t_n*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE, 0);
            __no_operation();
        }
    }
#else
    uint16_t r=0, c=0;
    _q15 *pDst, *pSrc;

    for (r=0; r<ifm_t_h; r++){
        for (c=0; c<ifm_t_w; c++){
            // fetch whole N dim (Tn elements) <-- 1 DMA (Tn block size)
            pSrc = ifm->data + (ifm->w * ifm->ch * (row+r)) + (ifm->ch * (col+c)) + ti;
            pDst = pBuffLoc + (((r * ifm_t_w) + c) * t_n);
            memcpy_dma(pDst, pSrc, t_n*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE, DMA_DIR_INC, DMA_DIR_INC);
        }
    }
#endif
}



// Assume storage layout order in main memory : RCMN (HWNC)
// Transfers a tile of the Weights to LEAMEM (starting from the given LEAMEM buffer location)
void _fetch_weights_tile_RCMN_ext(Mat_t* weights, _q15 *pBuffLoc, uint16_t to, uint16_t ti, uint16_t t_m, uint16_t t_n){
#ifdef EXTFRAM
    uint16_t r=0, c=0, too=0;
    _q15 *pDst;
    uint32_t pSrc;

    for (r=0; r<weights->h; r++){
        for (c=0; c<weights->w; c++){
            for (too=0; too<t_m; too++){
//            	DBG_W[k1][k2][ch][n]
                // fetch whole N dim (Tn elements) <-- 1 DMA (Tn block size)
                pSrc = weights->data + sizeof(_q15) * ((r * (weights->w*weights->n*weights->ch)) + (c * (weights->n*weights->ch)) + ((too+to)*weights->ch) + ti);
                pDst = pBuffLoc + (((t_m*weights->w*r) + (t_m*c) + (too))*t_n);
                // void memcpy_dma_ext(uint32_t nvm_addr, const void* vm_addr, size_t n, size_t dstSize, uint16_t RW)
                memcpy_dma_ext(pSrc, pDst, t_n*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE, 0);
//                __no_operation();
            }
        }
    }
#else
    uint16_t r=0, c=0, too=0;
    _q15 *pDst, *pSrc;

    for (r=0; r<weights->h; r++){
        for (c=0; c<weights->w; c++){
            for (too=0; too<t_m; too++){
                // fetch whole N dim (Tn elements) <-- 1 DMA (Tn block size)
                pSrc = weights->data + (r * (weights->w*weights->n*weights->ch)) + (c * (weights->n*weights->ch)) + ((too+to)*weights->ch) + ti;
                pDst = pBuffLoc + (((t_m*weights->w*r) + (t_m*c) + (too))*t_n);
                memcpy_dma(pDst, pSrc, t_n*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE, DMA_DIR_INC, DMA_DIR_INC);
            }
        }
    }
#endif
}

// assume storage layout order in main memory : RCN (HWC)
// Transfers a tile of the OFM to LEAMEM (starting from the given LEAMEM buffer location)
void _fetch_ofm_tile_RCN_ext(Mat_t* ofm, _q15 *pBuffLoc, uint16_t row, uint16_t col, uint16_t to, uint16_t t_m, uint16_t t_r, uint16_t t_c, uint16_t preSz, LOOPORDER_t lpOdr){
#ifdef EXTFRAM
    uint16_t r=0, c=0;
    _q15 *pDst;
    uint32_t pSrc;
    if(lpOdr == IFM_ORIENTED){
     	for (r=0; r<t_r; r++){
     		for (c=0; c<t_c; c++){
     			// fetch whole M dim (Tm elements) <-- 1 DMA (Tm block size)
     			pSrc = ofm->data + ((ofm->w * ofm->ch * (row+r)) + (ofm->ch * (col+c)) + to )*sizeof(_q15);
     			pDst = pBuffLoc + (((r * t_c) + c) * t_m * preSz) ;
     			memcpy_dma_ext(pSrc, pDst, t_m*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE, 0);
     		}
     	}
    }else{
    	for (r=0; r<t_r; r++){
    		for (c=0; c<t_c; c++){
    			// fetch whole M dim (Tm elements) <-- 1 DMA (Tm block size)
    			pSrc = ofm->data + ((ofm->w * ofm->ch * (row+r)) + (ofm->ch * (col+c)) + to )*sizeof(_q15);
    			pDst = pBuffLoc + (((r * t_c) + c) * t_m) ;
    			memcpy_dma_ext(pSrc, pDst, t_m*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE, 0);
    		}
    	}

    }

#else
    uint16_t r=0, c=0;
    _q15 *pDst, *pSrc;

    for (r=0; r<t_r; r++){
        for (c=0; c<t_c; c++){
            // fetch whole M dim (Tm elements) <-- 1 DMA (Tm block size)
            pSrc = ofm->data + (ofm->w * ofm->ch * (row+r)) + (ofm->ch * (col+c)) + to;
            pDst = pBuffLoc + (((r * t_c) + c) * t_m);
            memcpy_dma(pDst, pSrc, t_m*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE, DMA_DIR_INC, DMA_DIR_INC);
        }
    }
#endif
}

void _save_ofm_tile_RCN_ext(Mat_t* ofm, uint16_t buffer_idx, _q15 *pBuffLoc, uint16_t row, uint16_t col, uint16_t to, uint16_t t_m, uint16_t t_r, uint16_t t_c, uint16_t preSz, LOOPORDER_t lpOdr){

    uint16_t r=0, c=0 , sz=0;
    _q15 *pSrc;
    uint32_t pDst;
    if(lpOdr == OFM_ORIENTED){ // inner inter-tile loop N
        for (r=0; r<t_r; r++){
            for (c=0; c<t_c; c++){
                pSrc = pBuffLoc + r * t_c * t_m + c * t_m  ;
                pDst = ofm->data + (  (ofm->w * ofm->ch * (row+r)) + (ofm->ch * (col+c)) + to ) * sizeof(_q15);
                memcpy_dma_ext(pDst, pSrc, t_m*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE,1);
            }
        }
    }else if(lpOdr == WEIGHT_ORIENTED) { // inner inter-tile loop is R,C
    	uint16_t tile_offset = t_r * t_c * t_m ;
    	for(sz=1; sz <= preSz ; sz++ ){
            for (r=0; r<t_r; r++){
                for (c=0; c<t_c; c++){

                    pSrc = pBuffLoc +  r * t_c * t_m + c * t_m -  tile_offset * (preSz - sz);
                    pDst = ofm->data + ( (ofm->w * ofm->ch * (row+r)) + (ofm->ch * (col+c)) + to -  ((t_r * t_c * ofm->ch) * (preSz - sz)) ) * sizeof(_q15);
                    memcpy_dma_ext(pDst, pSrc, t_m*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE,1);
                }
            }
    	}
    }else{ // IFM ORIENTED // inner inter-tile loop is M
//    	uint16_t tile_offset = t_r * t_c * t_m ;
    	for (r=0; r<t_r; r++){
			for (c=0; c<t_c; c++){
				pSrc = pBuffLoc +  r * t_c * t_m * preSz + c * t_m * preSz  -  t_m *  (preSz - 1);
//				pSrc = pBuffLoc + r * t_c * t_m + c * t_m  ;
				pDst = ofm->data + (  (ofm->w * ofm->ch * (row+r)) + (ofm->ch * (col+c)) + to) * sizeof(_q15);
				memcpy_dma_ext(pDst, pSrc, t_m *preSz*Q15_SIZE, LEA_MEM_SIZE*Q15_SIZE,1);
			}
    	}
    }

#if ENABLE_FRAM_COUNTERS
    counters.tileCount++;
    // _DBGUART("tileCount=%d\r\n", counters.tileCount);
#endif
}














#define ADDR_SRAM_TOP        0x1C00
#define ADDR_SRAM_BOT        ADDR_SRAM_TOP + 0x2000

uint16_t addr_loc_type(uint32_t addr){
    uint16_t result;
    if ((addr >= ADDR_SRAM_TOP) && (addr < ADDR_SRAM_BOT)){
        result = 1;
    }else{
        result = 2;
    }
    return result;
}


/*
#ifdef __MSP430__

#pragma vector=DMA_VECTOR
__interrupt void DMA_ISR(void)
{
    switch(__even_in_range(DMAIV,16))
    {
        case 0: break;
        case 2: break; // DMA0IFG = DMA Channel 0
        case 4: break; // DMA1IFG = DMA Channel 1
        case 6: break; // DMA2IFG = DMA Channel 2
        case 8: break; // DMA3IFG = DMA Channel 3
        case 10: break; // DMA4IFG = DMA Channel 4
        case 12: break; // DMA5IFG = DMA Channel 5
        case 14: break; // DMA6IFG = DMA Channel 6
        case 16: break; // DMA7IFG = DMA Channel 7
        default: break;
    }
}

#endif
*/

void my_vsqrt_q15(int16_t* pIn, int16_t* pOut, uint32_t blockSize) {
#if defined(__MSP430__) || defined(SIMULATOR)
    const float Q15_DIVISOR = 32768.0f;
    for (uint32_t idx = 0; idx < blockSize; idx++) {
        float real_val = (1.0f * pIn[idx]) / Q15_DIVISOR;
        pOut[idx] = (int16_t)(sqrtf(real_val) * Q15_DIVISOR);
    }
#else
    // somehow arm_vsqrt_q15 is defined in headers but there is no implementation
    for (uint32_t idx = 0; idx < blockSize; idx++) {
        arm_sqrt_q15(*pIn, pOut);
        pIn++;
        pOut++;
    }
#endif
}

void my_div_q15(const int16_t *pSrcA, const int16_t *pSrcB, int16_t *pDst, uint32_t blockSize) {
    // XXX: use LEA?
    for (uint16_t idx = 0; idx < blockSize; idx++) {
        if(pSrcB[idx] == 0) {
            // Overflow
            pDst[idx] = 32767;
        } else {
            // https://sestevenson.wordpress.com/2010/09/20/fixed-point-division-2/
            int32_t tmp = ((int32_t)pSrcA[idx] << 15) / (int32_t)pSrcB[idx];
            tmp = MIN_VAL(32767, MAX_VAL(-32768, tmp));
            pDst[idx] = (int16_t)tmp;
        }
    }
}

/*******************************************************************
 * Getters/Setters
 *******************************************************************/

/* related to buffers */
_q15* CNN_GetLEAMemoryLocation(){
    return pLEAMemory;
}
_q15* CNN_GetLayerTempBuff1Location(){
    return pLayerTempBuff1;
}
_q15* CNN_GetLayerTempBuff2Location(){
    return pLayerTempBuff2;
}
_q15* CNN_GetSRAMBuffLocation(){
    return pSRAMBuff;
}
_q15* CNN_GetOFMBuff1Location(){
    return pOFMBuff1;
}
_q15* CNN_GetOFMBuff2Location(){
    return pOFMBuff2;
}
