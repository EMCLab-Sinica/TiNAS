#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>

#include "extfram.h"

// 2MB
const uint32_t NVM_SIZE = 2*1024*1024;
static uint8_t *fram = NULL;

int initSPI() {
    int fd = open("nvm.bin", O_RDWR);
    fram = mmap(NULL, NVM_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    return 0;
}

void SPI_READ(SPI_ADDR* A,uint8_t *dst, unsigned long len ) {
    memcpy(dst, fram + A->L, len);
}

void SPI_WRITE(SPI_ADDR* A, const uint8_t *src, unsigned long len ) {
    memcpy(fram + A->L, src, len);
}

void eraseFRAM() {
    memset(fram, 0, NVM_SIZE);
}
