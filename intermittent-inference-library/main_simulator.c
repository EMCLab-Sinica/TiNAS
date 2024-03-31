#include "cnn/cnn.h"
#include "utils/extfram.h"

int main() {
    initSPI();

    while(1){
        CNN_run();
    }
}
