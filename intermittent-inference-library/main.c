/*
 * main.c 
 */

#include <msp430.h>

#include "driverlib.h"
#include "DSPLib.h"

/*  local imports */

// general utilities
#include "utils/myuart.h"
#include "utils/voltage_monitor.h"
#include "utils/helper_functions.h"
#include "utils/util_macros.h"
#include "utils/extfram.h"

// cnn library
#include "cnn/cnn.h"
#include "cnn/cnn_utils.h" // for counters


// globals
unsigned int FreqLevel = 8;
int uartsetup=0;
volatile uint32_t cycleCount;

// func prototypes
void gpio_setup(void);
int boardSetup(void);
void VMon_ISR_Callback(void);








void VMon_ISR_Callback(void){
#if ENABLE_VMON
    switch (__even_in_range(CEIV, CEIV_CERDYIFG)) {
        case CEIV_NONE:        break;
        case CEIV_CEIFG:

			break;
        case CEIV_CEIIFG:
        	CEIV &= ~(CEIV_CEIIFG);

			VMon_disable();
			__disable_interrupt();
			_DBGUART("-- VM_I2 --\r\n");
			_SHUTDOWN(); _STOP();

            break;
        case CEIV_CERDYIFG:    break;
    }
#endif
}


void gpio_setup(void){
    /* set direction */
    GPIO_setAsOutputPin( GPIO_PORT_P1, GPIO_PIN0 ); // led
    GPIO_setAsOutputPin( GPIO_PORT_P1, GPIO_PIN1 ); // led
    GPIO_setAsOutputPin( GPIO_PORT_P4, GPIO_PIN1 ); // for indicating inference done
    GPIO_setAsOutputPin( GPIO_PORT_P5, GPIO_PIN0 ); // EHM shutdown
    GPIO_setAsOutputPin( GPIO_PORT_P5, GPIO_PIN1 ); // EXT_FLAG : to signal the FT232H to complete exp run
    /* set initial value */
    // leds
    GPIO_setOutputLowOnPin( GPIO_PORT_P1, GPIO_PIN0 );  // led
    GPIO_setOutputLowOnPin( GPIO_PORT_P1, GPIO_PIN1 ); // led
    GPIO_setOutputLowOnPin( GPIO_PORT_P4, GPIO_PIN1 );  // for indicating inference done
    GPIO_setOutputLowOnPin( GPIO_PORT_P5, GPIO_PIN0 );  // EHM shutdown
    GPIO_setOutputLowOnPin( GPIO_PORT_P5, GPIO_PIN1 );  // EXT_FLAG : to signal the FT232H to complete exp run

}


int boardSetup(){
    WDTCTL = WDTPW + WDTHOLD;    /* disable watchdog timer  */

    // initialize GPIO

    PM5CTL0 &= ~LOCKLPM5;       // Disable the GPIO power-on default high-impedance mode
//    P8DIR=0xfd;P8REN=GPIO_PIN1;
    P1DIR=0xff;P1OUT=0x00;
    P2DIR=0xff;P2OUT=0x00;
    P3DIR=0xff;P3OUT=0x00;
    P4DIR=0xff;P4OUT=0x00;
    P5DIR=0xff;P5OUT=0x00;
    P6DIR=0xff;P6OUT=0x00;
    P7DIR=0xff;P7OUT=0x00;
    P8DIR=0xff;P8OUT=0x00;
    PADIR=0xff;PAOUT=0x00;
    PBDIR=0xff;PBOUT=0x00;
    PCDIR=0xff;PCOUT=0x00;
    PDDIR=0xff;PDOUT=0x00;

    /* setup UART */
    uartsetup=0;
    setFrequency(8);
    CS_initClockSignal( CS_SMCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1 );
    uartinit();

    /* clock setup */

    // Configure one FRAM waitstate as required by the device datasheet for MCLK
    // operation beyond 8MHz _before_ configuring the clock system.
    FRCTL0 = FRCTLPW | NWAITS_1;
    // Clock System Setup
    CSCTL0_H = CSKEY_H;                     // Unlock CS registers
    CSCTL1 = DCOFSEL_0;                     // Set DCO to 1MHz
    // Set SMCLK = MCLK = DCO, ACLK = VLOCLK
    CSCTL2 = SELA__VLOCLK | SELS__DCOCLK | SELM__DCOCLK;
    // Per Device Errata set divider to 4 before changing frequency to
    // prevent out of spec operation from overshoot transient
    CSCTL3 = DIVA__4 | DIVS__4 | DIVM__4;   // Set all corresponding clk sources to divide by 4 for errata
    CSCTL1 = DCOFSEL_4 | DCORSEL;           // Set DCO to 16MHz
    // Delay by ~10us to let DCO settle. 60 cycles = 20 cycles buffer + (10us / (1/4MHz))
    __delay_cycles(60);
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;   // Set all dividers to 1 for 16MHz operation
    CSCTL0_H = 0;

    PMM_unlockLPM5();
    _enable_interrupt();

    // setup gpio
    gpio_setup();

    return 0;
};

#if ENABLE_FRAM_COUNTERS
static void report_counters() {
    // report counters after CNN is complete
    _DBGUART("mainCount=%d\r\n", counters.mainCount);
    _DBGUART("bootCount=%d\r\n", counters.bootCount);
    _DBGUART("fpCount=%d\r\n", counters.fpCount);
    _DBGUART("pulseCount=%d\r\n", counters.pulseCount);
    _DBGUART("shutdownCount=%d\r\n", counters.shutdownCount);
    _DBGUART("errroShutdownCount=%d\r\n", counters.errorShutdownCount);
    _DBGUART("tileCount=%d\r\n", counters.tileCount);
}
#endif

int main() {
    /* mandatory init stuff */
    WDTCTL = WDTPW | WDTHOLD;     //Stop WDT
    PM5CTL0 &= ~LOCKLPM5; // Disable the GPIO power-on default high-impedance mode to activate previously configured port settings


    boardSetup();

#if ENABLE_FRAM_COUNTERS
    if (counters.fpCount) {
        report_counters();
        _STOP();
    }

    counters.mainCount++;
#endif

    //__delay_cycles(400000);
    __delay_cycles(80000);
    if(0 != initSPI()){
    	_DBGUART("\r\nFRAM not ready!\r\n");

        // external FRAM failed to initialize - reset
        volatile uint16_t counter = 1000;
        // waiting some time seems to increase the possibility
        // of a successful FRAM initialization on next boot
        while (counter--);

        _ERROR_SHUTDOWN();
    }


#if ENABLE_VMON
    VMon_init(&VMon_ISR_Callback); // initiate Voltage Monitor (VMon)
    VMon_enable();  // start VMon
#endif
//    while(1);
    msp_lea_init();

    // run CNN model
#if ENABLE_FRAM_COUNTERS
    counters.bootCount++;
#endif
    // GPIO_setOutputHighOnPin( GPIO_PORT_P5, GPIO_PIN1 );  // notify entering CNN_run
    while(1){
    	CNN_run();
    }

    _DBGUART("------ FINISHED !\r\n");
    __delay_cycles(100);
    // GPIO_setOutputHighOnPin( GPIO_PORT_P5, GPIO_PIN1 );

	__no_operation();
	return 0;
}




