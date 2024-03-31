/*
 * util_macros.h
 */

#ifndef UTILS_UTIL_MACROS_H_
#define UTILS_UTIL_MACROS_H_

#ifdef __MSP430__
#include <driverlib.h>
#endif

#include <stdlib.h>
#include <stdbool.h>

#define PROACTIVE_SHUTDOWN 1

// bit manipulation
#define BSET(num,pos) num |= (1 << pos) // set bit
#define BCLR(num,pos) num &= ~(1<< pos) // clear bit
#define BCHK(num,pos) num & (1<<pos)    // check bit


#ifdef __MSP430__

#define _STOP() \
while(true) __no_operation();

#else

#define _STOP() \
while(true);

#endif

#ifdef __MSP430__

static inline void _SHUTDOWN() {
    GPIO_setOutputHighOnPin( GPIO_PORT_P5, GPIO_PIN0 );
    //P3OUT |= GPIO_PIN7;
#if ENABLE_FRAM_COUNTERS
    //counters.shutdownCount++;
#endif
}

static inline void _ERROR_SHUTDOWN() {
    GPIO_setOutputHighOnPin( GPIO_PORT_P3, GPIO_PIN7 );

#if PROACTIVE_SHUTDOWN
    GPIO_setOutputHighOnPin( GPIO_PORT_P5, GPIO_PIN0 );
    _STOP();
#endif

#if ENABLE_FRAM_COUNTERS
    //counters.errorShutdownCount++;
#endif

#if !PROACTIVE_SHUTDOWN
    WDTCTL = 0;
#endif
}

#else

static inline void _SHUTDOWN() {
    exit(1);
}

static inline void _ERROR_SHUTDOWN() {
    exit(2);
}

#endif

static inline void _SHUTDOWN_AFTER_TILE() {
#if PROACTIVE_SHUTDOWN
    _SHUTDOWN();
    _STOP();
#endif
}

#define CPU_F                                   ((double)16000000)

#define _delay_us(A)\
  __delay_cycles( (uint32_t) ( (double)(CPU_F) *((A)/1000000.0) + 0.5))

#define _delay_ms(A)\
  __delay_cycles( (uint32_t) ( (double)(CPU_F)*((A)/1000.0) + 0.5))

#define _delay_s(A)\
  __delay_cycles( (uint32_t) ( (double)(CPU_F)*((A)/1.0) + 0.5))




#endif /* UTILS_UTIL_MACROS_H_ */
