/*
 * sleep_timer.c
 * got to sleep for n cycles (LPM)
 */


#include "sleep_timer.h"


void _lpm_sleep(uint16_t delayCycles){
    uint16_t i;

    for(i=0;i<delayCycles*32;i++){
        _config_sleep_timer(delayCycles);
        __no_operation();
    }
}

void _config_sleep_timer(uint16_t delayCycles){
#ifdef __MSP430__
    // This will sleep for 62500 / 16MHz = 1/32 seconds. I cannot use a larger value as TA0CCR0 is a 16-bit register
    TA0CCR0 = 62500;  // set up terminal count @TODO: generalize
    TA0CTL = TASSEL__SMCLK + ID__8 + TIMER_A_UP_MODE; // configure and start timer, ID__8 => frequency divider 8
    TA0CCTL0 = CCIE;
    __bis_SR_register(LPM0_bits + GIE);
    __no_operation();
#else
    SCB->SCR |= SCB_SCR_SLEEPONEXIT_Msk;    // Enable sleep on exit from ISR

    // Ensures SLEEPONEXIT takes effect immediately
    __DSB();

    // Enable global interrupt
    __enable_irq();

    NVIC->ISER[0] = 1 << ((TA0_0_IRQn) & 31);

    TIMER_A0->CCTL[0] &= ~TIMER_A_CCTLN_CCIFG;
    TIMER_A0->CCTL[0] = TIMER_A_CCTLN_CCIE; // TACCR0 interrupt enabled
    TIMER_A0->CCR[0] = 50000;
    TIMER_A0->CTL = TIMER_A_CTL_SSEL__SMCLK | // SMCLK, UP mode
            TIMER_A_CTL_MC__UP;

    __sleep();
#endif



}
#ifdef __MSP430__
#pragma vector=TIMER0_A0_VECTOR
__interrupt void Timer_A0(void)
{
    TA0CTL = MC__STOP ;
    __bic_SR_register_on_exit(LPM0_bits);
}
#else
void TA0_0_IRQHandler(void) {
    // Clear the compare interrupt flag
    TIMER_A0->CCTL[0] &= ~TIMER_A_CCTLN_CCIFG;

    Interrupt_disableSleepOnIsrExit();
}

#endif


