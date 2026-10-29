#ifndef RGB_PWM_H
#define RGB_PWM_H

#include <stdint.h>

/* Default assumes common cathode; confirm the actual SZH-EK058 module. */
#ifndef RGB_PWM_ACTIVE_LOW
#define RGB_PWM_ACTIVE_LOW 0
#endif
#if RGB_PWM_ACTIVE_LOW != 0 && RGB_PWM_ACTIVE_LOW != 1
#error RGB_PWM_ACTIVE_LOW must be 0 or 1
#endif

/* PB3=Blue, PB4=Green, PB5=Red. Init before enabling the TIM2 IRQ. */
void RGB_Init(void);
/* Brightness 0..255; one atomic word, applied at the next BAM frame.
 * Main-context status effects own this API during normal operation. */
void RGB_Set(uint8_t red, uint8_t green, uint8_t blue);
/* Fatal path with IRQs disabled only. GPIO clock/output must be initialized. */
void RGB_ForceRed(void);
void RGB_ForceOff(void);
/* Eight bit planes lasting 1,2,4,...128 ticks: 255*50us = 12.75ms. */
void RGB_Tick50us(void);

#endif
