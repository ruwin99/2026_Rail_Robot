#include "rgb_pwm.h"
#include "main.h"

static volatile uint32_t pending_rgb;
static uint32_t active_rgb;
static uint8_t plane;
static uint8_t remaining;

static void write_channels(uint32_t on)
{
  uint32_t high = on ^ (RGB_PWM_ACTIVE_LOW ? 56U : 0U);
  /* Only PB3..5: preserve every other GPIOB output. */
  GPIOB->BSRR = high | ((56U ^ high) << 16U);
}

void RGB_Init(void)
{
  pending_rgb = active_rgb = 0U;
  plane = remaining = 0U;
  write_channels(0U);
}

void RGB_Set(uint8_t red, uint8_t green, uint8_t blue)
{
  pending_rgb = (uint32_t)red | ((uint32_t)green << 8U) | ((uint32_t)blue << 16U);
}

void RGB_ForceRed(void)
{
  /* Fatal path only, with interrupts disabled: no timer/clock assumptions. */
  write_channels(32U);
}

void RGB_ForceOff(void)
{
  write_channels(0U);
}

void RGB_Tick50us(void)
{
  if (remaining == 0U)
  {
    uint32_t bit = 1U << plane;
    uint32_t on = 0U;
    if (plane == 0U) active_rgb = pending_rgb;
    if (active_rgb & bit) on |= 32U;
    if (active_rgb & (bit << 8U)) on |= 16U;
    if (active_rgb & (bit << 16U)) on |= 8U;
    write_channels(on);
    remaining = (uint8_t)bit;
    plane = (uint8_t)((plane + 1U) & 7U);
  }
  --remaining;
}
