#include "stm32f1xx_hal.h"
#include "fake_hal.h"
#include "robot_app.h"
#include "mission.h"
#include "ultrasonic.h"
#include "drive_control.h"
#include "uart_protocol.h"
#include <assert.h>
#include <string.h>
TIM_TypeDef host_tim1, host_tim2, host_tim3, host_tim4;
IWDG_TypeDef host_iwdg;
DBGMCU_TypeDef host_dbgmcu;
TIM_HandleTypeDef htim1 = {TIM1}, htim2 = {TIM2}, htim3 = {TIM3}, htim4 = {TIM4};
UART_HandleTypeDef huart2 = {USART2}, huart3 = {USART3};
GPIO_TypeDef host_gpioa, host_gpiob, host_gpioc;
DWT_Type host_dwt;
CoreDebug_Type host_coredebug;
uint32_t SystemCoreClock = 72000000U;
uint32_t host_tick, host_primask, host_commit_delay, host_compare_writes;
uint32_t host_nonzero_writes, host_motor_accessory_nonzero_writes;
int host_stop_on_counter, host_fault_on_tick, host_receive_calls;
int host_mission_state_at_uart_error;
/* 1: behave like a live legacy Jetson and send LINK_OK while legacy driving time advances. */
int host_legacy_link = 1;
HAL_StatusTypeDef host_receive_status;
char host_output[32768];
static uint8_t *receive_byte;
static int host_uart_error_irq_restore_countdown = -1;

void host_reset(void) {
  memset(&host_tim1, 0, sizeof(host_tim1));
  memset(&host_tim2, 0, sizeof(host_tim2));
  memset(&host_tim3, 0, sizeof(host_tim3));
  memset(&host_tim4, 0, sizeof(host_tim4));
  memset(&host_gpioa, 0, sizeof(host_gpioa));
  memset(&host_gpiob, 0, sizeof(host_gpiob));
  memset(&host_gpioc, 0, sizeof(host_gpioc));
  memset(&host_dwt, 0, sizeof(host_dwt));
  memset(&host_iwdg, 0, sizeof(host_iwdg));
  memset(&host_dbgmcu, 0, sizeof(host_dbgmcu));
  memset(&host_coredebug, 0, sizeof(host_coredebug));
  host_tim1.ARR = host_tim2.ARR = 3599;
  host_tim3.ARR = host_tim4.ARR = 600;
  host_tick = host_primask = host_commit_delay = host_compare_writes = host_nonzero_writes = 0;
  host_motor_accessory_nonzero_writes = 0;
  host_stop_on_counter = host_fault_on_tick = host_receive_calls = 0;
  host_mission_state_at_uart_error = -1;
  host_legacy_link = 1;
  host_uart_error_irq_restore_countdown = -1;
  host_output[0] = '\0'; receive_byte = NULL; host_receive_status = HAL_OK;
  huart2.ErrorCode = huart3.ErrorCode = HAL_UART_ERROR_NONE;
  huart2.RxState = huart3.RxState = HAL_UART_STATE_READY;
  Robot_Init();
  /* Baseline scenarios start after a full clear 20-sample window. */
  for (unsigned n=0;n<20;++n) {
    while (!(GPIOB->ODR & GPIO_PIN_8)) Ultrasonic_Tick50us();
    Ultrasonic_Tick50us(); Ultrasonic_OnEcho(1);
    DWT->CYCCNT+=2000U*72U; Ultrasonic_OnEcho(0); Ultrasonic_Update();
  }
  DWT->CYCCNT=0;

}
void host_time(uint32_t tick) {
  MissionState state;
  host_tick = tick; DWT->CYCCNT = tick * 72000U;
  /* The legacy text path now faults after 1 s of silence while driving. A live Jetson keeps
   * sending after START (also while a capture ACK is pending); other sessions are untouched. */
  state = Mission_GetState();
  if (host_legacy_link && receive_byte != NULL && !UART_V2Active() &&
      (state == INITIAL_SCAN || state == WAIT_CAPTURE_ACK || state == RETURNING ||
       state == REALTIME_CLEANING ||
       state == RESCAN_RETURN || state == RESCAN || state == FINAL_RETURN))
    host_feed("LINK_OK\r\n");
}
void host_inject_uart_error_after_next_irq_restore(void) {
  host_uart_error_irq_restore_countdown = 0;
}
void host_inject_uart_error_after_irq_restores(unsigned restores_to_skip) {
  host_uart_error_irq_restore_countdown = (int)restores_to_skip;
}
void host_drain(void) { Robot_Update(); Robot_Update(); }
void host_feed(const char *text) {
  while (*text) {
    assert(receive_byte != NULL);
    *receive_byte = (uint8_t)*text++;
    Robot_OnUartRx(&huart2);
  }
}
void host_line(const char *text) { host_feed(text); host_drain(); }
uint32_t HAL_GetTick(void) {
  if (host_fault_on_tick) {
    host_fault_on_tick = 0;
    Robot_OnUartError(&huart2);
  }
  return host_tick;
}
uint32_t __get_PRIMASK(void) { return host_primask; }
void __disable_irq(void) {
  host_primask = 1;
  if (host_commit_delay) { DWT->CYCCNT += host_commit_delay; host_commit_delay = 0; }
}
void __enable_irq(void) {
  host_primask = 0;
  if (host_uart_error_irq_restore_countdown >= 0) {
    if (host_uart_error_irq_restore_countdown == 0) {
      host_uart_error_irq_restore_countdown = -1;
      Robot_OnUartError(&huart2);
      host_mission_state_at_uart_error = (int)Mission_GetState();
    } else {
      --host_uart_error_irq_restore_countdown;
    }
  }
}
static uint32_t *compare_register(TIM_HandleTypeDef *timer, uint32_t channel) {
  switch (channel) {
    case TIM_CHANNEL_1: return &timer->Instance->CCR1;
    case TIM_CHANNEL_2: return &timer->Instance->CCR2;
    case TIM_CHANNEL_3: return &timer->Instance->CCR3;
    case TIM_CHANNEL_4: return &timer->Instance->CCR4;
    default: assert(!"invalid compare channel"); return NULL;
  }
}
void host_set_compare(TIM_HandleTypeDef *timer, uint32_t channel, uint32_t value) {
  *compare_register(timer, channel) = value; ++host_compare_writes;
  if (value != 0) {
    ++host_nonzero_writes;
    if (timer == &htim1 || timer == &htim2)
      ++host_motor_accessory_nonzero_writes;
  }
}
uint32_t host_get_compare(TIM_HandleTypeDef *timer, uint32_t channel) {
  return *compare_register(timer, channel);
}
uint32_t host_get_counter(TIM_HandleTypeDef *timer) {
  if (host_stop_on_counter) { host_stop_on_counter = 0; Motor_Stop(); }
  return timer->Instance->CNT;
}
HAL_StatusTypeDef HAL_TIM_PWM_Start(TIM_HandleTypeDef *timer, uint32_t channel) {
  assert(host_get_compare(timer, channel) == 0); return HAL_OK;
}
HAL_StatusTypeDef HAL_TIM_Encoder_Start(TIM_HandleTypeDef *timer, uint32_t channel) {
  (void)timer; assert(channel == TIM_CHANNEL_ALL); return HAL_OK;
}
HAL_StatusTypeDef HAL_TIM_Encoder_Start_IT(TIM_HandleTypeDef *timer, uint32_t channel) {
  timer->Instance->DIER |= TIM_IT_UPDATE; return HAL_TIM_Encoder_Start(timer, channel);
}
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *uart, const uint8_t *data,
                                  uint16_t length, uint32_t timeout) {
  size_t size = strlen(host_output);
  assert(uart == &huart2); (void)timeout;
  assert(size + length < sizeof(host_output));
  memcpy(host_output + size, data, length); host_output[size + length] = '\0'; return HAL_OK;
}
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *uart, uint8_t *data, uint16_t length) {
  assert(uart == &huart2 && length == 1); ++host_receive_calls;
  if (host_receive_status == HAL_OK) receive_byte = data;
  return host_receive_status;
}
HAL_StatusTypeDef HAL_UART_AbortReceive(UART_HandleTypeDef *uart) {
  uart->RxState = HAL_UART_STATE_READY; return HAL_OK;
}
void HAL_GPIO_WritePin(GPIO_TypeDef *port, uint16_t pins, GPIO_PinState state) {
  if (state == GPIO_PIN_SET) port->ODR |= pins; else port->ODR &= ~pins;
}
void HAL_NVIC_ClearPendingIRQ(int irq) { (void)irq; }
void Error_Handler(void) { Robot_FatalStop(); assert(!"unexpected HAL failure"); }
