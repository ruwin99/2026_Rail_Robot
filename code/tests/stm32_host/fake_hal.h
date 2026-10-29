#ifndef HOST_FAKE_HAL_H
#define HOST_FAKE_HAL_H
#include "stm32f1xx_hal.h"
extern TIM_HandleTypeDef htim1, htim2, htim3, htim4;
extern UART_HandleTypeDef huart2, huart3;
extern uint32_t host_tick, host_primask, host_commit_delay, host_compare_writes;
extern uint32_t host_nonzero_writes;
extern uint32_t host_motor_accessory_nonzero_writes;
extern int host_stop_on_counter, host_fault_on_tick, host_receive_calls;
extern int host_mission_state_at_uart_error;
extern int host_legacy_link;
extern HAL_StatusTypeDef host_receive_status;
extern char host_output[32768];
void host_reset(void);
void host_feed(const char *text);
void host_line(const char *text);
void host_drain(void);
void host_time(uint32_t tick);
void host_inject_uart_error_after_next_irq_restore(void);
void host_inject_uart_error_after_irq_restores(unsigned restores_to_skip);
#endif
