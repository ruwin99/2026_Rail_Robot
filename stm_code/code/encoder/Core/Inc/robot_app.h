#ifndef ROBOT_APP_H
#define ROBOT_APP_H

#include "main.h"

/* 모든 MX_* 초기화 뒤 한 번 Init, main 루프마다 Update를 호출한다. */
void Robot_Init(void);
void Robot_Update(void);
/* HAL 콜백 전달부: UART 수신/오류와 TIM3 구간 경계를 ISR 문맥 그대로 처리한다. */
void Robot_OnUartRx(UART_HandleTypeDef *huart);
void Robot_OnUartError(UART_HandleTypeDef *huart);
void Robot_OnEncoderUpdate(TIM_HandleTypeDef *htim);
/* Error_Handler의 IRQ 차단 뒤 호출한다. 초기화 도중에도 안전하게 OFF한다. */
void Robot_FatalStop(void);
/* Main-context sensor/RR2 emergency: stop all outputs, latch FAULT; no auto-clear.
 * Call with normal interrupts running to retain the red blink. */
void Robot_EmergencyStop(void);

#endif
