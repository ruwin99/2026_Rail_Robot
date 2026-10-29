#ifndef UART_PROTOCOL_H
#define UART_PROTOCOL_H

#include "main.h"

/* RR2 capability/version negotiated by the mission bridge. */
#define UART_RR2_PROTOCOL_VERSION "2"
#define UART_RR2_MISSION_CAPABILITY "MISSION"

typedef enum
{
  UART_RR2_CAP_NONE = 0,
  UART_RR2_CAP_MISSION,
  UART_RR2_CAP_ENCODER,
  UART_RR2_CAP_DC,
  UART_RR2_CAP_PUMP,
  UART_RR2_CAP_ALL,
  UART_RR2_CAP_SENSOR
} UartRr2Capability;

typedef enum
{
  UART_COMMAND_NONE,
  UART_COMMAND_START,
  UART_COMMAND_MISSION_PREPARE,
  UART_COMMAND_MISSION_CANCEL_PREPARE,
  UART_COMMAND_CAPTURE_OK,
  UART_COMMAND_ACTUATOR_TEST_START,
  UART_COMMAND_ACTUATOR_TEST_STOP,
  UART_COMMAND_FRONT_CLEANER_PWM_12_5,
  UART_COMMAND_FRONT_CLEANER_PWM_25,
  UART_COMMAND_FRONT_CLEANER_OFF,
  UART_COMMAND_FRONT_PUMP_ON,
  UART_COMMAND_FRONT_PUMP_OFF,
  UART_COMMAND_SIDE_CLEANER_PWM_12_5,
  UART_COMMAND_SIDE_CLEANER_PWM_25,
  UART_COMMAND_SIDE_CLEANER_OFF,
  UART_COMMAND_SIDE_PUMP_ON,
  UART_COMMAND_SIDE_PUMP_OFF,
  /* legacy 통신 감시용 keepalive. 명령함에 넣지 않고 수신 시각만 갱신한다. */
  UART_COMMAND_LINK_OK,
  UART_COMMAND_COUNT
} UartCommand;

typedef enum
{
  UART_MESSAGE_STARTED,
  UART_MESSAGE_ACTUATOR_TEST_READY,
  UART_MESSAGE_CAMERA_CAPTURE,
  UART_MESSAGE_RETURN_START,
  UART_MESSAGE_REALTIME_START,
  UART_MESSAGE_RESCAN_RETURN_START,
  UART_MESSAGE_RESCAN_START,
  UART_MESSAGE_RESCAN_DONE,
  UART_MESSAGE_DONE,
  UART_MESSAGE_COUNT
} UartMessage;

void UART_Init(void);
void UART_Update(void);
void UART_OnRx(UART_HandleTypeDef *huart);
void UART_OnError(UART_HandleTypeDef *huart);
UartCommand UART_TakePendingCommand(void);
/* 초음파 hold 중 main이 처리할 ACTUATOR_TEST_STOP과 네 OFF만 꺼낸다. */
UartCommand UART_TakeHeldStopCommand(void);
void UART_ClearPending(void);
void UART_ObstacleBarrier(void);
void UART_QueueFailSafeAllOff(void);
/* 응답 송신은 main 문맥에서만 호출한다. */
void UART_SendMessage(UartMessage message);
void UART_SendText(const char *text);
int UART_V2Active(void);
UartRr2Capability UART_RR2GetCapability(void);
/* Discrete events/status are copied into bounded queues and sent from UART_Update. */
int UART_QueueEvent(const char *event_code, const char *payload);
void UART_QueueStatus(const char *status_key, const char *values);

/* RR2 debug adapter; implementations route through the existing DebugConsole safety handlers. */
int Debug_RR2Begin(const char *session, UartRr2Capability capability);
int Debug_RR2Set(uint32_t sequence, int direction, uint32_t percent_basis_points,
                 uint32_t actuator_mask);
int Debug_RR2Led(uint32_t sequence, const char *mode,
                 uint32_t red, uint32_t green, uint32_t blue);
void Debug_RR2Stop(void);
void Debug_RR2Heartbeat(void);

#endif
