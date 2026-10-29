/* 애플리케이션 호출 순서와 HAL 콜백 전달만 묶는다. 기능별 상태는 각 모듈이 소유한다.
 * Init/Update는 main 문맥, UART/엔코더 콜백은 기존 HAL ISR 문맥에서 실행한다. */
#include "robot_app.h"
#include "robot_board.h"
#include "robot_config.h"
#include "drive_control.h"
#include "actuators.h"
#include "mission.h"
#include "uart_protocol.h"
#include "debug_console.h"
#include "fault_stop.h"
#include "rgb_pwm.h"
#include "ultrasonic.h"

static uint8_t rgb_mode;
static uint32_t rgb_epoch;
static uint8_t mission_session_was_active;

static int mission_session_active(void)
{
  return UART_V2Active() && UART_RR2GetCapability() == UART_RR2_CAP_MISSION;
}

static void update_status_rgb(void)
{
  MissionState state = Mission_GetState();
  uint32_t now = HAL_GetTick();
  uint8_t mode;
  /* LED 모드: 0 파란 숨쉬기(1 s, 모드 사이 정지·센서 세션), 1 초록 숨쉬기(1 s, 임무 진행),
   * 2 빨간 깜빡임(FAULT·장애물), 3 꺼짐(rail_mission 연결 전 IDLE),
   * 4 빠른 파란 깜빡임(PREPARE: START 직전 준비 중), 5 느린 파란 숨쉬기(2 s, 연결된 대기 IDLE).
   * 래치된 FAULT는 어떤 상태에서도 가리지 않는다. */
  if (state == MISSION_FAULT) mode = 2U;
  else if (state == MISSION_IDLE && !mission_session_active()) mode = 3U;
  else if (!Debug_IsSensorSession() && Ultrasonic_HasObstacle()) mode = 2U;
  else if (Mission_IsPrepared()) mode = 4U;
  else if (Debug_IsSensorSession() || Mission_IsTransitionPending()) mode = 0U;
  else if (state == MISSION_IDLE) mode = 5U;
  else mode = 1U;
  uint32_t period = mode == 5U ? 2000U : 1000U;
  uint32_t half = period / 2U;
  uint32_t elapsed;
  uint8_t level;
  uint8_t red, green, blue;
  if (mode != rgb_mode) { rgb_mode = mode; rgb_epoch = now; }
  elapsed = (uint32_t)(now - rgb_epoch) % period;
  if (mode == 2U) RGB_Set(elapsed < 500U ? 255U : 0U, 0U, 0U);
  /* 200 ms 켜짐 / 200 ms 꺼짐. 대기(느린 숨쉬기)와 한눈에 구별되게 한다. */
  else if (mode == 4U) RGB_Set(0U, 0U, (elapsed % 400U) < 200U ? 255U : 0U);
  /* 센서 전용 DBG 세션의 수동 색상도 fault/장애물 빨강 경고를 가리지 않는다. */
  else if (mode != 3U && Debug_GetLedOverride(&red, &green, &blue)) RGB_Set(red, green, blue);
  else
  {
    level = (uint8_t)(((elapsed <= half ? elapsed : period - elapsed) * 255U) / half);
    RGB_Set(0U, mode == 1U ? level : 0U, (mode == 0U || mode == 5U) ? level : 0U);
  }
}

void Robot_Init(void)
{
  Mission_Init();
  Debug_Init();
  Actuators_Init();
  Drive_Init();
  UART_Init();
  Ultrasonic_Init();
  mission_session_was_active = 0U;
  rgb_mode = 255U;
  update_status_rgb();
}

void Robot_Update(void)
{
  /* main 루프가 도는 동안에만 워치독 먹이를 준다. 루프가 멈추면 약 0.5 s 뒤 MCU가 리셋된다.
   * FAULT가 래치된 상태에서도 루프는 돌므로 FAULT 표시는 그대로 유지된다. */
  Watchdog_Feed();
  uint8_t mission_session_active_now = (uint8_t)mission_session_active();
  /* Ignore range history collected while the mission bridge was absent. */
  if (mission_session_active_now && !mission_session_was_active)
    Ultrasonic_ResetWindow();
  mission_session_was_active = mission_session_active_now;
  /* UART 명령은 매 루프 초반부터 확인하고, 그 루프에서 상태기계·진단이 만든
   * 응답/이벤트는 마지막 UART_Update에서 비운다. 모든 HAL TX는 이 main 문맥에만 둔다. */
  Drive_MarkMainAlive();
  Ultrasonic_Update();
  Actuators_Update();
  Debug_Update();
  Mission_Update();
  update_status_rgb();
  UART_Update();
}

void Robot_OnUartRx(UART_HandleTypeDef *huart) { UART_OnRx(huart); }
void Robot_OnUartError(UART_HandleTypeDef *huart) { UART_OnError(huart); }
void Robot_OnEncoderUpdate(TIM_HandleTypeDef *htim) { Mission_OnEncoderUpdate(htim); }

void Robot_FatalStop(void)
{
  Mission_LatchFault();
  Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
  if (htim2.Instance == TIM2) Motor_Stop();
  CleaningActuators_StopAll();
}

void Robot_EmergencyStop(void)
{
  uint32_t mask = __get_PRIMASK();
  __disable_irq();
  Robot_FatalStop();
  __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
  UART_ClearPending();
  Actuators_ClearPending();
  Debug_Deactivate();
  if (!mask) __enable_irq();
  update_status_rgb();
}
