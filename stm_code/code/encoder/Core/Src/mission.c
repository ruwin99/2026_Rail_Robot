#include "ultrasonic.h"
/* 임무 상태, 구간 완료 이벤트/계수와 지연 전이를 이 파일에서 소유한다.
 * Update는 main 전용이며 TIM3 ISR은 구간 종료 시 즉시 정지한다. ISR에서 UART를 송신하지 않는다. */
#include "mission.h"
#include "robot_board.h"
#include "robot_config.h"
#include "drive_control.h"
#include "actuators.h"
#include "uart_protocol.h"
#include "debug_console.h"

typedef struct
{
  volatile int event;
  volatile int limit;
} EncoderContext;

typedef struct
{
  volatile MissionState state;
  volatile uint8_t prepared;
  MissionState capture_resume_state;
  volatile int transition_pending;
  MissionState transition_target;
  uint32_t transition_started_tick;
  uint32_t capture_wait_started_tick;
  /* 이번 임무의 단계별 구간 수와, 다음 START에 쓸 요청값(RR2 PREPARE 인자). 기본은 ENCODER_SECTION_LIMIT. */
  volatile uint8_t section_limit;
  uint8_t requested_section_limit;
} MissionContext;

static EncoderContext encoder;
static MissionContext mission;

void Mission_SetState(MissionState next_state)
{
  /* 물체가 이미 앞에 있어도 구동 권한이 없는 센서 디버깅은 열 수 있다. */
  if (Ultrasonic_IsHeld() && next_state!=MISSION_FAULT && next_state!=MISSION_IDLE &&
      !(next_state==DEBUG_TEST && Debug_IsSensorSession())) return;
  /* 같은 상태 재진입은 무시한다. 촬영 대기는 INITIAL_SCAN/RESCAN에서만 허용하고 복귀할 상태를 별도 저장한다.
   * 모든 실제 전이에서 네 청소 출력을 먼저 끈 뒤 아래 상태별 주행·통지 동작을 수행한다. */
  MissionState previous_state;
  uint32_t primask = __get_PRIMASK();

  /* 예약 만료 직전 UART ISR이 FAULT로 바꿨다면 늦은 전이로 되살리지 않는다.
   * 상태 확정만 짧게 원자화하고, 아래 UART 송신 전에 원래 IRQ 상태를 복원한다. */
  __disable_irq();
  previous_state = mission.state;
  if ((previous_state == MISSION_FAULT) && (next_state != MISSION_FAULT))
  {
    mission.transition_pending = 0;
    if (primask == 0U) __enable_irq();
    return;
  }

  /* 즉시 상태 변경(정지/오류/시험 포함)은 이전 예약을 취소한다. START나 FAULT 자체에 지연을 붙이지 않는다. */
  mission.transition_pending = 0;
  /* 새 임무 시작·중단·시험 전환에서는 이전 임무 구간의 관성 이월을 쓰지 않는다. */
  if ((next_state == MISSION_IDLE) || (next_state == MISSION_FAULT) ||
      (next_state == DEBUG_TEST) || (next_state == ACTUATOR_TEST) ||
      ((next_state == INITIAL_SCAN) && (previous_state == MISSION_IDLE)))
    Drive_ClearSectionCarry();

  if (previous_state == next_state)
  {
    if (primask == 0U) __enable_irq();
    return;
  }

  if (next_state == WAIT_CAPTURE_ACK)
  {
    if ((previous_state != INITIAL_SCAN) && (previous_state != RESCAN))
    {
      next_state = MISSION_FAULT;
    }
    else
    {
      mission.capture_resume_state = previous_state;
      mission.capture_wait_started_tick = Ultrasonic_OperationalTick();
    }
  }

  mission.state = next_state;
  mission.prepared = 0U;
  /* START(IDLE→INITIAL_SCAN)에서만 요청 구간 수를 이번 임무에 고정한다. 다른 모든 전이에서 요청값은
   * 기본으로 돌아가므로, 시험용 PREPARE 뒤 START 없이 끊겨도 다음 임무에 남지 않는다. */
  if ((next_state == INITIAL_SCAN) && (previous_state == MISSION_IDLE))
    mission.section_limit = mission.requested_section_limit;
  mission.requested_section_limit = ENCODER_SECTION_LIMIT;
  if (primask == 0U) __enable_irq();
  if (previous_state == MISSION_IDLE && next_state != MISSION_IDLE &&
      next_state != MISSION_FAULT &&
      !(next_state == DEBUG_TEST && Debug_IsSensorSession()))
    Ultrasonic_Arm();
  /* Every mission transition starts from a known four-channel OFF state. */
  CleaningActuators_StopAll();

  switch (mission.state)
  {
    case MISSION_IDLE:
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
      encoder.event = 0;
      encoder.limit = 0;
      mission.capture_resume_state = INITIAL_SCAN;
      break;

    case INITIAL_SCAN:
      /* 첫 START 전이에서만 STARTED를 알리고 구간 수를 초기화한다. 촬영 ACK 후 재진입은 카운터를 유지한다.
       * Jetson 서비스의 UART 쓰기 성공과 이 STARTED 이벤트 수신은 서로 다른 확인 단계다. */
      if (previous_state == MISSION_IDLE)
      {
        encoder.event = 0;
        encoder.limit = 0;
        UART_SendMessage(UART_MESSAGE_STARTED);
      }
      Drive_SetDirection(MOTOR_DIRECTION_FORWARD);
      Motor_StartNextEncoderSection(Drive_GetDirection(), MISSION_FORWARD_PWM_DUTY);
      break;

    case WAIT_CAPTURE_ACK:
      /* TIM3 구간 IRQ와 주행 출력을 끈 뒤 CAMERA_CAPTURE를 보낸다. ACK 전에는 다음 구간으로 가지 않는다.
       * RR2 세션에서는 30초를 넘기면 FAULT로 전이한다. legacy는 기존의 정지 대기를 유지한다. */
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      UART_SendMessage(UART_MESSAGE_CAMERA_CAPTURE);
      break;

    case RETURNING:
      encoder.limit = 0;
      Drive_SetDirection(MOTOR_DIRECTION_REVERSE);
      UART_SendMessage(UART_MESSAGE_RETURN_START);
      Motor_StartNextEncoderSection(Drive_GetDirection(), MISSION_REVERSE_PWM_DUTY);
      break;

    case REALTIME_CLEANING:
      /* 복귀 완료 후 실시간 구간을 시작한다. 이벤트 송신 뒤 STM이 바로 주행을 시작하며 별도 ROS 준비 ACK는 없다.
       * 초기 촬영/재검사와 같은 전진 PWM을 쓰고 중간 구간은 ISR에서 정지하지 않는다. */
      encoder.limit = 0;
      Drive_SetDirection(MOTOR_DIRECTION_FORWARD);
      UART_SendMessage(UART_MESSAGE_REALTIME_START);
      Motor_StartNextEncoderSection(Drive_GetDirection(),
                                    MISSION_FORWARD_PWM_DUTY);
      break;

    case DEBUG_TEST:
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      __HAL_TIM_DISABLE_IT(&htim4, TIM_IT_UPDATE);
      Motor_Stop();
      Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
      encoder.event = 0;
      encoder.limit = 0;
      break;

    case ACTUATOR_TEST:
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
      encoder.event = 0;
      encoder.limit = 0;
      UART_SendMessage(UART_MESSAGE_ACTUATOR_TEST_READY);
      break;

    case RESCAN_RETURN:
      encoder.limit = 0;
      Drive_SetDirection(MOTOR_DIRECTION_REVERSE);
      UART_SendMessage(UART_MESSAGE_RESCAN_RETURN_START);
      Motor_StartNextEncoderSection(Drive_GetDirection(), MISSION_REVERSE_PWM_DUTY);
      break;

    case RESCAN:
      if (previous_state == RESCAN_RETURN)
      {
        encoder.limit = 0;
        Drive_SetDirection(MOTOR_DIRECTION_FORWARD);
        UART_SendMessage(UART_MESSAGE_RESCAN_START);
      }
      Motor_StartNextEncoderSection(Drive_GetDirection(), MISSION_FORWARD_PWM_DUTY);
      break;

    case FINAL_RETURN:
      /* 재검사 마지막 촬영 뒤 먼 쪽 끝에서 출발점으로 돌아온다. 재검사 완료는 복귀 출발 때 알리고
       * (Jetson은 RESCAN_DONE 대기 상태로 넘어감), 임무 완료 DONE은 출발점 도착 뒤 MISSION_DONE에서 보낸다.
       * 재검사 복귀처럼 중간 경계에서 멈추지 않는 연속 후진이다. */
      encoder.limit = 0;
      Drive_SetDirection(MOTOR_DIRECTION_REVERSE);
      UART_SendMessage(UART_MESSAGE_RESCAN_DONE);
      Motor_StartNextEncoderSection(Drive_GetDirection(), MISSION_REVERSE_PWM_DUTY);
      break;

    case MISSION_DONE:
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
      UART_SendMessage(UART_MESSAGE_DONE);
      break;

    case MISSION_FAULT:
    default:
      mission.state = MISSION_FAULT;
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
      break;
  }
  /* OFF is confirmed by the state switch before idle/sensor-only operation
   * drops the obstacle interlock. Echo measurements continue for telemetry. */
  if (next_state == MISSION_IDLE ||
      (next_state == DEBUG_TEST && Debug_IsSensorSession()))
    Ultrasonic_Disarm();
}

static void Mission_ScheduleTransition(MissionState next_state)
{
  uint32_t primask = __get_PRIMASK();

  /* 상위 작업 완료 경계에서만 호출한다. 구간별 WAIT_CAPTURE_ACK 왕복에는 정지시간을 추가하지 않는다.
   * HAL_Delay 없이 예약하므로 UART 수신과 main의 장치 시간 제한 검사는 계속 진행된다. */
  __disable_irq();
  if ((mission.transition_pending == 0) &&
      (mission.state != MISSION_FAULT))
  {
    mission.transition_pending = 1;
    mission.transition_target = next_state;
    __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
    Motor_Stop();
    CleaningActuators_StopAll();
    Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
    encoder.event = 0;
    UART_ClearPending();
    Actuators_ClearPending();
    mission.transition_started_tick = Ultrasonic_OperationalTick();
  }
  if (primask == 0U)
  {
    __enable_irq();
  }
}

static void Mission_HandleUartCommand(UartCommand command)
{
  /* DEBUG_TEST는 세션 프로토콜만 구동을 허용하고 legacy ON/START는 버린다. legacy OFF 네 종류만 통과한다.
   * 수신 비트가 있었는데 동작이 없으면 이 상태 게이트와 아래 명령별 전제 조건을 확인한다. */
  if (((mission.state == DEBUG_TEST) ||
       (mission.transition_pending != 0)) &&
      (command != UART_COMMAND_FRONT_CLEANER_OFF) &&
      (command != UART_COMMAND_FRONT_PUMP_OFF) &&
      (command != UART_COMMAND_SIDE_CLEANER_OFF) &&
      (command != UART_COMMAND_SIDE_PUMP_OFF))
  {
    /* Debug is a separate session; legacy ON/START cannot energize it. */
    return;
  }
  switch (command)
  {
    case UART_COMMAND_START:
      if (mission.state == MISSION_IDLE)
      {
        Mission_SetState(INITIAL_SCAN);
      }
      break;

    case UART_COMMAND_MISSION_PREPARE:
      /* legacy 문자열 경로에는 구간 수 인자가 없으므로 항상 기본 구간 수다. */
      if (mission.state == MISSION_IDLE && !Ultrasonic_IsHeld())
      {
        mission.prepared = 1U;
        mission.requested_section_limit = ENCODER_SECTION_LIMIT;
      }
      break;

    case UART_COMMAND_MISSION_CANCEL_PREPARE:
      if (mission.state == MISSION_IDLE)
      {
        mission.prepared = 0U;
        mission.requested_section_limit = ENCODER_SECTION_LIMIT;
      }
      break;

    case UART_COMMAND_CAPTURE_OK:
      /* 촬영을 요청했던 단계와 이미 지난 구간 수를 보고 다음 구간/복귀/완료를 선택한다.
       * limit는 엔코더 구간 도달 때 증가하며 이 ACK가 다시 증가시키지는 않는다. 원본 저장 결과를 MCU가 직접 검사하지 않는다. */
      if (mission.state != WAIT_CAPTURE_ACK)
      {
        break;
      }

      if (mission.capture_resume_state == INITIAL_SCAN)
      {
        if (encoder.limit >= mission.section_limit)
        {
          Mission_ScheduleTransition(RETURNING);
        }
        else
        {
          Mission_SetState(INITIAL_SCAN);
        }
      }
      else if (mission.capture_resume_state == RESCAN)
      {
        if (encoder.limit >= mission.section_limit)
        {
          Mission_ScheduleTransition(FINAL_RETURN);
        }
        else
        {
          Mission_SetState(RESCAN);
        }
      }
      else
      {
        Mission_SetState(MISSION_FAULT);
      }
      break;

    case UART_COMMAND_ACTUATOR_TEST_START:
      if (mission.state == MISSION_IDLE)
      {
        /* Drop any commands sent before the READY handshake. */
        UART_QueueFailSafeAllOff();
        Mission_SetState(ACTUATOR_TEST);
      }
      break;

    case UART_COMMAND_ACTUATOR_TEST_STOP:
      if (mission.state == ACTUATOR_TEST)
      {
        UART_QueueFailSafeAllOff();
        Mission_SetState(MISSION_IDLE);
      }
      break;

    case UART_COMMAND_FRONT_CLEANER_PWM_12_5:
      Actuators_QueueCommand(ACTUATOR_FRONT_CLEANER,
                            ACTUATOR_COMMAND_CLEANER_PWM_12_5);
      break;

    case UART_COMMAND_FRONT_CLEANER_PWM_25:
      Actuators_QueueCommand(ACTUATOR_FRONT_CLEANER,
                            ACTUATOR_COMMAND_CLEANER_PWM_25);
      break;

    case UART_COMMAND_FRONT_CLEANER_OFF:
      Actuators_QueueCommand(ACTUATOR_FRONT_CLEANER, ACTUATOR_COMMAND_OFF);
      break;

    case UART_COMMAND_FRONT_PUMP_ON:
      Actuators_QueueCommand(ACTUATOR_FRONT_PUMP, ACTUATOR_COMMAND_ON);
      break;

    case UART_COMMAND_FRONT_PUMP_OFF:
      Actuators_QueueCommand(ACTUATOR_FRONT_PUMP, ACTUATOR_COMMAND_OFF);
      break;

    case UART_COMMAND_SIDE_CLEANER_PWM_12_5:
      Actuators_QueueCommand(ACTUATOR_SIDE_CLEANER,
                            ACTUATOR_COMMAND_CLEANER_PWM_12_5);
      break;

    case UART_COMMAND_SIDE_CLEANER_PWM_25:
      Actuators_QueueCommand(ACTUATOR_SIDE_CLEANER,
                            ACTUATOR_COMMAND_CLEANER_PWM_25);
      break;

    case UART_COMMAND_SIDE_CLEANER_OFF:
      Actuators_QueueCommand(ACTUATOR_SIDE_CLEANER, ACTUATOR_COMMAND_OFF);
      break;

    case UART_COMMAND_SIDE_PUMP_ON:
      Actuators_QueueCommand(ACTUATOR_SIDE_PUMP, ACTUATOR_COMMAND_ON);
      break;

    case UART_COMMAND_SIDE_PUMP_OFF:
      Actuators_QueueCommand(ACTUATOR_SIDE_PUMP, ACTUATOR_COMMAND_OFF);
      break;

    case UART_COMMAND_NONE:
    case UART_COMMAND_COUNT:
    default:
      break;
  }
}

int Mission_ApplyReliableCommand(int start_command)
{
  /* Only main calls this. An ACK is possible only after the FSM accepts the
   * command, not when USART's byte interrupt happens to receive it. */
  if (Ultrasonic_IsHeld() || mission.transition_pending != 0 ||
      mission.state == MISSION_FAULT)
    return 0;
  if (start_command) {
    if (mission.state != MISSION_IDLE) return 0;
    Mission_HandleUartCommand(UART_COMMAND_START);
    return mission.state == INITIAL_SCAN;
  }
  if (mission.state != WAIT_CAPTURE_ACK) return 0;
  Mission_HandleUartCommand(UART_COMMAND_CAPTURE_OK);
  return mission.state != MISSION_FAULT &&
         (mission.state != WAIT_CAPTURE_ACK || mission.transition_pending != 0);
}

int Mission_ApplyReliablePrepare(int prepare, int sections)
{
  /* sections는 다음 START 임무의 단계별 구간 수(1..ENCODER_SECTION_LIMIT)다. 시험 모드는 2를 보낸다.
   * 기본보다 많은 구간은 받지 않아 계획한 주행 거리를 넘지 않는다. CANCEL은 기본값으로 되돌린다. */
  if ((prepare != 0 && prepare != 1) || mission.state != MISSION_IDLE ||
      mission.transition_pending != 0 || mission.state == MISSION_FAULT ||
      (prepare != 0 && Ultrasonic_IsHeld()) ||
      sections < 1 || sections > ENCODER_SECTION_LIMIT)
    return 0;
  mission.prepared = (uint8_t)prepare;
  mission.requested_section_limit = (uint8_t)(prepare ? sections : ENCODER_SECTION_LIMIT);
  return 1;
}

void Mission_Update(void)
{
  /* 우선 수신 명령을 소비한 뒤 ISR의 encoder.event를 읽어 상태를 바꾼다. ISR은 복잡한 FSM/UART 처리를 미룬다.
   * 촬영/일반 복귀는 여기서 limit를 증가시키지만 연속 주행(REALTIME/RESCAN_RETURN)은 ISR이 이미 증가시킨다. */
  UartCommand command;

  if (Ultrasonic_IsHeld())
  {
    /* hold 중에도 시험 종료·OFF는 main에서 처리한다. START/ACK/ON 비트는 해제 뒤까지 남긴다. */
    while ((command = UART_TakeHeldStopCommand()) != UART_COMMAND_NONE)
    {
      Mission_HandleUartCommand(command);
    }
    return;
  }

  while ((command = UART_TakePendingCommand()) != UART_COMMAND_NONE)
  {
    Mission_HandleUartCommand(command);
  }

  /* 정지 시작부터 1초가 지난 뒤에만 다음 모드로 진입한다. 부호 없는 뺄셈은 tick 순환에도 대응한다.
   * 대기 중 기존 mode를 유지해도 transition_pending 게이트가 모든 청소 ON을 차단한다. */
  if (mission.transition_pending != 0)
  {
    if ((uint32_t)(Ultrasonic_OperationalTick() - mission.transition_started_tick) >=
        MISSION_TRANSITION_PAUSE_MS)
    {
      Mission_SetState(mission.transition_target);
    }
    return;
  }

  switch (mission.state)
  {
    case MISSION_IDLE:
      break;

    case INITIAL_SCAN:
      if (encoder.event != 0)
      {
        encoder.event = 0;
        encoder.limit++;
        Mission_SetState(WAIT_CAPTURE_ACK);
      }
      break;

    case WAIT_CAPTURE_ACK:
      if (UART_V2Active() &&
          (uint32_t)(Ultrasonic_OperationalTick() - mission.capture_wait_started_tick) >= 30000U)
        Mission_SetState(MISSION_FAULT);
      break;

    case RETURNING:
      if (encoder.event != 0)
      {
        encoder.event = 0;
        __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
        Motor_Stop();
        encoder.limit++;

        if (encoder.limit >= mission.section_limit)
        {
          /* 총 4구간 복귀 후 실시간 주행으로 바꾼다. 목표 미달이면 다음 역방향 구간을 다시 시작한다.
           * 각 구간마다 CNT 기준을 재설정하므로 limit와 CNT를 혼동해 전체 이동량으로 해석하지 않는다. */
          Mission_ScheduleTransition(REALTIME_CLEANING);
        }
        else
        {
          Motor_StartNextEncoderSection(Drive_GetDirection(),
                                        MISSION_REVERSE_PWM_DUTY);
        }
      }
      break;

    case REALTIME_CLEANING:
      if (encoder.event != 0)
      {
        encoder.event = 0;

        if (encoder.limit >= mission.section_limit)
        {
          __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
          Motor_Stop();
          Mission_ScheduleTransition(RESCAN_RETURN);
        }
      }
      break;

    case DEBUG_TEST:
      /* Debug_Update owns bounded drive commands; capture IRQ stays disabled. */
      break;

    case ACTUATOR_TEST:
      /* This service state must never energize the drive motor. */
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
      encoder.event = 0;
      break;

    case RESCAN_RETURN:
      if (encoder.event != 0)
      {
        encoder.event = 0;

        if (encoder.limit >= mission.section_limit)
        {
          __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
          Motor_Stop();
          Mission_ScheduleTransition(RESCAN);
        }
      }
      break;

    case RESCAN:
      if (encoder.event != 0)
      {
        encoder.event = 0;
        encoder.limit++;
        Mission_SetState(WAIT_CAPTURE_ACK);
      }
      break;

    case FINAL_RETURN:
      /* ISR이 연속 후진 구간을 세고 마지막 경계에서 정지했다. 1초 정지 뒤 임무를 마친다. */
      if (encoder.event != 0)
      {
        encoder.event = 0;
        if (encoder.limit >= mission.section_limit)
        {
          __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
          Motor_Stop();
          Mission_ScheduleTransition(MISSION_DONE);
        }
      }
      break;

    case MISSION_DONE:
      Mission_SetState(MISSION_IDLE);
      break;

    case MISSION_FAULT:
      UART_ClearPending();
      encoder.event = 0;
      break;

    default:
      Mission_SetState(MISSION_FAULT);
      break;
  }
}

void Mission_OnEncoderUpdate(TIM_HandleTypeDef *htim)
{
  if (Ultrasonic_IsHeld()) return;
  /* TIM3의 ARR 경계 이벤트가 구간 기준이다. TIM4 CNT는 PID/DBG1_STATE에 쓰며 이 FSM의 완료 조건이 아니다.
   * 전진/후진 모두 배선에 따른 count 방향이 맞아야 한다. event=1이면 main이 다음 임무 상태를 처리한다. */
  if (htim->Instance == TIM3)
  {
    /* 방향 전환 단계의 첫 보정 이벤트: 이전 방향 관성만큼 되돌아온 원래 경계다. 구간으로 세지 않고 계속 간다. */
    if (Drive_ConsumeSectionPreroll()) return;
    if ((mission.state == REALTIME_CLEANING) ||
        (mission.state == RESCAN_RETURN) ||
        (mission.state == FINAL_RETURN))
    {
      /* Continuous passes keep PWM running through ARR events 1 to 3. */
      /* 연속 통과 구간은 ISR에서 limit를 세어 1~3구간은 주행을 유지하고 4번째에서만 정지/알림을 건다.
       * REALTIME 종료일 때는 청소 출력도 ISR에서 즉시 끈 뒤 main의 RESCAN_RETURN 전이를 기다린다. */
      encoder.limit++;
      if (encoder.limit >= mission.section_limit)
      {
        __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
        Motor_Stop();
        /* 연속 주행의 마지막 경계도 표시해, 다음(반대 방향) 단계가 관성만큼 첫 구간을 늘리게 한다. */
        Drive_MarkSectionBoundary(Drive_GetDirection());
        if (mission.state == REALTIME_CLEANING)
        {
          CleaningActuators_StopAll();
        }
        encoder.event = 1;
      }
    }
    else if ((mission.state == INITIAL_SCAN) ||
             (mission.state == RETURNING) ||
             (mission.state == RESCAN))
    {
      /* Stop at each section; main handles the state transition and UART. */
      /* 촬영/일반 복귀는 매 경계에서 먼저 PWM을 끄고 event만 세운다. limit 증가는 main의 Mission_Update가 한다.
       * 여기서 CAMERA_CAPTURE를 직접 보내지 않아 UART 대기가 IRQ 안에서 구간 처리를 막지 않게 한다. */
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
      /* 정지 뒤 관성으로 경계를 지나 굴러간 거리를 다음 구간에 반영하도록 표시한다.
       * 같은 방향이면 다음 구간에 넣고, 반대 방향(단계 전환)이면 첫 구간을 그만큼 늘린다. */
      Drive_MarkSectionBoundary(Drive_GetDirection());
      encoder.event = 1;
    }
    else
    {
      __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
      Motor_Stop();
    }
  }
}

void Mission_Init(void)
{
  mission = (MissionContext){0};
  encoder = (EncoderContext){0};
  mission.state = MISSION_IDLE;
  mission.section_limit = mission.requested_section_limit = ENCODER_SECTION_LIMIT;
  mission.capture_resume_state = INITIAL_SCAN;
}

MissionState Mission_GetState(void) { return mission.state; }
int Mission_IsTransitionPending(void) { return mission.transition_pending; }
int Mission_IsPrepared(void) { return mission.prepared != 0U; }
int Mission_GetSectionCount(void) { return encoder.limit; }
int Mission_GetSectionLimit(void) { return mission.section_limit; }
int Mission_HasEncoderEvent(void) { return encoder.event; }

void Mission_LatchFault(void)
{
  mission.transition_pending = 0;
  Drive_ClearSectionCarry();
  mission.prepared = 0U;
  mission.state = MISSION_FAULT;
}

void Mission_OnUartFault(void)
{
  /* 대기 중 통신 오류가 발생하면 예약을 취소한다. 수신 복구만으로 다음 주행이 자동 시작되지 않는다. */
  if (mission.transition_pending != 0)
  {
    mission.transition_pending = 0;
    mission.state = MISSION_FAULT;
    __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
    Motor_Stop();
    Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
  }
  if (mission.state == DEBUG_TEST)
  {
    /* UART errors must stop debug drive as well as all four accessories. */
    Motor_Stop();
    Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
    Debug_RequestAbort();
  }
}
