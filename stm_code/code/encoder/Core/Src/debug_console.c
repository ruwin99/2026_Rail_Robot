#include "ultrasonic.h"
/* DBG1 세션, 수신 줄과 시험 watchdog 상태를 이 파일에서 소유한다.
 * ISR은 줄을 수집하고 main의 Update가 검증/구동/응답한다. 다른 모듈의 비공개 상태에 접근하지 않는다. */
#include "debug_console.h"
#include "robot_board.h"
#include "robot_config.h"
#include "drive_sync_config.h"
#include "drive_sync_pid.h"
#include "drive_control.h"
#include "actuators.h"
#include "mission.h"
#include "uart_protocol.h"

#include <stdio.h>
#include <string.h>

typedef struct
{
  char rx_line[96];
  char pending_line[96];
  char session[33];
  uint8_t rx_length;
  uint8_t rx_mode;
  volatile int line_pending;
  volatile int abort_requested;
  volatile int active;
  volatile int pending_drive_fault;
  uint32_t capabilities;
  uint32_t sequence;
  uint32_t percent_basis_points;
  uint32_t last_command_tick;
  uint32_t last_state_tick;
  uint32_t last_control_tick;
  uint8_t control_sent;
  uint32_t last_stop_tick;
  int last_drive_direction;
  uint8_t led_manual, led_red, led_green, led_blue;
  uint8_t led_effect; /* 0=fixed, 1=breathe, 2=blink */
  uint32_t led_epoch;
} DebugContext;

static DebugContext debug_context;

/* 통신 생존과 feedback timestamp는 물체 정지 중에도 흐르는 실제 시간을 쓴다.
 * 임무 이동/촬영 전환의 pause 가능한 시간은 이 모듈에서 변경하지 않는다. */
static uint32_t Debug_Tick(void)
{
  return HAL_GetTick();
}

static void Debug_StopAll(void);

static void Debug_SendControl(void)
{
  if (!(debug_context.capabilities & 16U)) return;
  uint32_t now=Debug_Tick();
  /* SET 응답은 즉시 보내지만 제어 진단은 최대 10Hz. ISR에서는 송신하지 않는다. */
  if (debug_context.control_sent && (uint32_t)(now-debug_context.last_control_tick)<DEBUG_STATE_PERIOD_MS) return;
  DriveControlSnapshot sample;
  char line[192];
  Drive_GetControlSnapshot(&sample);
  int length=snprintf(line,sizeof(line),
    "DBG1_CTRL %s %lu %lu %lu %lu %ld %lu %ld %ld %ld %ld %ld %ld %ld\r\n",
    debug_context.session,(unsigned long)debug_context.sequence,
    (unsigned long)sample.sequence,(unsigned long)sample.tick,(unsigned long)sample.elapsed_ms,
    (long)sample.direction,(unsigned long)sample.flags,(long)sample.delta1,(long)sample.delta2,
    (long)sample.normalized_right,(long)sample.normalized_left,(long)sample.error,
    (long)sample.integral_q8,(long)sample.correction);
  /* direction=-1..1, flags=0..7이고 나머지32비트 정수가 최대 길이여도177바이트.
   * 잘린 프레임은 절대 전송하지 않는다. */
  if (length>0 && (unsigned)length<sizeof(line)) {
    if (UART_V2Active()) {
      char values[72];
      snprintf(values, sizeof(values), "%lu %lu %ld %lu %ld %ld",
        (unsigned long)sample.sequence, (unsigned long)sample.elapsed_ms,
        (long)sample.direction, (unsigned long)sample.flags,
        (long)sample.error, (long)sample.correction);
      UART_QueueStatus("CTRL", values);
    } else UART_SendText(line);
    debug_context.last_control_tick=now;
    debug_context.control_sent=1U;
  }
}

static const char *Debug_DriveFaultReason(int fault)
{
  switch (fault) {
    case DRIVE_SYNC_BAD_CONFIG: return "DRIVE_BAD_CONFIG";
    case DRIVE_SYNC_LATE: return "DRIVE_LATE";
    case DRIVE_SYNC_ENCODER_RANGE: return "DRIVE_ENCODER_RANGE";
    case DRIVE_SYNC_STALL: return "DRIVE_STALL";
    case DRIVE_SYNC_WRONG_DIRECTION: return "DRIVE_WRONG_DIRECTION";
    default: return "READY";
  }
}

static void Debug_SendDiagnostic(int fault)
{
  char line[192];
  UltrasonicSnapshot range;
  uint32_t left, right, pwm1, pwm2;
  uint32_t primask=__get_PRIMASK();
  const char *reason;
  __disable_irq();
  Ultrasonic_GetSnapshot(&range);
  left=__HAL_TIM_GET_COUNTER(&htim3);
  right=__HAL_TIM_GET_COUNTER(&htim4);
  pwm1=__HAL_TIM_GET_COMPARE(&htim2,TIM_CHANNEL_1)+__HAL_TIM_GET_COMPARE(&htim2,TIM_CHANNEL_2);
  pwm2=__HAL_TIM_GET_COMPARE(&htim2,TIM_CHANNEL_3)+__HAL_TIM_GET_COMPARE(&htim2,TIM_CHANNEL_4);
  if (!primask) __enable_irq();
  reason=fault ? Debug_DriveFaultReason(fault) :
    (range.held ? "ULTRASONIC_HOLD" : ((pwm1 || pwm2) ? "RUNNING" : "READY"));
  snprintf(line,sizeof(line),"DBG1_DIAG %s %lu %lu %s %d %u %u %lu %lu %lu %lu\r\n",
    debug_context.session,(unsigned long)debug_context.sequence,
    (unsigned long)Debug_Tick(),reason,fault,(unsigned)range.held,(unsigned)range.detected,
    (unsigned long)left,(unsigned long)right,(unsigned long)pwm1,(unsigned long)pwm2);
  if (UART_V2Active()) {
    char values[72];
    snprintf(values,sizeof(values),"%lu %s %d %u %u %lu %lu %lu %lu",
      (unsigned long)Debug_Tick(),reason,fault,(unsigned)range.held,
      (unsigned)range.detected,(unsigned long)left,(unsigned long)right,
      (unsigned long)pwm1,(unsigned long)pwm2);
    UART_QueueStatus("DIAG",values);
  } else UART_SendText(line);
}

static int Debug_ReportPendingDriveFault(void)
{
  int fault;
  uint32_t primask=__get_PRIMASK();
  __disable_irq();
  fault=debug_context.pending_drive_fault;
  debug_context.pending_drive_fault=0;
  if (!primask) __enable_irq();
  if (fault) {
    char line[112];
    /* ISR이 정지시킨 출력을 확인 가능한 DIAG로 먼저 보내고 세션 오류를 보낸다.
     * CLOSE/deactivate가 먼저 실행되어도 pending fault는 지우지 않는다. */
    Debug_StopAll();
    Debug_SendDiagnostic(fault);
    if (UART_V2Active())
      (void)UART_QueueEvent("DEBUG_FAULT", Debug_DriveFaultReason(fault));
    else {
      snprintf(line,sizeof(line),"DBG1_ERROR %s %s\r\n",debug_context.session,Debug_DriveFaultReason(fault));
      UART_SendText(line);
    }
    Debug_Deactivate();
    return 1;
  }
  return 0;
}

int Debug_ParseByte(uint8_t byte)
{
  /* D로 시작한 줄은 개행까지 DBG 프레임으로 모으고 CR은 생략한다. 다른 줄은 legacy 파서에 넘긴다.
   * 완성 줄 하나만 pending_line에 보관한다. main 소비 전 다음 줄/96바이트 버퍼 초과는 abort_requested를 세운다.
   * DBG1 응답이 없으면 송신 문자열의 줄바꿈과 rx_mode/rx_length/line_pending을 먼저 확인한다. */
  DebugContext *debug = &debug_context;
  if (UART_V2Active()) {
    Debug_ResetRx();
    return 1;
  }
  if (debug->rx_mode == 0U)
  {
    debug->rx_mode = (byte == (uint8_t)'D') ? 1U : 2U;
  }
  if (debug->rx_mode == 2U)
  {
    if (byte == (uint8_t)'\n') debug->rx_mode = 0U;
    return 0;
  }
  if (byte == (uint8_t)'\n')
  {
    if (debug->line_pending != 0)
    {
      debug->abort_requested = 1;
    }
    else
    {
      debug->rx_line[debug->rx_length] = '\0';
      memcpy(debug->pending_line, debug->rx_line,
             (size_t)debug->rx_length + 1U);
      debug->line_pending = 1;
    }
    debug->rx_length = 0U;
    debug->rx_mode = 0U;
  }
  else if (byte != (uint8_t)'\r')
  {
    if (debug->rx_length < sizeof(debug->rx_line) - 1U)
    {
      debug->rx_line[debug->rx_length++] = (char)byte;
    }
    else
    {
      debug->abort_requested = 1;
    }
  }
  return 1;
}

static int Debug_Unsigned(const char *text, uint32_t limit, uint32_t *value)
{
  uint32_t parsed = 0U;
  if ((text == NULL) || (*text == '\0')) return 0;
  while (*text != '\0')
  {
    uint32_t digit;
    if ((*text < '0') || (*text > '9')) return 0;
    digit = (uint32_t)(*text++ - '0');
    if ((digit > limit) || (parsed > (limit - digit) / 10U)) return 0;
    parsed = parsed * 10U + digit;
  }
  *value = parsed;
  return 1;
}

static void Debug_SendState(void)
{
  if (Debug_ReportPendingDriveFault()) return;
  if (debug_context.capabilities == 8U)
  {
    char line[192];
    UltrasonicSnapshot range;
    const char *status;
    Ultrasonic_GetSnapshot(&range);
    status=range.status==ULTRASONIC_ECHO ? "ECHO" :
      (range.status==ULTRASONIC_NO_ECHO ? "NO_ECHO" : "WAITING");
    /* RGB 값은 요청값이다. 센서 전용 세션은 거리 값을 표시만 하고
     * fault가 발생한 경우에만 실제 LED를 빨강으로 덮어쓴다.
     * valid=0/NO_ECHO는 물체 거리 미측정이며 센서가 정상이라는 뜻이 아니다. */
    snprintf(line,sizeof(line),"DBG1_SENSOR %s %lu %lu %lu %u %u %u %u %u %u %u %s\r\n",
      debug_context.session,(unsigned long)debug_context.sequence,
      (unsigned long)HAL_GetTick(),(unsigned long)range.distance_mm,
      (unsigned)range.valid,(unsigned)range.detected,(unsigned)range.held,
      (unsigned)!debug_context.led_manual,(unsigned)debug_context.led_red,
      (unsigned)debug_context.led_green,(unsigned)debug_context.led_blue,status);
    if (UART_V2Active()) {
      char values[72];
      snprintf(values,sizeof(values),"%lu %lu %u %u %u %u %u %u %u %s",
        (unsigned long)HAL_GetTick(),(unsigned long)range.distance_mm,
        (unsigned)range.valid,(unsigned)range.detected,(unsigned)range.held,
        (unsigned)!debug_context.led_manual,(unsigned)debug_context.led_red,
        (unsigned)debug_context.led_green,(unsigned)debug_context.led_blue,status);
      UART_QueueStatus("ULTRASONIC",values);
    } else UART_SendText(line);
    debug_context.last_state_tick=Debug_Tick();
    return;
  }
  /* DBG1_STATE는 session/sequence/tick, TIM3·TIM4 원시 CNT, 방향, 0~10000 속도값, 장치 mask, 두 PWM을 보낸다.
   * CNT는 타이머 범위에서 순환하는 원시값이며 누적 거리나 RPM이 아니다. 출력 mask도 실제 회전/유량 측정값은 아니다. */
  if (Ultrasonic_IsHeld()) Debug_StopAll();
  char line[192];
  uint32_t pwm1 = __HAL_TIM_GET_COMPARE(&htim2, TIM_CHANNEL_1) +
                  __HAL_TIM_GET_COMPARE(&htim2, TIM_CHANNEL_2);
  uint32_t pwm2 = __HAL_TIM_GET_COMPARE(&htim2, TIM_CHANNEL_3) +
                  __HAL_TIM_GET_COMPARE(&htim2, TIM_CHANNEL_4);
  snprintf(line, sizeof(line),
           "DBG1_STATE %s %lu %lu %lu %lu %d %lu %lu %lu %lu\r\n",
           debug_context.session, (unsigned long)debug_context.sequence,
           (unsigned long)Debug_Tick(),
           (unsigned long)__HAL_TIM_GET_COUNTER(&htim3),
           (unsigned long)__HAL_TIM_GET_COUNTER(&htim4),
           Drive_GetDirection(),
           (unsigned long)debug_context.percent_basis_points,
           (unsigned long)Actuators_GetOutputMask(), (unsigned long)pwm1,
           (unsigned long)pwm2);
  if (UART_V2Active()) {
    char values[72];
    snprintf(values,sizeof(values),"%lu %lu %lu %d %lu %lu %lu %lu",
      (unsigned long)Debug_Tick(),
      (unsigned long)__HAL_TIM_GET_COUNTER(&htim3),
      (unsigned long)__HAL_TIM_GET_COUNTER(&htim4),
      Drive_GetDirection(),
      (unsigned long)debug_context.percent_basis_points,
      (unsigned long)Actuators_GetOutputMask(),
      (unsigned long)pwm1,(unsigned long)pwm2);
    UART_QueueStatus("ENCODER",values);
  } else UART_SendText(line);
  debug_context.last_state_tick = Debug_Tick();
  Debug_SendDiagnostic(Drive_GetFault());
  Debug_SendControl();
}

static void Debug_StopAll(void)
{
  Motor_Stop();
  CleaningActuators_StopAll();
  Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
  debug_context.percent_basis_points = 0U;
  debug_context.last_stop_tick = Ultrasonic_OperationalTick();
}

static void Debug_Fault(const char *reason)
{
  /* 세션 오류는 주행·부속장치 OFF 후 DBG1_ERROR를 알리고 MISSION_FAULT에 머문다.
   * 통신 시간 초과가 IDLE로 복귀하는 Debug_Update 경로와 다르므로 오류 문자열과 state를 함께 확인한다. */
  char line[96];
  Debug_StopAll();
  debug_context.active = 0;
  debug_context.led_manual = 0U;
  if (UART_V2Active()) (void)UART_QueueEvent("DEBUG_FAULT", reason);
  else {
    snprintf(line, sizeof(line), "DBG1_ERROR %s %s\r\n",
             debug_context.session, reason);
    UART_SendText(line);
  }
  Mission_SetState(MISSION_FAULT);
}

static void Debug_Apply(int direction, uint32_t percent, uint32_t mask)
{
#if DRIVE_SYNC_PID_ENABLED
  if (Drive_GetFault() != 0 || Mission_GetState() == MISSION_FAULT) {
    Debug_StopAll();
    return;
  }
#endif
  /* percent는 백분율의 1/100 단위(10000=100%)다. CCR은 PWM_DUTY를 기준으로 비례 계산한다.
   * 방향 0 또는 속도 0은 주행만 정지시킨다. 네 청소 장치는 이 함수 뒤쪽의 mask 값대로 별도로 적용한다. */
  uint32_t tick = Ultrasonic_OperationalTick();
  if ((direction == MOTOR_DIRECTION_STOPPED) || (percent == 0U))
  {
    if (Drive_GetDirection() != MOTOR_DIRECTION_STOPPED)
      debug_context.last_stop_tick = tick;
    Motor_Stop();
    Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
    debug_context.percent_basis_points = 0U;
  }
  else if ((Drive_GetDirection() != MOTOR_DIRECTION_STOPPED) &&
           (direction != Drive_GetDirection()))
  {
    /* Even a client that skips STOP must not reverse an energized bridge. */
    /* 회전 중 반대 방향 SET는 우선 정지만 수행한다. 이후 SET가 다시 들어와야 아래 재시작 조건을 검사한다.
     * 100ms 정지 간격이 지나기 전에는 PWM을 켜지 않으므로 반전이 한 번 늦어 보일 수 있다. */
    Motor_Stop();
    Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
    debug_context.percent_basis_points = 0U;
    debug_context.last_stop_tick = tick;
  }
  else if ((debug_context.last_drive_direction != MOTOR_DIRECTION_STOPPED) &&
           (direction != debug_context.last_drive_direction) &&
           ((uint32_t)(tick - debug_context.last_stop_tick) < DEBUG_REVERSE_STOP_MS))
  {
    Motor_Stop();
  }
  else
  {
    uint32_t pwm = ((uint32_t)PWM_DUTY * percent + 5000U) / 10000U;
    Motor_Run(direction, (int)pwm);
    Drive_SetDirection(direction);
    debug_context.last_drive_direction = direction;
    debug_context.percent_basis_points = percent;
  }
  if ((mask & 1U) != 0U)
  {
    /* mask 비트는 앞 브러시=1, 옆 브러시=2, 앞 펌프=4, 옆 펌프=8이다.
     * 한 SET가 네 장치 전체 상태를 나타내며 포함되지 않은 비트는 OFF로 적용된다. */
    Front_Cleaner_Run(FRONT_DC_MOTOR_PWM_DUTY_12_5);
    Actuators_Refresh(ACTUATOR_FRONT_CLEANER, tick);
  }
  else Front_Cleaner_Stop();
  if ((mask & 2U) != 0U)
  {
    Side_Cleaner_Run(SIDE_DC_MOTOR_PWM_DUTY_12_5);
    Actuators_Refresh(ACTUATOR_SIDE_CLEANER, tick);
  }
  else Side_Cleaner_Stop();
  if ((mask & 4U) != 0U)
  {
    Front_Pump_Start();
    Actuators_Refresh(ACTUATOR_FRONT_PUMP, tick);
  }
  else Front_Pump_Stop();
  if ((mask & 8U) != 0U)
  {
    Side_Pump_Start();
    Actuators_Refresh(ACTUATOR_SIDE_PUMP, tick);
  }
  else Side_Pump_Stop();
}

static void Debug_HandleLine(char *line)
{
  /* capability 1~7은 기존 구동 시험, 8은 구동 권한 없는 LED/초음파 전용이다.
   * bit16은 모터 권한(bit1)이 있는 세션만 협상하는 제어 표본 확장이다.
   * 센서 전용 세션은 hold 중에도 열 수 있지만 임무 실행 중에는 BUSY다. */
  char *tokens[7];
  char reply[160];
  uint32_t count = 0U, capability, sequence, percent, mask;
  int direction;
  char *token = strtok(line, " ");
  while ((token != NULL) && (count < 7U))
  {
    tokens[count++] = token;
    token = strtok(NULL, " ");
  }
  if (count == 0U) return;
  /* 토큰 배열이 찬 뒤 남은 문자열도 잘못된 명령으로 처리한다. */
  if (token != NULL)
  {
    if (debug_context.active && count>1U && strcmp(tokens[1],debug_context.session)==0)
      Debug_Fault("INVALID_COMMAND");
    return;
  }
  if ((strcmp(tokens[0], "DBG1_HELLO") == 0) && (count == 3U))
  {
    if ((strlen(tokens[1]) != 32U) ||
        (strspn(tokens[1], "0123456789abcdef") != 32U) ||
        (Debug_Unsigned(tokens[2], 23U, &capability) == 0) ||
        !((capability>=1U && capability<=8U) ||
          ((capability&~23U)==0U && (capability&17U)==17U)))
      return;
    if (Mission_GetState() != MISSION_IDLE)
    {
      snprintf(reply, sizeof(reply), "DBG1_REJECT %s BUSY state=%d fault=%d\r\n",
        tokens[1],(int)Mission_GetState(),Drive_GetFault());
      UART_SendText(reply);
      return;
    }
    if (Ultrasonic_IsHeld() && capability!=8U) {
      snprintf(reply,sizeof(reply),"DBG1_REJECT %s ULTRASONIC_HOLD\r\n",tokens[1]);
      UART_SendText(reply);
      return;
    }
    UART_QueueFailSafeAllOff();
    /* 세션을 새로 기록하고 모든 출력을 정지시킨 후 READY로 실제 PWM 기준·보정·엔코더 주기를 알려준다.
     * Jetson이 DBG1 미지원이라고 하면 HELLO 프레임, BUSY 응답, 실제 반영 펌웨어를 이 지점과 대조한다. */
    memcpy(debug_context.session, tokens[1], 33U);
    debug_context.active = 1;
    debug_context.abort_requested = 0;
    debug_context.sequence = 0U;
    debug_context.capabilities = capability;
    debug_context.control_sent=0U;
    debug_context.led_manual=debug_context.led_red=debug_context.led_green=debug_context.led_blue=0U;
    Mission_SetState(DEBUG_TEST);
    if (Mission_GetState()!=DEBUG_TEST)
    {
      /* HELLO 도중 장애물이 진입하면 구동 시험은 열지 않는다. 센서 전용만 허용된다. */
      debug_context.active=0;
      if (Ultrasonic_IsHeld() && Mission_GetState()==MISSION_IDLE)
        snprintf(reply,sizeof(reply),"DBG1_REJECT %s ULTRASONIC_HOLD\r\n",tokens[1]);
      else
        snprintf(reply,sizeof(reply),"DBG1_REJECT %s BUSY state=%d fault=%d\r\n",
          tokens[1],(int)Mission_GetState(),Drive_GetFault());
      UART_SendText(reply);
      return;
    }
    debug_context.last_drive_direction = MOTOR_DIRECTION_STOPPED;
    debug_context.last_command_tick = Debug_Tick();
    Debug_StopAll();
    /* READY의 기존 단일 trim 필드는 전진 보정값이다. wire 형식/필드 수는 유지한다.
     * 단일 DC 필드는 호환용 SIDE 저속값이며 FRONT는 별도 FRONT_DC_MOTOR_PWM_DUTY_12_5 설정을 실제 출력에 적용한다. */
    snprintf(reply, sizeof(reply), "DBG1_READY %s %lu %u %u %u %u %lu %lu BP\r\n",
             debug_context.session, (unsigned long)capability, PWM_DUTY,
             (unsigned)RIGHT_MOTOR_FORWARD_TRIM_BP, SIDE_DC_MOTOR_PWM_DUTY_12_5,
             WATER_PUMP_PWM_DUTY,
             (unsigned long)__HAL_TIM_GET_AUTORELOAD(&htim3) + 1UL,
             (unsigned long)__HAL_TIM_GET_AUTORELOAD(&htim4) + 1UL);
    UART_SendText(reply);
    Debug_SendState();
    return;
  }
  if ((debug_context.active == 0) || (Mission_GetState() != DEBUG_TEST)) return;
  /* 다른 세션의 지연 명령은 무시한다. CLOSE는 주행/청소를 모두 끄고 상태를 한 번 보낸 뒤 IDLE로 돌아간다.
   * 세션이 맞아도 잘못된 SET 형식·방향·권한은 아래 fault 경로로 가며 임의 값으로 보정하지 않는다. */
  if ((count < 2U) || (strcmp(tokens[1], debug_context.session) != 0)) return;
  if ((strcmp(tokens[0], "DBG1_CLOSE") == 0) && (count == 2U))
  {
    Debug_StopAll();
    debug_context.led_manual=0U;
    Debug_SendState();
    debug_context.active = 0;
    Mission_SetState(MISSION_IDLE);
    return;
  }
  if (strcmp(tokens[0], "DBG1_LED") == 0)
  {
    uint32_t red, green, blue;
    if (debug_context.capabilities!=8U || count!=7U ||
        !Debug_Unsigned(tokens[2],UINT32_MAX,&sequence) ||
        (strcmp(tokens[3],"auto")!=0 && strcmp(tokens[3],"manual")!=0 &&
         strcmp(tokens[3],"breathe")!=0 && strcmp(tokens[3],"blink")!=0) ||
        !Debug_Unsigned(tokens[4],255U,&red) ||
        !Debug_Unsigned(tokens[5],255U,&green) ||
        !Debug_Unsigned(tokens[6],255U,&blue))
    { Debug_Fault("INVALID_LED"); return; }
    if (sequence<=debug_context.sequence) return;
    debug_context.sequence=sequence;
    debug_context.last_command_tick=Debug_Tick();
    debug_context.led_manual=(strcmp(tokens[3],"auto")!=0);
    debug_context.led_effect=(strcmp(tokens[3],"breathe")==0) ? 1U :
      ((strcmp(tokens[3],"blink")==0) ? 2U : 0U);
    debug_context.led_epoch=Debug_Tick();
    debug_context.led_red=(uint8_t)red;
    debug_context.led_green=(uint8_t)green;
    debug_context.led_blue=(uint8_t)blue;
    Debug_StopAll();
    Debug_SendState();
    return;
  }
  if ((strcmp(tokens[0], "DBG1_SET") != 0) || (count != 6U) ||
      (Debug_Unsigned(tokens[2], UINT32_MAX, &sequence) == 0) ||
      (Debug_Unsigned(tokens[4], 10000U, &percent) == 0) ||
      (Debug_Unsigned(tokens[5], 15U, &mask) == 0))
  {
    Debug_Fault("INVALID_COMMAND");
    return;
  }
  if (strcmp(tokens[3], "-1") == 0) direction = MOTOR_DIRECTION_REVERSE;
  else if (strcmp(tokens[3], "1") == 0) direction = MOTOR_DIRECTION_FORWARD;
  else if (strcmp(tokens[3], "0") == 0) direction = MOTOR_DIRECTION_STOPPED;
  else { Debug_Fault("INVALID_DIRECTION"); return; }
  if ((((debug_context.capabilities & 1U) == 0U) && ((direction != 0) || (percent != 0U))) ||
      (((debug_context.capabilities & 2U) == 0U) && ((mask & 3U) != 0U)) ||
      (((debug_context.capabilities & 4U) == 0U) && ((mask & 12U) != 0U)) ||
      ((direction == 0) && (percent != 0U)))
  { Debug_Fault("CAPABILITY_OR_RANGE"); return; }
  if (sequence <= debug_context.sequence) return;
  /* 새 순번만 채택하고 watchdog 시각을 갱신한다. 같은 순번 반복은 heartbeat가 되지 않는다.
   * SET 직전에 ISR이 오류를 표시했다면 ON 적용을 막아 수신 오류 OFF와 경합하지 않게 한다. */
  debug_context.sequence = sequence;
  debug_context.last_command_tick = Debug_Tick();
  {
    /* An RX error may request OFF while the mailbox is being consumed.
       Do not overwrite that stop with the remainder of an ON snapshot. */
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    if (debug_context.abort_requested == 0) {
      /* hold 중 유효 명령은 heartbeat로 접수하지만 ON은 저장/적용하지 않는다.
       * 실제 UART 문제와 센서에 의한 정지를 구별할 수 있게 ACK/state를 계속 보낸다. */
      if (Ultrasonic_IsHeld()) Debug_StopAll();
      else Debug_Apply(direction, percent, mask);
    }
    if (primask == 0U) __enable_irq();
  }
  if (debug_context.abort_requested != 0)
  {
    debug_context.abort_requested = 0;
    Debug_Fault("UART_OR_OVERFLOW");
    return;
  }
  Debug_SendState();
}

void Debug_Update(void)
{
  /* ISR의 한 칸 줄 명령함을 짧은 임계구역에서 복사하고 비운 뒤 main에서 처리한다.
   * overflow/UART 중단 요청을 먼저 처리하므로 이미 받은 ON 줄이 있어도 이후에 되살리지 않는다. */
  char line[96];
  int pending = 0;
  uint32_t primask = __get_PRIMASK();
  __disable_irq();
  if (debug_context.line_pending != 0)
  {
    memcpy(line, debug_context.pending_line, sizeof(line));
    debug_context.line_pending = 0;
    pending = 1;
  }
  if (primask == 0U) __enable_irq();
  if (Debug_ReportPendingDriveFault()) return;
  if (debug_context.abort_requested != 0)
  {
    debug_context.abort_requested = 0;
    if (debug_context.active != 0) Debug_Fault("UART_OR_OVERFLOW");
    return;
  }
  /* Expiry precedes mailbox consumption: a late SET must never renew an
     already expired session or briefly energize its previous target. */
  /* 600ms 동안 새 유효 SET가 없으면 모든 출력을 끄고 TIMEOUT 후 IDLE로 복귀한다.
   * 늦게 도착한 SET보다 만료 판정이 우선한다. 이 감시는 main 반복 루프가 실행되어야 동작하는 소프트웨어 감시다. */
  if ((debug_context.active != 0) &&
      ((uint32_t)(Debug_Tick() - debug_context.last_command_tick) >= DEBUG_COMMAND_TIMEOUT_MS))
  {
    char message[80];
    Debug_StopAll();
    if (UART_V2Active()) (void)UART_QueueEvent("DEBUG_TIMEOUT", NULL);
    else {
      snprintf(message, sizeof(message), "DBG1_TIMEOUT %s\r\n", debug_context.session);
      UART_SendText(message);
    }
    debug_context.active = 0;
    debug_context.led_manual=0U;
    Mission_SetState(MISSION_IDLE);
    return;
  }
  if (pending != 0) Debug_HandleLine(line);
  if ((debug_context.active != 0) &&
      ((uint32_t)(Debug_Tick() - debug_context.last_state_tick) >= DEBUG_STATE_PERIOD_MS))
    Debug_SendState();
}

void Debug_Init(void) { debug_context = (DebugContext){0}; }

int Debug_RR2Begin(const char *session, UartRr2Capability capability)
{
  uint32_t capabilities;
  if (session == NULL || Mission_GetState() != MISSION_IDLE ||
      Mission_IsTransitionPending()) return 0;
  if (Ultrasonic_IsHeld() && capability != UART_RR2_CAP_SENSOR) return 0;
  switch (capability) {
    case UART_RR2_CAP_ENCODER: capabilities = 1U; break;
    case UART_RR2_CAP_DC: capabilities = 2U; break;
    case UART_RR2_CAP_PUMP: capabilities = 4U; break;
    case UART_RR2_CAP_ALL: capabilities = 23U; break;
    case UART_RR2_CAP_SENSOR: capabilities = 8U; break;
    default: return 0;
  }
  debug_context.active = 1;
  debug_context.abort_requested = 0;
  debug_context.pending_drive_fault = 0;
  debug_context.sequence = 0U;
  debug_context.capabilities = capabilities;
  debug_context.control_sent = 0U;
  debug_context.led_manual = 0U;
  debug_context.led_red = debug_context.led_green = debug_context.led_blue = 0U;
  memcpy(debug_context.session, session, 33U);
  Mission_SetState(DEBUG_TEST);
  if (Mission_GetState() != DEBUG_TEST) {
    debug_context.active = 0;
    return 0;
  }
  debug_context.last_drive_direction = MOTOR_DIRECTION_STOPPED;
  debug_context.last_command_tick = Debug_Tick();
  debug_context.last_state_tick = debug_context.last_command_tick;
  Debug_StopAll();
  return 1;
}

int Debug_RR2Set(uint32_t sequence, int direction, uint32_t percent_basis_points,
                 uint32_t actuator_mask)
{
  uint32_t capabilities = debug_context.capabilities;
  if (!debug_context.active || Mission_GetState() != DEBUG_TEST ||
      direction < MOTOR_DIRECTION_REVERSE || direction > MOTOR_DIRECTION_FORWARD ||
      percent_basis_points > 10000U || actuator_mask > 15U ||
      (direction == MOTOR_DIRECTION_STOPPED && percent_basis_points != 0U)) return 0;
  if (((direction != MOTOR_DIRECTION_STOPPED || percent_basis_points != 0U) &&
       (capabilities & 1U) == 0U) ||
      ((actuator_mask & 3U) != 0U && (capabilities & 2U) == 0U) ||
      ((actuator_mask & 12U) != 0U && (capabilities & 4U) == 0U)) return 0;
  debug_context.sequence = sequence;
  debug_context.last_command_tick = Debug_Tick();
  if (Ultrasonic_IsHeld()) Debug_StopAll();
  else Debug_Apply(direction, percent_basis_points, actuator_mask);
  Debug_SendState();
  return 1;
}

int Debug_RR2Led(uint32_t sequence, const char *mode,
                 uint32_t red, uint32_t green, uint32_t blue)
{
  if (!debug_context.active || Mission_GetState() != DEBUG_TEST ||
      debug_context.capabilities != 8U || mode == NULL ||
      red > 255U || green > 255U || blue > 255U ||
      (strcmp(mode, "auto") != 0 && strcmp(mode, "manual") != 0 &&
       strcmp(mode, "breathe") != 0 && strcmp(mode, "blink") != 0)) return 0;
  debug_context.sequence = sequence;
  debug_context.last_command_tick = Debug_Tick();
  debug_context.led_manual = (strcmp(mode, "auto") != 0);
  debug_context.led_effect = (strcmp(mode, "breathe") == 0) ? 1U :
      ((strcmp(mode, "blink") == 0) ? 2U : 0U);
  debug_context.led_epoch = Debug_Tick();
  debug_context.led_red = (uint8_t)red;
  debug_context.led_green = (uint8_t)green;
  debug_context.led_blue = (uint8_t)blue;
  Debug_StopAll();
  Debug_SendState();
  return 1;
}

void Debug_RR2Stop(void)
{
  if (!debug_context.active) return;
  Debug_StopAll();
  debug_context.last_command_tick = Debug_Tick();
  Debug_SendState();
}

void Debug_RR2Heartbeat(void)
{
  if (debug_context.active) debug_context.last_command_tick = Debug_Tick();
}

void Debug_ResetRx(void)
{
  debug_context.rx_length = 0U;
  debug_context.rx_mode = 0U;
}
void Debug_RequestAbort(void) { debug_context.abort_requested = 1; }
void Debug_LatchDriveFault(int fault)
{
  /* SysTick fault 경로 전용: UART나 format 연산 없이 첫 오류만 main에 넘긴다. */
  if (debug_context.active && debug_context.capabilities!=8U &&
      !debug_context.pending_drive_fault)
    debug_context.pending_drive_fault=fault;
}
void Debug_Deactivate(void) { debug_context.active = 0; debug_context.led_manual=0U; }
int Debug_IsSensorSession(void) { return debug_context.active && debug_context.capabilities==8U; }

int Debug_GetLedOverride(uint8_t *red, uint8_t *green, uint8_t *blue)
{
  uint32_t elapsed, level=255U;
  if (!debug_context.active || debug_context.capabilities!=8U ||
      !debug_context.led_manual || Mission_GetState()!=DEBUG_TEST) return 0;
  elapsed=(uint32_t)(Debug_Tick()-debug_context.led_epoch) % 1000U;
  if (debug_context.led_effect==1U)
    level=((elapsed<=500U ? elapsed : 1000U-elapsed) * 255U) / 500U;
  else if (debug_context.led_effect==2U)
    level=elapsed<500U ? 255U : 0U;
  *red=(uint8_t)(((uint32_t)debug_context.led_red*level)/255U);
  *green=(uint8_t)(((uint32_t)debug_context.led_green*level)/255U);
  *blue=(uint8_t)(((uint32_t)debug_context.led_blue*level)/255U);
  return 1;
}

void Debug_ObstacleBarrier(void)
{
  Debug_ResetRx();
  debug_context.line_pending=0;
}
