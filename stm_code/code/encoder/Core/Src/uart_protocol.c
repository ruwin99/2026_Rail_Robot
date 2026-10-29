#include "drive_control.h"
#include "ultrasonic.h"
/* USART2 수신 바이트, 문자열 매칭 상태와 legacy 명령함을 이 파일에서 소유한다.
 * ISR은 수신/오류를 처리하고 main은 명령을 꺼내 실행한다. 응답 송신은 main에서만 한다. */
#include "uart_protocol.h"
#include "robot_board.h"
#include "robot_config.h"
#include "actuators.h"
#include "mission.h"
#include "robot_app.h"
#include "debug_console.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define RR2_LINE_CAP 128U
#define RR2_HEARTBEAT_TIMEOUT_MS 600U
/* legacy 문자열 경로: 주행 중 이 시간 동안 legacy 명령(LINK_OK 포함)이 한 줄도 없으면 정지·FAULT.
 * Jetson legacy 송신측은 START 뒤 200 ms마다 LINK_OK를 보낸다. */
#define LEGACY_LINK_TIMEOUT_MS 1000U
#define RR2_EVENT_CAP 4U
#define RR2_EVENT_RETRY_MS 100U
#define RR2_EVENT_MAX_ATTEMPTS 3U
#define RR2_STATUS_CAP 4U
#define UART_TX_CAP 8U
#define UART_TX_LINE_CAP 192U
#define UART_TX_BURST 4U

typedef struct
{
  char code[24];
  char payload[64];
  uint32_t sequence;
  uint32_t last_sent_tick;
  uint8_t attempts;
} Rr2Event;

typedef struct
{
  char key[16];
  char values[72];
  uint8_t pending;
} Rr2Status;

typedef struct
{
  char bytes[UART_TX_LINE_CAP];
  uint16_t length;
} UartTxItem;

typedef struct
{
  volatile uint8_t pending_count;
  volatile uint8_t read_slot;
  volatile uint8_t write_slot;
  volatile uint8_t receiving;
  volatile uint8_t length;
  volatile uint8_t active;
  char rx[RR2_LINE_CAP];
  char line[2][RR2_LINE_CAP];
  char session[33];
  uint32_t last_seen_tick;
  /* Host CMD retries and PING heartbeat have independent sequence lanes.
   * Interleaving a newer PING must not evict an unacknowledged CMD response. */
  uint32_t last_cmd_sequence;
  uint32_t last_ping_sequence;
  uint32_t tx_sequence;
  char last_cmd_request[RR2_LINE_CAP];
  char last_cmd_reply[RR2_LINE_CAP];
  char last_ping_request[RR2_LINE_CAP];
  char last_ping_reply[RR2_LINE_CAP];
  UartRr2Capability capability;
  Rr2Event events[RR2_EVENT_CAP];
  uint8_t event_head;
  uint8_t event_count;
  Rr2Status statuses[RR2_STATUS_CAP];
  uint8_t status_cursor;
} Rr2Context;

static Rr2Context rr2;
static uint8_t rr2_line_start;
static UartTxItem uart_tx[UART_TX_CAP];
static volatile uint8_t uart_tx_head, uart_tx_count;
static volatile uint32_t uart_tx_generation;
static volatile uint8_t rr2_fault_pending;

static int UART_QueueCommand(UartCommand command);

static uint32_t UART_TxEnterCritical(void)
{
  uint32_t primask = __get_PRIMASK();
  __disable_irq();
  return primask;
}

static void UART_TxExitCritical(uint32_t primask)
{
  if (primask == 0U) __enable_irq();
}

static void UART_ClearTxQueue(void)
{
  uint32_t primask = UART_TxEnterCritical();
  uart_tx_head = 0U;
  uart_tx_count = 0U;
  uart_tx_generation++;
  UART_TxExitCritical(primask);
}

static int UART_TakeTxItem(UartTxItem *item, uint32_t *generation)
{
  uint8_t head;
  uint32_t primask;
  /* The queue payload is main-owned; keep the relatively long copy outside
   * the IRQ mask and protect only its shared ring metadata. */
  if (uart_tx_count == 0U) return 0;
  head = uart_tx_head;
  *item = uart_tx[head];
  primask = UART_TxEnterCritical();
  uart_tx_head = (uint8_t)((uart_tx_head + 1U) % UART_TX_CAP);
  uart_tx_count--;
  *generation = uart_tx_generation;
  UART_TxExitCritical(primask);
  return 1;
}

static uint16_t RR2_Crc(const char *body)
{
  uint16_t crc = 0xffffU;
  while (*body != '\0') {
    crc ^= (uint16_t)(uint8_t)*body++ << 8;
    for (unsigned bit = 0U; bit < 8U; ++bit)
      crc = (uint16_t)((crc & 0x8000U) ? (crc << 1) ^ 0x1021U : crc << 1);
  }
  return crc;
}

static void RR2_Send(const char *body)
{
  char line[RR2_LINE_CAP];
  int length = snprintf(line, sizeof(line), "%s %04X\r\n", body, RR2_Crc(body));
  if (length <= 0 || (size_t)length >= sizeof(line)) return;
  UART_SendText(line);
}

static void RR2_Fault(void)
{
  /* Main-context session teardown. UART IRQs only latch a pending fault and
   * perform the immediate output stop; queue/session bookkeeping stays here. */
  rr2.active = 0U;
  rr2.capability = UART_RR2_CAP_NONE;
  rr2.event_count = 0U;
  rr2.event_head = 0U;
  memset(rr2.statuses, 0, sizeof(rr2.statuses));
  UART_ClearTxQueue();
  Debug_Deactivate();
  Mission_LatchFault();
  __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
  Motor_Stop();
  Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
  CleaningActuators_StopAll();
  UART_ClearPending();
}

static void RR2_RequestFault(void)
{
  if (rr2.active == 0U) return;
  rr2_fault_pending = 1U;
  uart_tx_generation++;
  /* Latch the volatile mission interlock before returning to a preempted main
   * update; otherwise its pending actuator/mission work could re-enable output.
   * Session and TX-queue teardown remain deferred to UART_Update. */
  Mission_LatchFault();
  __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
  Motor_Stop();
  Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
  CleaningActuators_StopAll();
}

static int RR2_ServicePendingFault(void)
{
  uint8_t pending;
  uint32_t primask;
  if (rr2_fault_pending == 0U) return 0;
  primask = __get_PRIMASK();
  __disable_irq();
  pending = rr2_fault_pending;
  rr2_fault_pending = 0U;
  if (primask == 0U) __enable_irq();
  if (pending == 0U) return 0;
  RR2_Fault();
  return 1;
}

static int RR2_CanEndWithoutFault(void)
{
  MissionState state = Mission_GetState();
  /* MISSION: 정상 완료 뒤처럼 아무것도 구동하지 않는 IDLE일 때만 참이다.
   * 준비·전이 예약·주행 PWM·부속장치 출력 중 하나라도 있으면 기존대로 FAULT를 건다. */
  if (rr2.capability == UART_RR2_CAP_MISSION)
    return state == MISSION_IDLE && !Mission_IsPrepared() &&
           !Mission_IsTransitionPending() && !Ultrasonic_IsHeld() &&
           Actuators_GetOutputMask() == 0U &&
           __HAL_TIM_GET_COMPARE(&htim2, TIM_CHANNEL_1) == 0U &&
           __HAL_TIM_GET_COMPARE(&htim2, TIM_CHANNEL_2) == 0U &&
           __HAL_TIM_GET_COMPARE(&htim2, TIM_CHANNEL_3) == 0U &&
           __HAL_TIM_GET_COMPARE(&htim2, TIM_CHANNEL_4) == 0U;
  /* 디버그 세션(센서·주행·DC·펌프·ALL): 호스트가 떠난 것은 세션 종료로 본다. 출력은 종료할 때 모두 끈다.
   * UART 오류·PID/엔코더 이상은 이미 FAULT로 래치되어 여기까지 오지 않는다. 드라이브 fault가 남아 있으면 FAULT를 유지한다. */
  return rr2.capability != UART_RR2_CAP_NONE &&
         (state == DEBUG_TEST || state == MISSION_IDLE) &&
         !Mission_IsTransitionPending() && Drive_GetFault() == 0;
}

static void RR2_EndSessionWithoutFault(void)
{
  /* 끊김을 세션 종료로만 처리한다. FAULT를 래치하지 않으므로 새 HELLO로 다시 연결할 수 있다.
   * 출력을 먼저 끄고, 남은 이벤트·상태·송신·legacy 명령은 모두 버린다. 디버그 세션이면 IDLE로 되돌린다. */
  int debug_session = rr2.capability != UART_RR2_CAP_MISSION;
  __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
  Motor_Stop();
  Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
  CleaningActuators_StopAll();
  Actuators_ClearPending();
  rr2.active = 0U;
  rr2.capability = UART_RR2_CAP_NONE;
  rr2.event_count = 0U;
  rr2.event_head = 0U;
  memset(rr2.events, 0, sizeof(rr2.events));
  memset(rr2.statuses, 0, sizeof(rr2.statuses));
  UART_ClearTxQueue();
  UART_ClearPending();
  if (debug_session) {
    Debug_Deactivate();
    if (Mission_GetState() == DEBUG_TEST) Mission_SetState(MISSION_IDLE);
  }
}

int UART_V2Active(void) { return rr2.active != 0U; }
UartRr2Capability UART_RR2GetCapability(void) { return rr2.capability; }

typedef struct
{
  volatile uint32_t pending_commands;
  /* 마지막으로 끝까지 일치한 legacy 명령 한 줄의 수신 시각(ISR이 기록, main이 감시). */
  volatile uint32_t legacy_seen_tick;
  uint8_t match_index[UART_COMMAND_COUNT];
  volatile int rearm_required;
  uint8_t rx_byte;
} UartContext;

typedef struct
{
  const char *text;
  uint16_t length;
} UartText;

static UartContext uart;

static const UartText uart_command_table[UART_COMMAND_COUNT] =
{
  [UART_COMMAND_NONE] = {NULL, 0U},
  [UART_COMMAND_START] = {"START\r\n", (uint16_t)(sizeof("START\r\n") - 1U)},
  [UART_COMMAND_MISSION_PREPARE] = {"MISSION_PREPARE\r\n", (uint16_t)(sizeof("MISSION_PREPARE\r\n") - 1U)},
  [UART_COMMAND_MISSION_CANCEL_PREPARE] = {"MISSION_CANCEL_PREPARE\r\n", (uint16_t)(sizeof("MISSION_CANCEL_PREPARE\r\n") - 1U)},
  [UART_COMMAND_CAPTURE_OK] = {"CAPTURE_OK\r\n", (uint16_t)(sizeof("CAPTURE_OK\r\n") - 1U)},
  [UART_COMMAND_ACTUATOR_TEST_START] = {"ACTUATOR_TEST_START\r\n", (uint16_t)(sizeof("ACTUATOR_TEST_START\r\n") - 1U)},
  [UART_COMMAND_ACTUATOR_TEST_STOP] = {"ACTUATOR_TEST_STOP\r\n", (uint16_t)(sizeof("ACTUATOR_TEST_STOP\r\n") - 1U)},
  [UART_COMMAND_FRONT_CLEANER_PWM_12_5] = {"FRONT_CLEANER_PWM_12_5\r\n", (uint16_t)(sizeof("FRONT_CLEANER_PWM_12_5\r\n") - 1U)},
  [UART_COMMAND_FRONT_CLEANER_PWM_25] = {"FRONT_CLEANER_PWM_25\r\n", (uint16_t)(sizeof("FRONT_CLEANER_PWM_25\r\n") - 1U)},
  [UART_COMMAND_FRONT_CLEANER_OFF] = {"FRONT_CLEANER_OFF\r\n", (uint16_t)(sizeof("FRONT_CLEANER_OFF\r\n") - 1U)},
  [UART_COMMAND_FRONT_PUMP_ON] = {"FRONT_PUMP_ON\r\n", (uint16_t)(sizeof("FRONT_PUMP_ON\r\n") - 1U)},
  [UART_COMMAND_FRONT_PUMP_OFF] = {"FRONT_PUMP_OFF\r\n", (uint16_t)(sizeof("FRONT_PUMP_OFF\r\n") - 1U)},
  [UART_COMMAND_SIDE_CLEANER_PWM_12_5] = {"SIDE_CLEANER_PWM_12_5\r\n", (uint16_t)(sizeof("SIDE_CLEANER_PWM_12_5\r\n") - 1U)},
  [UART_COMMAND_SIDE_CLEANER_PWM_25] = {"SIDE_CLEANER_PWM_25\r\n", (uint16_t)(sizeof("SIDE_CLEANER_PWM_25\r\n") - 1U)},
  [UART_COMMAND_SIDE_CLEANER_OFF] = {"SIDE_CLEANER_OFF\r\n", (uint16_t)(sizeof("SIDE_CLEANER_OFF\r\n") - 1U)},
  [UART_COMMAND_SIDE_PUMP_ON] = {"SIDE_PUMP_ON\r\n", (uint16_t)(sizeof("SIDE_PUMP_ON\r\n") - 1U)},
  [UART_COMMAND_SIDE_PUMP_OFF] = {"SIDE_PUMP_OFF\r\n", (uint16_t)(sizeof("SIDE_PUMP_OFF\r\n") - 1U)},
  [UART_COMMAND_LINK_OK] = {"LINK_OK\r\n", (uint16_t)(sizeof("LINK_OK\r\n") - 1U)}
};

static const UartText uart_message_table[UART_MESSAGE_COUNT] =
{
  [UART_MESSAGE_STARTED] = {"STARTED\r\n", (uint16_t)(sizeof("STARTED\r\n") - 1U)},
  [UART_MESSAGE_ACTUATOR_TEST_READY] = {"ACTUATOR_TEST_READY\r\n", (uint16_t)(sizeof("ACTUATOR_TEST_READY\r\n") - 1U)},
  [UART_MESSAGE_CAMERA_CAPTURE] = {"CAMERA_CAPTURE\r\n", (uint16_t)(sizeof("CAMERA_CAPTURE\r\n") - 1U)},
  [UART_MESSAGE_RETURN_START] = {"RETURN_START\r\n", (uint16_t)(sizeof("RETURN_START\r\n") - 1U)},
  [UART_MESSAGE_REALTIME_START] = {"REALTIME_START\r\n", (uint16_t)(sizeof("REALTIME_START\r\n") - 1U)},
  [UART_MESSAGE_RESCAN_RETURN_START] = {"RESCAN_RETURN_START\r\n", (uint16_t)(sizeof("RESCAN_RETURN_START\r\n") - 1U)},
  [UART_MESSAGE_RESCAN_START] = {"RESCAN_START\r\n", (uint16_t)(sizeof("RESCAN_START\r\n") - 1U)},
  [UART_MESSAGE_RESCAN_DONE] = {"RESCAN_DONE\r\n", (uint16_t)(sizeof("RESCAN_DONE\r\n") - 1U)},
  [UART_MESSAGE_DONE] = {"DONE\r\n", (uint16_t)(sizeof("DONE\r\n") - 1U)}
};

static int RR2_ValidSession(const char *session)
{
  if (strlen(session) != 32U) return 0;
  for (unsigned i = 0U; i < 32U; ++i)
    if (!((session[i] >= '0' && session[i] <= '9') ||
          (session[i] >= 'a' && session[i] <= 'f'))) return 0;
  return 1;
}

static int RR2_ParseSequence(const char *text, uint32_t *sequence)
{
  uint32_t value = 0U;
  if (*text == '\0') return 0;
  while (*text != '\0') {
    uint32_t digit;
    if (*text < '0' || *text > '9') return 0;
    digit = (uint32_t)(*text++ - '0');
    if (value > (UINT32_MAX - digit) / 10U) return 0;
    value = value * 10U + digit;
  }
  if (value == 0U) return 0;
  *sequence = value;
  return 1;
}

static void RR2_Reply(const char *verb, const char *session, uint32_t sequence,
                      const char *opcode, const char *result, char *cache)
{
  char body[RR2_LINE_CAP];
  int length = snprintf(body, sizeof(body), "RR2 %s %s %lu %s %s %d",
                        verb, session, (unsigned long)sequence, opcode, result,
                        (int)Mission_GetState());
  if (length <= 0 || (size_t)length >= sizeof(body)) return;
  if (cache != NULL) memcpy(cache, body, (size_t)length + 1U);
  RR2_Send(body);
}

static int RR2_IsPrintableBody(const char *body)
{
  if (body[0] == '\0' || body[0] == ' ') return 0;
  for (size_t i = 0U; body[i] != '\0'; ++i) {
    unsigned char byte = (unsigned char)body[i];
    if ((byte < 33U && byte != ' ') || byte > 126U) return 0;
    if (byte == ' ' && (body[i + 1U] == '\0' || body[i + 1U] == ' ')) return 0;
  }
  return 1;
}

static UartRr2Capability RR2_ParseCapability(const char *text)
{
  if (strcmp(text, "MISSION") == 0) return UART_RR2_CAP_MISSION;
  if (strcmp(text, "ACTUATOR_DEBUG_ENCODER") == 0) return UART_RR2_CAP_ENCODER;
  if (strcmp(text, "ACTUATOR_DEBUG_DC") == 0) return UART_RR2_CAP_DC;
  if (strcmp(text, "ACTUATOR_DEBUG_PUMP") == 0) return UART_RR2_CAP_PUMP;
  if (strcmp(text, "ACTUATOR_DEBUG_ALL") == 0) return UART_RR2_CAP_ALL;
  if (strcmp(text, "SENSOR_DEBUG") == 0) return UART_RR2_CAP_SENSOR;
  return UART_RR2_CAP_NONE;
}

static const char *RR2_CapabilityName(UartRr2Capability capability)
{
  switch (capability) {
    case UART_RR2_CAP_MISSION: return "MISSION";
    case UART_RR2_CAP_ENCODER: return "ACTUATOR_DEBUG_ENCODER";
    case UART_RR2_CAP_DC: return "ACTUATOR_DEBUG_DC";
    case UART_RR2_CAP_PUMP: return "ACTUATOR_DEBUG_PUMP";
    case UART_RR2_CAP_ALL: return "ACTUATOR_DEBUG_ALL";
    case UART_RR2_CAP_SENSOR: return "SENSOR_DEBUG";
    default: return "NONE";
  }
}

static uint32_t RR2_NextTransmitSequence(void)
{
  if (rr2.tx_sequence == UINT32_MAX) {
    RR2_Fault();
    return 0U;
  }
  return ++rr2.tx_sequence;
}

int UART_QueueEvent(const char *event_code, const char *payload)
{
  char body[RR2_LINE_CAP];
  size_t code_length;
  size_t payload_length = payload == NULL ? 0U : strlen(payload);
  if (event_code == NULL || event_code[0] == '\0') return 0;
  code_length = strlen(event_code);
  if (code_length >= sizeof(rr2.events[0].code) ||
      payload_length >= sizeof(rr2.events[0].payload)) {
    if (rr2.active != 0U) RR2_Fault();
    return 0;
  }
  if (payload_length == 0U) payload = "";
  /* Events are also used by the legacy ultrasonic path. Preserve its text
   * interface when no RR2 session is negotiated. */
  if (rr2.active == 0U) {
    char legacy[64];
    int length = snprintf(legacy, sizeof(legacy), "%s\r\n", event_code);
    if (length > 0 && (size_t)length < sizeof(legacy)) UART_SendText(legacy);
    return 1;
  }
  if (snprintf(body, sizeof(body), "RR2 EVENT %s 1 %s%s%s",
               rr2.session, event_code, payload_length ? " " : "", payload) >=
      (int)sizeof(body)) {
    RR2_Fault();
    return 0;
  }
  if (!RR2_IsPrintableBody(body)) {
    RR2_Fault();
    return 0;
  }
  if (rr2.event_count >= RR2_EVENT_CAP) {
    RR2_Fault();
    return 0;
  }
  uint8_t slot = (uint8_t)((rr2.event_head + rr2.event_count) % RR2_EVENT_CAP);
  memset(&rr2.events[slot], 0, sizeof(rr2.events[slot]));
  memcpy(rr2.events[slot].code, event_code, code_length + 1U);
  if (payload_length != 0U)
    memcpy(rr2.events[slot].payload, payload, payload_length + 1U);
  rr2.event_count++;
  return 1;
}

void UART_QueueStatus(const char *status_key, const char *values)
{
  char body[RR2_LINE_CAP];
  size_t key_length, values_length;
  int slot = -1;
  if (rr2.active == 0U || status_key == NULL || values == NULL) return;
  key_length = strlen(status_key);
  values_length = strlen(values);
  if (key_length == 0U || key_length >= sizeof(rr2.statuses[0].key) ||
      values_length == 0U || values_length >= sizeof(rr2.statuses[0].values) ||
      snprintf(body, sizeof(body), "RR2 STATUS %s 4294967295 %s %s",
               rr2.session, status_key, values) >= (int)sizeof(body) ||
      !RR2_IsPrintableBody(body)) {
    RR2_Fault();
    return;
  }
  for (unsigned i = 0U; i < RR2_STATUS_CAP; ++i) {
    if (rr2.statuses[i].pending && strcmp(rr2.statuses[i].key, status_key) == 0) {
      slot = (int)i;
      break;
    }
    if (!rr2.statuses[i].pending && slot < 0) slot = (int)i;
  }
  if (slot < 0) {
    slot = (int)(rr2.status_cursor++ % RR2_STATUS_CAP);
  }
  memcpy(rr2.statuses[slot].key, status_key, key_length + 1U);
  memcpy(rr2.statuses[slot].values, values, values_length + 1U);
  rr2.statuses[slot].pending = 1U;
}

static void RR2_ServiceEventsAndStatus(void)
{
  if (rr2.active == 0U) return;
  if (rr2.event_count != 0U) {
    Rr2Event *event = &rr2.events[rr2.event_head];
    uint32_t now = HAL_GetTick();
    if (event->attempts == 0U ||
        (uint32_t)(now - event->last_sent_tick) >= RR2_EVENT_RETRY_MS) {
      if (event->attempts >= RR2_EVENT_MAX_ATTEMPTS) {
        /* 호스트가 이벤트를 받지 않는 것도 링크가 끊긴 것과 같게 판정한다.
         * 예: 디버그 명령 시간 초과로 보낸 DEBUG_TIMEOUT을 이미 떠난 호스트가 받지 못한 경우. */
        if (RR2_CanEndWithoutFault()) RR2_EndSessionWithoutFault();
        else RR2_Fault();
        return;
      }
      if (event->sequence == 0U) {
        event->sequence = RR2_NextTransmitSequence();
        if (event->sequence == 0U) return;
      }
      char body[RR2_LINE_CAP];
      int length = snprintf(body, sizeof(body), "RR2 EVENT %s %lu %s%s%s",
          rr2.session, (unsigned long)event->sequence, event->code,
          event->payload[0] ? " " : "", event->payload);
      if (length <= 0 || (size_t)length >= sizeof(body) || !RR2_IsPrintableBody(body)) {
        RR2_Fault();
        return;
      }
      RR2_Send(body);
      event->attempts++;
      event->last_sent_tick = now;
      return;
    }
  }
  for (unsigned offset = 0U; offset < RR2_STATUS_CAP; ++offset) {
    unsigned slot = (unsigned)(rr2.status_cursor + offset) % RR2_STATUS_CAP;
    Rr2Status *status = &rr2.statuses[slot];
    if (!status->pending) continue;
    uint32_t sequence = RR2_NextTransmitSequence();
    if (sequence == 0U) return;
    char body[RR2_LINE_CAP];
    int length = snprintf(body, sizeof(body), "RR2 STATUS %s %lu %s %s",
        rr2.session, (unsigned long)sequence, status->key, status->values);
    if (length <= 0 || (size_t)length >= sizeof(body) || !RR2_IsPrintableBody(body)) {
      RR2_Fault();
      return;
    }
    status->pending = 0U;
    rr2.status_cursor = (uint8_t)((slot + 1U) % RR2_STATUS_CAP);
    RR2_Send(body);
    return;
  }
}

static void UART_DrainTx(void)
{
  unsigned dequeued = 0U;
  while (dequeued < UART_TX_BURST) {
    UartTxItem item;
    uint32_t generation;
    if (RR2_ServicePendingFault()) return;
    if (!UART_TakeTxItem(&item, &generation)) break;
    ++dequeued;
    /* Handle a USART fault delivered as IRQs are restored after dequeue before
     * a local copy can reach HAL_UART_Transmit. */
    if (RR2_ServicePendingFault()) return;
    if (generation != uart_tx_generation) continue;
    if (HAL_UART_Transmit(&huart2, (const uint8_t *)item.bytes,
                          item.length, 20U) != HAL_OK) {
      UART_QueueFailSafeAllOff();
      if (rr2.active != 0U) RR2_Fault();
      return;
    }
  }
}

static unsigned RR2_SplitFields(char *body, char **fields, unsigned capacity)
{
  unsigned count = 1U;
  fields[0] = body;
  for (char *cursor = body; *cursor != '\0'; ++cursor) {
    if (*cursor == ' ') {
      *cursor = '\0';
      if (count >= capacity) return capacity + 1U;
      fields[count++] = cursor + 1;
    }
  }
  return count;
}

static int RR2_ParseUnsigned(const char *text, uint32_t limit, uint32_t *value)
{
  uint32_t parsed = 0U;
  if (text == NULL || *text == '\0') return 0;
  while (*text != '\0') {
    uint32_t digit;
    if (*text < '0' || *text > '9') return 0;
    digit = (uint32_t)(*text++ - '0');
    if (digit > limit || parsed > (limit - digit) / 10U) return 0;
    parsed = parsed * 10U + digit;
  }
  *value = parsed;
  return 1;
}

static UartCommand RR2_FindMissionAccessory(const char *opcode)
{
  for (UartCommand command = UART_COMMAND_FRONT_CLEANER_PWM_12_5;
       command < UART_COMMAND_COUNT;
       command = (UartCommand)((int)command + 1)) {
    const UartText *entry = &uart_command_table[command];
    size_t length = entry->length >= 2U ? entry->length - 2U : 0U;
    if (length != 0U && strlen(opcode) == length &&
        memcmp(entry->text, opcode, length) == 0) return command;
  }
  return UART_COMMAND_NONE;
}

static int RR2_IsAccessoryOff(UartCommand command)
{
  return command == UART_COMMAND_FRONT_CLEANER_OFF ||
         command == UART_COMMAND_FRONT_PUMP_OFF ||
         command == UART_COMMAND_SIDE_CLEANER_OFF ||
         command == UART_COMMAND_SIDE_PUMP_OFF;
}

static int RR2_IsMissionAccessory(UartCommand command)
{
  return command >= UART_COMMAND_FRONT_CLEANER_PWM_12_5 &&
         command < UART_COMMAND_COUNT;
}

static int RR2_AcknowledgeEvent(const char *session, uint32_t sequence,
                                const char *result)
{
  if (rr2.active == 0U || strcmp(session, rr2.session) != 0 ||
      rr2.event_count == 0U) return 1;
  Rr2Event *event = &rr2.events[rr2.event_head];
  if (event->sequence != sequence) return 1; /* duplicate/stale EVENT_ACK */
  if (strcmp(result, "ACCEPTED") != 0) {
    RR2_Fault();
    return 0;
  }
  memset(event, 0, sizeof(*event));
  rr2.event_head = (uint8_t)((rr2.event_head + 1U) % RR2_EVENT_CAP);
  rr2.event_count--;
  return 1;
}

static void RR2_ProcessLine(char *line)
{
  char request[RR2_LINE_CAP];
  char *crc_separator = strrchr(line, ' ');
  char *tokens[10];
  unsigned count;
  uint16_t supplied_crc = 0U;
  uint32_t sequence;
  uint32_t *lane_sequence;
  char *lane_request;
  char *lane_reply;
  const char *opcode;
  const char *kind;
  if (crc_separator == NULL || strlen(crc_separator + 1) != 4U) return;
  *crc_separator = '\0';
  if (!RR2_IsPrintableBody(line)) return;
  memcpy(request, line, strlen(line) + 1U);
  /* CRC is parsed only after the body grammar has been checked. */
  const char *crc_text = crc_separator + 1;
  for (unsigned i = 0U; i < 4U; ++i) {
    char hex = crc_text[i];
    uint16_t digit;
    if (hex >= '0' && hex <= '9') digit = (uint16_t)(hex - '0');
    else if (hex >= 'A' && hex <= 'F') digit = (uint16_t)(hex - 'A' + 10);
    else if (hex >= 'a' && hex <= 'f') digit = (uint16_t)(hex - 'a' + 10);
    else return;
    supplied_crc = (uint16_t)((supplied_crc << 4) | digit);
  }
  if (RR2_Crc(line) != supplied_crc) return;
  count = RR2_SplitFields(line, tokens, (unsigned)(sizeof(tokens) / sizeof(tokens[0])));
  if (count > (unsigned)(sizeof(tokens) / sizeof(tokens[0])) || count < 3U ||
      strcmp(tokens[0], "RR2") != 0) return;
  kind = tokens[1];
  if (strcmp(kind, "HELLO") == 0) {
    char body[RR2_LINE_CAP];
    UartRr2Capability capability;
    if (count != 4U || !RR2_ValidSession(tokens[2]) ||
        (capability = RR2_ParseCapability(tokens[3])) == UART_RR2_CAP_NONE) return;
    if (rr2.active != 0U) {
      if (strcmp(rr2.session, tokens[2]) != 0 || rr2.capability != capability ||
          Mission_GetState() == MISSION_FAULT) return;
    } else {
      if (Mission_GetState() != MISSION_IDLE || Mission_IsTransitionPending()) return;
      /* No START already received as legacy bytes may survive the handshake. */
      UART_ClearPending();
      for (UartCommand cmd = UART_COMMAND_START; cmd < UART_COMMAND_COUNT;
           cmd = (UartCommand)((int)cmd + 1)) uart.match_index[cmd] = 0U;
      CleaningActuators_StopAll();
      memcpy(rr2.session, tokens[2], 33U);
      rr2.last_cmd_sequence = 0U;
      rr2.last_ping_sequence = 0U;
      rr2.tx_sequence = 0U;
      rr2.last_cmd_request[0] = rr2.last_cmd_reply[0] = '\0';
      rr2.last_ping_request[0] = rr2.last_ping_reply[0] = '\0';
      rr2.active = 1U;
      rr2.capability = capability;
      rr2.last_seen_tick = HAL_GetTick();
      rr2.event_head = rr2.event_count = 0U;
      memset(rr2.events, 0, sizeof(rr2.events));
      memset(rr2.statuses, 0, sizeof(rr2.statuses));
      UART_ClearTxQueue();
      if (capability != UART_RR2_CAP_MISSION &&
          !Debug_RR2Begin(rr2.session, capability)) {
        rr2.active = 0U;
        rr2.capability = UART_RR2_CAP_NONE;
        return;
      }
    }
    /* A duplicate HELLO may replay READY, but only PING or an accepted mission
     * command refreshes liveness after the session has been opened. */
    snprintf(body, sizeof(body), "RR2 READY %s %s %s", rr2.session,
             UART_RR2_PROTOCOL_VERSION, RR2_CapabilityName(rr2.capability));
    RR2_Send(body);
    return;
  }
  if (rr2.active == 0U || count < 4U || !RR2_ValidSession(tokens[2])) return;
  if (strcmp(kind, "EVENT_ACK") == 0) {
    if (count != 5U || !RR2_ParseSequence(tokens[3], &sequence)) return;
    (void)RR2_AcknowledgeEvent(tokens[2], sequence, tokens[4]);
    return;
  }
  if (!RR2_ParseSequence(tokens[3], &sequence)) return;
  if (strcmp(tokens[2], rr2.session) != 0) return;
  if (strcmp(kind, "PING") == 0) {
    opcode = "PING";
    lane_sequence = &rr2.last_ping_sequence;
    lane_request = rr2.last_ping_request;
    lane_reply = rr2.last_ping_reply;
  } else if (strcmp(kind, "CMD") == 0) {
    opcode = count >= 5U ? tokens[4] : "UNKNOWN";
    if (strlen(opcode) > 28U) opcode = "UNKNOWN";
    lane_sequence = &rr2.last_cmd_sequence;
    lane_request = rr2.last_cmd_request;
    lane_reply = rr2.last_cmd_reply;
  } else return;
  if (sequence < *lane_sequence) {
    RR2_Reply("NACK", rr2.session, sequence, opcode, "STALE", NULL);
    return;
  }
  if (sequence == *lane_sequence) {
    if (strcmp(request, lane_request) == 0 && lane_reply[0] != '\0') {
      if (strncmp(lane_reply, "RR2 ACK ", 8U) == 0)
        rr2.last_seen_tick = HAL_GetTick();
      RR2_Send(lane_reply);
    } else RR2_Reply("NACK", rr2.session, sequence, opcode, "CONFLICT", NULL);
    return;
  }
  *lane_sequence = sequence;
  memcpy(lane_request, request, strlen(request) + 1U);
  lane_reply[0] = '\0';
  if (strcmp(kind, "CMD") == 0 && strcmp(opcode, "ESTOP") == 0) {
    if (count != 5U) {
      RR2_Reply("NACK", rr2.session, sequence, opcode, "ARGS", lane_reply);
    } else if (rr2.capability != UART_RR2_CAP_MISSION) {
      RR2_Reply("NACK", rr2.session, sequence, opcode, "CAPABILITY", lane_reply);
    } else {
      /* Stop before queuing the ACK. Keep the session and command reply cache
       * so a lost ACK can be replayed while FAULT remains latched. */
      Robot_EmergencyStop();
      UART_ClearTxQueue();
      rr2.event_head = rr2.event_count = 0U;
      memset(rr2.statuses, 0, sizeof(rr2.statuses));
      RR2_Reply("ACK", rr2.session, sequence, opcode, "ACCEPTED", lane_reply);
    }
  } else if (Mission_GetState() == MISSION_FAULT) {
    RR2_Reply("NACK", rr2.session, sequence, opcode, "FAULT", lane_reply);
  } else if (strcmp(kind, "PING") == 0) {
    if (count == 4U) {
      rr2.last_seen_tick = HAL_GetTick();
      RR2_Reply("ACK", rr2.session, sequence, opcode, "ACCEPTED", lane_reply);
    } else RR2_Reply("NACK", rr2.session, sequence, opcode, "ARGS", lane_reply);
  } else if (strcmp(opcode, "PREPARE") == 0 ||
             strcmp(opcode, "CANCEL_PREPARE") == 0) {
    /* PREPARE는 선택 인자 하나로 다음 임무의 단계별 구간 수(한 자리, 1..ENCODER_SECTION_LIMIT)를 받는다.
     * 인자가 없으면 기본 구간 수다. 이 인자를 모르는 이전 펌웨어는 ARGS로 거부하므로 구간 수가 바뀐
     * 채로 주행하지 않는다. */
    int prepare = strcmp(opcode, "PREPARE") == 0;
    int sections = ENCODER_SECTION_LIMIT;
    int args_ok = count == 5U;
    if (prepare && count == 6U && tokens[5][0] >= '1' &&
        tokens[5][0] <= (char)('0' + ENCODER_SECTION_LIMIT) && tokens[5][1] == '\0') {
      sections = tokens[5][0] - '0';
      args_ok = 1;
    }
    if (rr2.capability != UART_RR2_CAP_MISSION || !args_ok) {
      RR2_Reply("NACK", rr2.session, sequence, opcode,
                rr2.capability == UART_RR2_CAP_MISSION ? "ARGS" : "CAPABILITY",
                lane_reply);
    } else if (Mission_ApplyReliablePrepare(prepare, sections)) {
      rr2.last_seen_tick = HAL_GetTick();
      RR2_Reply("ACK", rr2.session, sequence, opcode, "ACCEPTED", lane_reply);
    } else RR2_Reply("NACK", rr2.session, sequence, opcode, "STATE", lane_reply);
  } else if (strcmp(opcode, "START") == 0 || strcmp(opcode, "CAPTURE_OK") == 0) {
    if (rr2.capability != UART_RR2_CAP_MISSION || count != 5U) {
      RR2_Reply("NACK", rr2.session, sequence, opcode,
                rr2.capability == UART_RR2_CAP_MISSION ? "ARGS" : "CAPABILITY",
                lane_reply);
    } else if (Mission_ApplyReliableCommand(strcmp(opcode, "START") == 0)) {
      rr2.last_seen_tick = HAL_GetTick();
      Debug_RR2Heartbeat();
      RR2_Reply("ACK", rr2.session, sequence, opcode, "ACCEPTED", lane_reply);
    } else RR2_Reply("NACK", rr2.session, sequence, opcode, "STATE", lane_reply);
  } else if (RR2_IsMissionAccessory(RR2_FindMissionAccessory(opcode))) {
    UartCommand command = RR2_FindMissionAccessory(opcode);
    if (rr2.capability != UART_RR2_CAP_MISSION) {
      RR2_Reply("NACK", rr2.session, sequence, opcode, "CAPABILITY", lane_reply);
    } else if (count != 5U) {
      RR2_Fault();
      RR2_Reply("NACK", rr2.session, sequence, opcode, "ARGS", lane_reply);
    } else if (!RR2_IsAccessoryOff(command) &&
               Mission_GetState() != REALTIME_CLEANING) {
      RR2_Reply("NACK", rr2.session, sequence, opcode, "STATE", lane_reply);
    } else if (UART_QueueCommand(command)) {
      rr2.last_seen_tick = HAL_GetTick();
      RR2_Reply("ACK", rr2.session, sequence, opcode, "ACCEPTED", lane_reply);
    } else RR2_Reply("NACK", rr2.session, sequence, opcode, "SAFETY", lane_reply);
  } else if (strcmp(opcode, "STOP") == 0) {
    if (count != 5U || rr2.capability == UART_RR2_CAP_MISSION) {
      RR2_Reply("NACK", rr2.session, sequence, opcode, "CAPABILITY", lane_reply);
    } else {
      Debug_RR2Stop();
      rr2.last_seen_tick = HAL_GetTick();
      RR2_Reply("ACK", rr2.session, sequence, opcode, "ACCEPTED", lane_reply);
    }
  } else if (strcmp(opcode, "SET") == 0) {
    uint32_t percent, mask;
    int direction;
    if (count != 8U) {
      RR2_Fault();
      RR2_Reply("NACK", rr2.session, sequence, opcode, "ARGS", lane_reply);
    } else if (rr2.capability == UART_RR2_CAP_MISSION ||
               rr2.capability == UART_RR2_CAP_SENSOR) {
      RR2_Reply("NACK", rr2.session, sequence, opcode, "CAPABILITY", lane_reply);
    } else {
      if (strcmp(tokens[5], "-1") == 0) direction = -1;
      else if (strcmp(tokens[5], "0") == 0) direction = 0;
      else if (strcmp(tokens[5], "1") == 0) direction = 1;
      else direction = 99;
      if (direction == 99 || !RR2_ParseUnsigned(tokens[6], 10000U, &percent) ||
          !RR2_ParseUnsigned(tokens[7], 15U, &mask)) {
        RR2_Fault();
        RR2_Reply("NACK", rr2.session, sequence, opcode, "ARGS", lane_reply);
      } else if (!Debug_RR2Set(sequence, direction, percent, mask)) {
        RR2_Reply("NACK", rr2.session, sequence, opcode, "CAPABILITY", lane_reply);
      } else {
        rr2.last_seen_tick = HAL_GetTick();
        RR2_Reply("ACK", rr2.session, sequence, opcode, "ACCEPTED", lane_reply);
      }
    }
  } else if (strcmp(opcode, "LED") == 0) {
    uint32_t red, green, blue;
    if (count != 9U) {
      RR2_Fault();
      RR2_Reply("NACK", rr2.session, sequence, opcode, "ARGS", lane_reply);
    } else if (rr2.capability != UART_RR2_CAP_SENSOR) {
      RR2_Reply("NACK", rr2.session, sequence, opcode, "CAPABILITY", lane_reply);
    } else if (!RR2_ParseUnsigned(tokens[6], 255U, &red) ||
               !RR2_ParseUnsigned(tokens[7], 255U, &green) ||
               !RR2_ParseUnsigned(tokens[8], 255U, &blue) ||
               !Debug_RR2Led(sequence, tokens[5], red, green, blue)) {
      RR2_Fault();
      RR2_Reply("NACK", rr2.session, sequence, opcode, "ARGS", lane_reply);
    } else {
      rr2.last_seen_tick = HAL_GetTick();
      RR2_Reply("ACK", rr2.session, sequence, opcode, "ACCEPTED", lane_reply);
    }
  } else if (count != 5U) {
    RR2_Reply("NACK", rr2.session, sequence, opcode, "ARGS", lane_reply);
  } else {
    RR2_Reply("NACK", rr2.session, sequence, opcode,
              rr2.capability == UART_RR2_CAP_MISSION ? "COMMAND" : "CAPABILITY",
              lane_reply);
  }
}

static int RR2_ParseByte(uint8_t byte)
{
  if (!rr2.receiving) {
    if (byte != 'R') return 0;
    rr2.receiving = 1U;
    rr2.length = 0U;
  }
  if (byte == '\n') {
    uint8_t length = rr2.length;
    rr2.receiving = 0U;
    rr2.length = 0U;
    if (length == 0U || rr2.rx[length - 1U] != '\r') {
      if (rr2.active != 0U) RR2_RequestFault();
      return 1;
    }
    --length;
    rr2.rx[length] = '\0';
    if (rr2.pending_count >= 2U) {
      if (rr2.active != 0U) RR2_RequestFault();
    } else {
      memcpy(rr2.line[rr2.write_slot], rr2.rx, (size_t)length + 1U);
      rr2.write_slot ^= 1U;
      rr2.pending_count++;
    }
    return 1;
  }
  if ((byte != '\r' && (byte < 32U || byte > 126U)) ||
      rr2.length >= RR2_LINE_CAP - 1U) {
    rr2.receiving = 0U;
    rr2.length = 0U;
    if (rr2.active != 0U) RR2_RequestFault();
    return 1;
  }
  rr2.rx[rr2.length++] = (char)byte;
  return 1;
}

static int UART_QueueCommand(UartCommand command)
{
  /* legacy 수신 명령은 FIFO가 아니라 비트 집합에 저장한다. 같은 명령 반복은 합쳐지고 PWM 두 단계는 최신값이 남는다.
   * OFF가 이미 대기 중이면 해당 ON을 버리며, 시험 모드 전환은 이전 구동 명령을 비운다. */
  uint32_t command_bit;
  /* [legacy 통신 감시] 끝까지 일치한 legacy 명령 한 줄이면 Jetson이 살아 있다고 본다.
   * LINK_OK는 이 시각 갱신만 하며 명령함·상태기계에는 들어가지 않는다. */
  uart.legacy_seen_tick = HAL_GetTick();
  if (command == UART_COMMAND_LINK_OK) return 1;
  if (Ultrasonic_IsHeld()) {
    switch(command) {
      case UART_COMMAND_FRONT_CLEANER_OFF: Front_Cleaner_Stop(); break;
      case UART_COMMAND_FRONT_PUMP_OFF: Front_Pump_Stop(); break;
      case UART_COMMAND_SIDE_CLEANER_OFF: Side_Cleaner_Stop(); break;
      case UART_COMMAND_SIDE_PUMP_OFF: Side_Pump_Stop(); break;
      /* ISR에서는 출력만 즉시 끄고, IDLE 전이는 main의 Mission_Update가 비트를 꺼내 처리한다. */
      case UART_COMMAND_ACTUATOR_TEST_STOP: Motor_Stop(); CleaningActuators_StopAll(); break;
      case UART_COMMAND_MISSION_CANCEL_PREPARE: break;
      case UART_COMMAND_START: case UART_COMMAND_CAPTURE_OK: break;
      default: return 0;
    }
  }

  /* 모드 사이 정지 중 들어온 ON/START/촬영 ACK는 다음 모드까지 남기지 않는다.
   * OFF는 계속 접수하므로 1초 대기 중에도 안전 해제 요청을 처리할 수 있다. */
  if ((Mission_IsTransitionPending() != 0) &&
      (command != UART_COMMAND_FRONT_CLEANER_OFF) &&
      (command != UART_COMMAND_FRONT_PUMP_OFF) &&
      (command != UART_COMMAND_SIDE_CLEANER_OFF) &&
      (command != UART_COMMAND_SIDE_PUMP_OFF))
  {
    return 0;
  }

  if ((command <= UART_COMMAND_NONE) || (command >= UART_COMMAND_COUNT))
  {
    return 0;
  }

  if ((((command == UART_COMMAND_FRONT_CLEANER_PWM_12_5) ||
        (command == UART_COMMAND_FRONT_CLEANER_PWM_25)) &&
       ((uart.pending_commands &
         (1UL << UART_COMMAND_FRONT_CLEANER_OFF)) != 0U)) ||
      ((command == UART_COMMAND_FRONT_PUMP_ON) &&
       ((uart.pending_commands &
         (1UL << UART_COMMAND_FRONT_PUMP_OFF)) != 0U)) ||
      (((command == UART_COMMAND_SIDE_CLEANER_PWM_12_5) ||
        (command == UART_COMMAND_SIDE_CLEANER_PWM_25)) &&
       ((uart.pending_commands &
         (1UL << UART_COMMAND_SIDE_CLEANER_OFF)) != 0U)) ||
      ((command == UART_COMMAND_SIDE_PUMP_ON) &&
       ((uart.pending_commands &
         (1UL << UART_COMMAND_SIDE_PUMP_OFF)) != 0U)))
  {
    return 0;
  }

  switch (command)
  {
    case UART_COMMAND_ACTUATOR_TEST_START:
    case UART_COMMAND_ACTUATOR_TEST_STOP:
      /* Test mode transitions supersede queued motion/actuator commands. */
      uart.pending_commands = 0U;
      break;

    case UART_COMMAND_MISSION_PREPARE:
    case UART_COMMAND_MISSION_CANCEL_PREPARE:
    case UART_COMMAND_FRONT_PUMP_ON:
    case UART_COMMAND_SIDE_PUMP_ON:
    case UART_COMMAND_START:
    case UART_COMMAND_CAPTURE_OK:
      break;

    case UART_COMMAND_FRONT_CLEANER_OFF:
      uart.pending_commands &=
          ~(1UL << UART_COMMAND_FRONT_CLEANER_PWM_12_5);
      uart.pending_commands &=
          ~(1UL << UART_COMMAND_FRONT_CLEANER_PWM_25);
      break;

    case UART_COMMAND_FRONT_CLEANER_PWM_12_5:
    case UART_COMMAND_FRONT_CLEANER_PWM_25:
      uart.pending_commands &=
          ~(1UL << UART_COMMAND_FRONT_CLEANER_PWM_12_5);
      uart.pending_commands &=
          ~(1UL << UART_COMMAND_FRONT_CLEANER_PWM_25);
      break;

    case UART_COMMAND_FRONT_PUMP_OFF:
      uart.pending_commands &= ~(1UL << UART_COMMAND_FRONT_PUMP_ON);
      break;

    case UART_COMMAND_SIDE_CLEANER_OFF:
      uart.pending_commands &=
          ~(1UL << UART_COMMAND_SIDE_CLEANER_PWM_12_5);
      uart.pending_commands &=
          ~(1UL << UART_COMMAND_SIDE_CLEANER_PWM_25);
      break;

    case UART_COMMAND_SIDE_CLEANER_PWM_12_5:
    case UART_COMMAND_SIDE_CLEANER_PWM_25:
      uart.pending_commands &=
          ~(1UL << UART_COMMAND_SIDE_CLEANER_PWM_12_5);
      uart.pending_commands &=
          ~(1UL << UART_COMMAND_SIDE_CLEANER_PWM_25);
      break;

    case UART_COMMAND_SIDE_PUMP_OFF:
      uart.pending_commands &= ~(1UL << UART_COMMAND_SIDE_PUMP_ON);
      break;

    case UART_COMMAND_NONE:
    case UART_COMMAND_COUNT:
    default:
      return 0;
  }

  command_bit = 1UL << (uint32_t)command;
  uart.pending_commands |= command_bit;
  return 1;
}

static UartCommand UART_TakePendingCommandMasked(uint32_t allowed)
{
  /* main이 한 번에 하나씩 꺼내며 아래 분기 순서가 우선순위다(시험 STOP/장치 OFF → 시작·ACK → ON).
   * 비트를 읽고 지우는 동안 ISR이 새 명령을 기록하지 못하게 한다. 수신 시간 순서 그대로 실행하는 큐는 아니다.
   * allowed에 없는 비트는 꺼내지 않고 그대로 남긴다. */
  uint32_t primask;
  uint32_t command_bit = 0U;
  uint32_t pending;
  UartCommand command = UART_COMMAND_NONE;

  primask = __get_PRIMASK();
  __disable_irq();
  pending = uart.pending_commands & allowed;

  if ((pending &
       (1UL << UART_COMMAND_ACTUATOR_TEST_STOP)) != 0U)
  {
    command = UART_COMMAND_ACTUATOR_TEST_STOP;
  }
  else if ((pending &
       (1UL << UART_COMMAND_FRONT_CLEANER_OFF)) != 0U)
  {
    command = UART_COMMAND_FRONT_CLEANER_OFF;
  }
  else if ((pending &
            (1UL << UART_COMMAND_FRONT_PUMP_OFF)) != 0U)
  {
    command = UART_COMMAND_FRONT_PUMP_OFF;
  }
  else if ((pending &
            (1UL << UART_COMMAND_SIDE_CLEANER_OFF)) != 0U)
  {
    command = UART_COMMAND_SIDE_CLEANER_OFF;
  }
  else if ((pending &
            (1UL << UART_COMMAND_SIDE_PUMP_OFF)) != 0U)
  {
    command = UART_COMMAND_SIDE_PUMP_OFF;
  }
  else if ((pending &
            (1UL << UART_COMMAND_MISSION_CANCEL_PREPARE)) != 0U)
  {
    command = UART_COMMAND_MISSION_CANCEL_PREPARE;
  }
  else if ((pending &
            (1UL << UART_COMMAND_MISSION_PREPARE)) != 0U)
  {
    command = UART_COMMAND_MISSION_PREPARE;
  }
  else if ((pending &
            (1UL << UART_COMMAND_ACTUATOR_TEST_START)) != 0U)
  {
    command = UART_COMMAND_ACTUATOR_TEST_START;
  }
  else if ((pending &
            (1UL << UART_COMMAND_CAPTURE_OK)) != 0U)
  {
    command = UART_COMMAND_CAPTURE_OK;
  }
  else if ((pending &
            (1UL << UART_COMMAND_START)) != 0U)
  {
    command = UART_COMMAND_START;
  }
  else if ((pending &
            (1UL << UART_COMMAND_FRONT_CLEANER_PWM_25)) != 0U)
  {
    command = UART_COMMAND_FRONT_CLEANER_PWM_25;
  }
  else if ((pending &
            (1UL << UART_COMMAND_FRONT_CLEANER_PWM_12_5)) != 0U)
  {
    command = UART_COMMAND_FRONT_CLEANER_PWM_12_5;
  }
  else if ((pending &
            (1UL << UART_COMMAND_FRONT_PUMP_ON)) != 0U)
  {
    command = UART_COMMAND_FRONT_PUMP_ON;
  }
  else if ((pending &
            (1UL << UART_COMMAND_SIDE_CLEANER_PWM_25)) != 0U)
  {
    command = UART_COMMAND_SIDE_CLEANER_PWM_25;
  }
  else if ((pending &
            (1UL << UART_COMMAND_SIDE_CLEANER_PWM_12_5)) != 0U)
  {
    command = UART_COMMAND_SIDE_CLEANER_PWM_12_5;
  }
  else if ((pending &
            (1UL << UART_COMMAND_SIDE_PUMP_ON)) != 0U)
  {
    command = UART_COMMAND_SIDE_PUMP_ON;
  }

  if (command != UART_COMMAND_NONE)
  {
    command_bit = 1UL << (uint32_t)command;
    uart.pending_commands &= ~command_bit;
  }

  if (primask == 0U)
  {
    __enable_irq();
  }

  return command;
}

UartCommand UART_TakePendingCommand(void)
{
  return UART_TakePendingCommandMasked(UINT32_MAX);
}

UartCommand UART_TakeHeldStopCommand(void)
{
  /* 초음파 hold 중에도 main이 처리해야 하는 시험 종료·OFF만 꺼낸다. START/ACK/ON은 해제 후 처리한다. */
  return UART_TakePendingCommandMasked((1UL << UART_COMMAND_ACTUATOR_TEST_STOP) |
                                       (1UL << UART_COMMAND_FRONT_CLEANER_OFF) |
                                       (1UL << UART_COMMAND_FRONT_PUMP_OFF) |
                                       (1UL << UART_COMMAND_SIDE_CLEANER_OFF) |
                                       (1UL << UART_COMMAND_SIDE_PUMP_OFF));
}

static void UART_ParseByte(uint8_t byte)
{
  /* DBG 프레임은 legacy 문자열 매칭을 통과하지 않도록 먼저 분리한다.
   * legacy는 표의 문자 일치를 바이트마다 누적하고 현재 상태에서 허용하지 않는 START/ACK/시험 명령은 매칭을 초기화한다.
   * START는 IDLE, CAPTURE_OK는 WAIT_CAPTURE_ACK에서만 접수된다. */
  UartCommand command;
  uint8_t rr2_candidate = rr2.receiving || (rr2_line_start && byte == 'R');
  rr2_line_start = (byte == '\n');

  if (rr2_candidate && RR2_ParseByte(byte) != 0) {
    for (command = UART_COMMAND_START; command < UART_COMMAND_COUNT;
         command = (UartCommand)((int)command + 1))
      uart.match_index[command] = 0U;
    return;
  }

  /* An active RR2 capability owns the parser. Never allow raw legacy
   * accessories, ACTUATOR_TEST_START, or DBG1 to bypass its session gate. */
  if (rr2.active != 0U) {
    Debug_ResetRx();
    for (command = UART_COMMAND_START; command < UART_COMMAND_COUNT;
         command = (UartCommand)((int)command + 1))
      uart.match_index[command] = 0U;
    return;
  }

  /* Versioned debug lines never pass through legacy substring matchers. */
  if (Debug_ParseByte(byte) != 0)
  {
    for (command = UART_COMMAND_START; command < UART_COMMAND_COUNT;
         command = (UartCommand)((int)command + 1))
      uart.match_index[command] = 0U;
    return;
  }

  for (command = UART_COMMAND_START;
       command < UART_COMMAND_COUNT;
       command = (UartCommand)((int)command + 1))
  {
    const UartText *definition = &uart_command_table[command];
    uint8_t match_index = uart.match_index[command];

    if (((command == UART_COMMAND_START) &&
         (Mission_GetState() != MISSION_IDLE || rr2.active != 0U)) ||
        ((command == UART_COMMAND_CAPTURE_OK) &&
         (Mission_GetState() != WAIT_CAPTURE_ACK || rr2.active != 0U)) ||
        ((command == UART_COMMAND_ACTUATOR_TEST_START) &&
         (Mission_GetState() != MISSION_IDLE)) ||
        ((command == UART_COMMAND_ACTUATOR_TEST_STOP) &&
         (Mission_GetState() != ACTUATOR_TEST)))
    {
      uart.match_index[command] = 0U;
      continue;
    }

    if (byte == (uint8_t)definition->text[match_index])
    {
      match_index++;
      if (match_index >= definition->length)
      {
        match_index = 0U;
        UART_QueueCommand(command);
      }
    }
    else if (byte == (uint8_t)definition->text[0])
    {
      match_index = 1U;
    }
    else
    {
      match_index = 0U;
    }

    uart.match_index[command] = match_index;
  }
}

static void UART_ResetMatchers(void)
{
  UartCommand command;

  Debug_ResetRx();
  rr2.receiving = 0U;
  rr2.length = 0U;
  rr2_line_start = 1U;

  for (command = UART_COMMAND_START;
       command < UART_COMMAND_COUNT;
       command = (UartCommand)((int)command + 1))
  {
    uart.match_index[command] = 0U;
  }
}

void UART_SendMessage(UartMessage message)
{
  /* Legacy keeps its text event. RR2 uses discrete EVENT/ACK retries. */
  const UartText *entry;
  const char *event_code;

  if (message >= UART_MESSAGE_COUNT)
  {
    Error_Handler();
  }

  entry = &uart_message_table[message];
  if (rr2.active == 0U) {
    UART_SendText(entry->text);
    return;
  }
  switch (message) {
    case UART_MESSAGE_STARTED: event_code = "MISSION_STARTED"; break;
    case UART_MESSAGE_ACTUATOR_TEST_READY: event_code = "ACTUATOR_TEST_READY"; break;
    case UART_MESSAGE_CAMERA_CAPTURE: event_code = "CAMERA_CAPTURE"; break;
    case UART_MESSAGE_RETURN_START: event_code = "RETURN_START"; break;
    case UART_MESSAGE_REALTIME_START: event_code = "REALTIME_START"; break;
    case UART_MESSAGE_RESCAN_RETURN_START: event_code = "RESCAN_RETURN_START"; break;
    case UART_MESSAGE_RESCAN_START: event_code = "RESCAN_START"; break;
    case UART_MESSAGE_RESCAN_DONE: event_code = "RESCAN_DONE"; break;
    case UART_MESSAGE_DONE: event_code = "MISSION_DONE"; break;
    case UART_MESSAGE_COUNT: default: event_code = "UNKNOWN"; break;
  }
  (void)UART_QueueEvent(event_code, NULL);
}

void UART_QueueFailSafeAllOff(void)
{
  /* 통신 오류 시 legacy 대기 명령을 비우고 네 부속장치 OFF를 예약한다.
   * DEBUG_TEST에서는 주행 PWM도 즉시 끄고 세션 중단을 요청하지만, 일반 임무 주행을 여기서 일괄 정지하지는 않는다. */
  uint32_t primask = __get_PRIMASK();

  __disable_irq();
  Mission_OnUartFault();
  uart.pending_commands = 0U;
  Actuators_QueueCommand(ACTUATOR_FRONT_CLEANER, ACTUATOR_COMMAND_OFF);
  Actuators_QueueCommand(ACTUATOR_FRONT_PUMP, ACTUATOR_COMMAND_OFF);
  Actuators_QueueCommand(ACTUATOR_SIDE_CLEANER, ACTUATOR_COMMAND_OFF);
  Actuators_QueueCommand(ACTUATOR_SIDE_PUMP, ACTUATOR_COMMAND_OFF);
  if (primask == 0U)
  {
    __enable_irq();
  }
}

void UART_SendText(const char *text)
{
  size_t length;
  if (text == NULL) return;
  length = strlen(text);
  if (length == 0U || length >= UART_TX_LINE_CAP) return;
  if (uart_tx_count >= UART_TX_CAP) {
    if (rr2.active != 0U) RR2_Fault();
    else UART_QueueFailSafeAllOff();
    return;
  }
  uint8_t slot = (uint8_t)((uart_tx_head + uart_tx_count) % UART_TX_CAP);
  memcpy(uart_tx[slot].bytes, text, length + 1U);
  uart_tx[slot].length = (uint16_t)length;
  uart_tx_count++;
}

void UART_OnRx(UART_HandleTypeDef *huart)
{
  /* 정상 수신은 파서/명령함을 갱신하고 다음 1바이트 수신을 재등록한다. 일반 명령 실행·응답 송신은 main이 맡는다.
   * 재등록/수신 오류는 부속장치 OFF를 예약하며, DEBUG_TEST 주행은 IRQ 문맥에서 즉시 정지한다.
   * 수신 재등록 재시도는 rearm_required를 확인하는 main에서 처리한다. */
  if (huart->Instance == USART2)
  {
    if (huart->ErrorCode == HAL_UART_ERROR_NONE)
    {
      UART_ParseByte(uart.rx_byte);

      if (HAL_UART_Receive_IT(&huart2, &uart.rx_byte, 1) != HAL_OK)
      {
        uart.rearm_required = 1;
        if (rr2.active != 0U) RR2_RequestFault();
        else UART_QueueFailSafeAllOff();
      }
      else
      {
        uart.rearm_required = 0;
      }
    }
    else
    {
      UART_ResetMatchers();
      uart.rearm_required = 1;
      if (rr2.active != 0U) RR2_RequestFault();
      else UART_QueueFailSafeAllOff();
    }
  }
}

void UART_OnError(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    UART_ResetMatchers();
    if (rr2.active != 0U) RR2_RequestFault();
    else UART_QueueFailSafeAllOff();
    if (huart->RxState == HAL_UART_STATE_READY)
    {
      uart.rearm_required = 1;
    }
  }
}

void UART_Init(void)
{
  uart = (UartContext){0};
  rr2 = (Rr2Context){0};
  rr2_fault_pending = 0U;
  memset(uart_tx, 0, sizeof(uart_tx));
  UART_ClearTxQueue();
  rr2_line_start = 1U;
  HAL_NVIC_ClearPendingIRQ(USART2_IRQn);
  if (HAL_UART_Receive_IT(&huart2, &uart.rx_byte, 1) != HAL_OK)
  {
    Error_Handler();
  }
}

static int UART_LegacyDriveState(MissionState state)
{
  return state == INITIAL_SCAN || state == RETURNING || state == REALTIME_CLEANING ||
         state == RESCAN_RETURN || state == RESCAN || state == FINAL_RETURN;
}

static void UART_LegacyLinkFault(void)
{
  /* RR2_Fault와 같은 순서로 출력을 끄고 FAULT를 래치한다. 세션 정보는 없으므로 건드리지 않는다. */
  Mission_LatchFault();
  __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
  Motor_Stop();
  Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
  CleaningActuators_StopAll();
  UART_ClearPending();
}

void UART_Update(void)
{
  if (RR2_ServicePendingFault()) return;
  /* UART 콜백이 수신 재등록에 실패하면 READY 상태에서 여기서 재시도한다.
   * rearm_required가 계속 1이면 파싱보다 HAL 수신 상태/오류를 먼저 확인하며 실패 시 부속장치 OFF를 유지한다. */
  if ((uart.rearm_required != 0) &&
      (huart2.RxState == HAL_UART_STATE_READY))
  {
    uart.rearm_required = 0;
    if (HAL_UART_Receive_IT(&huart2, &uart.rx_byte, 1) != HAL_OK)
    {
      uart.rearm_required = 1;
      UART_ResetMatchers();
      UART_QueueFailSafeAllOff();
      CleaningActuators_StopAll();
      if (rr2.active != 0U) RR2_Fault();
    }
  }
  /* Check elapsed liveness before trusting a newly arrived PING/CMD. A late
   * packet must never revive a drive after the 600 ms supervision gap. */
  if (rr2.active != 0U && Mission_GetState() != MISSION_FAULT &&
      (uint32_t)(HAL_GetTick() - rr2.last_seen_tick) >= RR2_HEARTBEAT_TIMEOUT_MS) {
    if (RR2_CanEndWithoutFault()) RR2_EndSessionWithoutFault();
    else RR2_Fault();
  }
  /* legacy 통신 감시. RR2 세션이 없고 주행 상태일 때만 본다. 촬영 ACK 대기(정지 상태)는
   * 기존 30 s 촬영 시간 제한이 맡으므로 여기서 제외한다. 복구는 RR2와 같이 STM 리셋이다. */
  if (rr2.active == 0U && UART_LegacyDriveState(Mission_GetState()) &&
      (uint32_t)(HAL_GetTick() - uart.legacy_seen_tick) >= LEGACY_LINK_TIMEOUT_MS) {
    UART_LegacyLinkFault();
  }
  while (rr2.pending_count != 0U) {
    if (RR2_ServicePendingFault()) return;
    char line[RR2_LINE_CAP];
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    memcpy(line, rr2.line[rr2.read_slot], sizeof(line));
    rr2.read_slot ^= 1U;
    rr2.pending_count--;
    if (primask == 0U) __enable_irq();
    RR2_ProcessLine(line);
  }
  if (RR2_ServicePendingFault()) return;
  RR2_ServiceEventsAndStatus();
  if (RR2_ServicePendingFault()) return;
  UART_DrainTx();
}

void UART_ClearPending(void) { uart.pending_commands = 0U; }

void UART_ObstacleBarrier(void)
{
  /* Caller masks IRQs. Preserve START/capture ACK; discard stale energizing commands. */
  uart.pending_commands &= (1UL<<UART_COMMAND_START) | (1UL<<UART_COMMAND_CAPTURE_OK) |
                           (1UL<<UART_COMMAND_MISSION_CANCEL_PREPARE);
  for (unsigned i=UART_COMMAND_ACTUATOR_TEST_START;i<UART_COMMAND_COUNT;++i) uart.match_index[i]=0U;
  Debug_ObstacleBarrier();
}
