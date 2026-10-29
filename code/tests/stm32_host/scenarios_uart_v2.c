#include "checks.h"
#include "../fixtures/rr2_vectors.h"
#include "debug_console.h"
#include "ultrasonic.h"
#include <stdlib.h>

static void ultrasonic_pulse(unsigned echo_us);

/* Independent fixture encoder; flips below exercise the firmware CRC gate. */
static uint16_t fixture_crc(const char *data)
{
  uint16_t crc = 0xffffU;
  while (*data) {
    crc ^= (uint16_t)(uint8_t)*data++ << 8;
    for (unsigned bit = 0; bit < 8; ++bit)
      crc = (uint16_t)((crc & 0x8000U) ? (crc << 1) ^ 0x1021U : crc << 1);
  }
  return crc;
}

static void rr2(const char *body)
{
  char frame[150];
  snprintf(frame, sizeof(frame), "%s %04X\r\n", body, fixture_crc(body));
  host_line(frame);
}

static void rr2_feed(const char *body)
{
  char frame[150];
  snprintf(frame, sizeof(frame), "%s %04X\r\n", body, fixture_crc(body));
  host_feed(frame);
}

static void rr2_capability(const char *capability)
{
  char body[128];
  snprintf(body, sizeof(body), "RR2 HELLO %s %s", SID, capability);
  rr2(body);
  /* Model current sensor data after the handshake reset; samples collected
   * before the MISSION session no longer make its start appear safe. */
  if (strcmp(capability, "MISSION") == 0)
    for (unsigned i=0; i<20U; ++i) ultrasonic_pulse(2000U);
}

static void rr2_command(unsigned long sequence, const char *opcode,
                        const char *arguments)
{
  char body[128];
  if (arguments != NULL && arguments[0] != '\0')
    snprintf(body, sizeof(body), "RR2 CMD %s %lu %s %s", SID, sequence,
             opcode, arguments);
  else
    snprintf(body, sizeof(body), "RR2 CMD %s %lu %s", SID, sequence, opcode);
  rr2(body);
}

static unsigned count_text(const char *haystack, const char *needle)
{
  unsigned count = 0U;
  size_t length = strlen(needle);
  const char *cursor = haystack;
  while ((cursor = strstr(cursor, needle)) != NULL) {
    ++count;
    cursor += length;
  }
  return count;
}

static void ultrasonic_pulse(unsigned echo_us)
{
  unsigned ticks = 0U;
  while ((GPIOB->ODR & GPIO_PIN_8) == 0U && ticks++ < 1201U)
    Ultrasonic_Tick50us();
  assert((GPIOB->ODR & GPIO_PIN_8) != 0U);
  Ultrasonic_Tick50us();
  Ultrasonic_OnEcho(1);
  DWT->CYCCNT += echo_us * 72U;
  Ultrasonic_OnEcho(0);
}

static const char *assert_valid_status(void)
{
  const char *line = strstr(host_output, "RR2 STATUS ");
  const char *end;
  const char *separator;
  char body[128], crc_text[5];
  char status_session[33], status_key[16];
  unsigned long sequence;
  assert(line != NULL);
  end = strstr(line, "\r\n");
  assert(end != NULL);
  separator = end;
  while (separator > line && separator[-1] != ' ') --separator;
  assert(separator > line && separator < end);
  size_t body_length = (size_t)(separator - line - 1);
  size_t crc_length = (size_t)(end - separator);
  assert(body_length < sizeof(body) && crc_length == 4U);
  memcpy(body, line, body_length); body[body_length] = '\0';
  memcpy(crc_text, separator, 4U); crc_text[4] = '\0';
  assert(fixture_crc(body) == (uint16_t)strtoul(crc_text, NULL, 16));
  assert(sscanf(body, "RR2 STATUS %32s %lu %15s", status_session,
                &sequence, status_key) == 3);
  assert(strcmp(status_session, SID) == 0 && sequence > 0UL &&
         status_key[0] != '\0');
  return line;
}

void scenario_uart_v2(const char *name)
{
  const char *session = SID;
  char body[128];
  host_reset();
  if (!strcmp(name, "uart_v2_shared_vectors")) {
    host_line(RR2_VECTOR_HELLO_WIRE);
    assert(strcmp(host_output, RR2_VECTOR_READY_WIRE) == 0);
    host_line(RR2_VECTOR_CMD_WIRE);
    assert(strstr(host_output + strlen(RR2_VECTOR_READY_WIRE),
                  RR2_VECTOR_ACK_WIRE));
    assert(Mission_GetState() == INITIAL_SCAN);
    return;
  }
  if (!strcmp(name, "uart_v2_independent_cmd_ping_lanes")) {
    rr2_capability("MISSION");
    rr2_command(1U, "START", NULL);
    Robot_Update();
    MissionState started_state = Mission_GetState();
    char expected_start_ack[96];
    snprintf(expected_start_ack, sizeof(expected_start_ack),
             "RR2 ACK %s 1 START ACCEPTED", SID);
    assert(count_text(host_output, expected_start_ack) == 1U);

    /* PING uses its own monotonically increasing sequence lane; sequence 1
     * may coexist with the unacknowledged/retried CMD sequence 1. */
    snprintf(body, sizeof(body), "RR2 PING %s 1", SID);
    rr2(body);
    Robot_Update();
    char expected_ping_ack[96];
    snprintf(expected_ping_ack, sizeof(expected_ping_ack),
             "RR2 ACK %s 1 PING ACCEPTED", SID);
    assert(count_text(host_output, expected_ping_ack) == 1U);

    /* A lost CMD ACK may be retried byte-for-byte after PING. It must replay
     * the cached CMD ACK and must not execute START a second time. */
    rr2_command(1U, "START", NULL);
    Robot_Update();
    assert(count_text(host_output, expected_start_ack) == 2U);
    assert(Mission_GetState() == started_state);
    assert(count_text(host_output, "RR2 NACK") == 0U);
    return;
  }
  if (!strcmp(name, "uart_v2_hello_retry_not_heartbeat")) {
    host_line(RR2_VECTOR_HELLO_WIRE);
    assert(strcmp(host_output, RR2_VECTOR_READY_WIRE) == 0);
    host_time(600U - 10U);
    size_t before = strlen(host_output);
    host_line(RR2_VECTOR_HELLO_WIRE);
    assert(strcmp(host_output + before, RR2_VECTOR_READY_WIRE) == 0);
    assert(Mission_GetState() == MISSION_IDLE);
    host_time(600U - 1U);
    Robot_Update();
    assert(Mission_GetState() == MISSION_IDLE);
    host_time(600U);
    Robot_Update();
    /* Clean IDLE: the lost heartbeat closes the session without latching FAULT. */
    assert(Mission_GetState() == MISSION_IDLE && !UART_V2Active());
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_idle_close_reconnect")) {
    /* After a normal stop the ROS bridge exits; a new bridge must reconnect
     * with a new session without an STM reset. */
    rr2_capability("MISSION");
    host_time(600U);
    Robot_Update();
    assert(Mission_GetState() == MISSION_IDLE && !UART_V2Active());
    rr2("RR2 HELLO fedcba9876543210fedcba9876543210 MISSION");
    assert(strstr(host_output, "RR2 READY fedcba9876543210fedcba9876543210 2 MISSION"));
    assert(UART_V2Active() && Mission_GetState() == MISSION_IDLE);
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_active_timeout_still_faults")) {
    /* Prepared or moving sessions keep the existing FAULT latch. */
    rr2_capability("MISSION");
    rr2_command(1U, "PREPARE", NULL); Robot_Update();
    assert(Mission_IsPrepared());
    host_time(600U);
    Robot_Update();
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    host_reset();
    rr2_capability("MISSION");
    rr2_command(1U, "START", NULL); Robot_Update();
    assert(Mission_GetState() == INITIAL_SCAN);
    host_time(600U);
    Robot_Update();
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    /* A latched FAULT is never cleared by a reconnect attempt. */
    rr2("RR2 HELLO fedcba9876543210fedcba9876543210 MISSION");
    assert(!strstr(host_output, "RR2 READY fedcba9876543210fedcba9876543210"));
    assert(Mission_GetState() == MISSION_FAULT);
    return;
  }
  if (!strcmp(name, "uart_v2_bad_capability")) {
    snprintf(body, sizeof(body), "RR2 HELLO %s UNKNOWN_CAPABILITY", session);
    rr2(body);
    assert(!strstr(host_output, "RR2 READY"));
    assert(!UART_V2Active());
    snprintf(body, sizeof(body), "RR2  HELLO %s MISSION", session);
    rr2(body);
    assert(!strstr(host_output, "RR2 READY"));
    assert(!UART_V2Active());
    host_line(RR2_VECTOR_HELLO_WIRE);
    assert(strcmp(host_output, RR2_VECTOR_READY_WIRE) == 0);
    return;
  }
  if (!strcmp(name, "uart_v2_bad_crc")) {
    char corrupt[sizeof(RR2_VECTOR_HELLO_WIRE)];
    char lf_only[sizeof(RR2_VECTOR_HELLO_WIRE)];
    strcpy(corrupt, RR2_VECTOR_HELLO_WIRE);
    corrupt[strlen(corrupt) - 4U] = corrupt[strlen(corrupt) - 4U] == '0' ? '1' : '0';
    host_line(corrupt);
    assert(host_output[0] == '\0');
    assert(!UART_V2Active());
    strcpy(lf_only, RR2_VECTOR_HELLO_WIRE);
    lf_only[strlen(lf_only) - 2U] = '\n';
    lf_only[strlen(lf_only) - 1U] = '\0';
    host_line(lf_only);
    assert(host_output[0] == '\0');
    assert(!UART_V2Active());
    return;
  }
  if (!strcmp(name, "uart_v2_partial_and_oversize")) {
    host_feed("RR2 HELLO " SID " MISSION");
    host_drain();
    assert(host_output[0] == '\0');
    assert(!UART_V2Active());
    host_feed(" 3F49\r\n");
    host_drain();
    assert(strcmp(host_output, RR2_VECTOR_READY_WIRE) == 0);
    host_feed("RR2 ");
    for (unsigned i = 0U; i < 140U; ++i) host_feed("X");
    host_feed("\r\n");
    host_drain();
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_fault_nack")) {
    host_line(RR2_VECTOR_HELLO_WIRE);
    Mission_LatchFault();
    snprintf(body, sizeof(body), "RR2 CMD %s 1 START", session);
    rr2(body);
    assert(strstr(host_output, "RR2 NACK " SID " 1 START FAULT"));
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_estop")) {
    host_line(RR2_VECTOR_HELLO_WIRE);
    assert(strstr(host_output, "RR2 READY " SID " 2 MISSION"));
    Motor_Run(1, 800);
    assert(TIM2->CCR2 != 0U || TIM2->CCR3 != 0U);
    rr2_command(1UL, "ESTOP", NULL);
    assert(strstr(host_output, "RR2 ACK " SID " 1 ESTOP ACCEPTED"));
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    assert((TIM3->DIER & TIM_IT_UPDATE) == 0U);
    rr2_command(1UL, "ESTOP", NULL);
    assert(count_text(host_output, "RR2 ACK " SID " 1 ESTOP ACCEPTED") == 2U);
    rr2_command(2UL, "CAPTURE_OK", NULL);
    assert(strstr(host_output, "RR2 NACK " SID " 2 CAPTURE_OK FAULT"));
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_malformed_command")) {
    host_line(RR2_VECTOR_HELLO_WIRE);
    snprintf(body, sizeof(body), "RR2 CMD %s 1 START unexpected", session);
    rr2(body);
    assert(strstr(host_output, "RR2 NACK " SID " 1 START ARGS 0 "));
    assert(Mission_GetState() == MISSION_IDLE);
    size_t before = strlen(host_output);
    rr2(body);
    assert(strstr(host_output + before, "RR2 NACK " SID " 1 START ARGS 0 "));
    snprintf(body, sizeof(body), "RR2 CMD %s 1 START", session);
    rr2(body);
    assert(strstr(host_output + before, "RR2 NACK " SID " 1 START CONFLICT 0 "));
    assert(Mission_GetState() == MISSION_IDLE);
    return;
  }
  if (!strcmp(name, "uart_v2_crc")) {
    snprintf(body, sizeof(body), "RR2 HELLO %s MISSION", session);
    host_line("RR2 HELLO " SID " 0000\r\n");
    assert(!strstr(host_output, "RR2 READY"));
    rr2(body);
    assert(strstr(host_output, "RR2 READY " SID " 2 "));
    host_line("RR2 CMD " SID " 1 START 0000\r\n");
    assert(Mission_GetState() == MISSION_IDLE);
    return;
  }
  if (!strcmp(name, "uart_v2_busy_hello")) {
    host_line("START\r\n");
    assert(Mission_GetState() == INITIAL_SCAN);
    snprintf(body, sizeof(body), "RR2 HELLO %s MISSION", session);
    rr2(body);
    assert(!strstr(host_output, "RR2 READY"));
    return;
  }
  if (!strcmp(name, "uart_v2_prepare_sections")) {
    /* PREPARE 인자: 다음 임무의 단계별 구간 수(1..ENCODER_SECTION_LIMIT). 잘못된 인자는 상태를 바꾸지 않는다. */
    static const char *bad[][2] = {{"PREPARE", "0"}, {"PREPARE", "7"}, {"PREPARE", "x"}, {"PREPARE", "22"},
                                   {"PREPARE", "2 3"}, {"CANCEL_PREPARE", "2"}};
    rr2_capability("MISSION");
    for (unsigned i = 0U; i < sizeof(bad) / sizeof(bad[0]); ++i) {
      char reply[96];
      rr2_command(1U + i, bad[i][0], bad[i][1]); Robot_Update();
      snprintf(reply, sizeof(reply), "RR2 NACK " SID " %u %s ARGS", 1U + i, bad[i][0]);
      assert(strstr(host_output, reply) && !Mission_IsPrepared());
    }
    rr2_command(10U, "PREPARE", "2"); Robot_Update();
    assert(strstr(host_output, "RR2 ACK " SID " 10 PREPARE ACCEPTED") && Mission_IsPrepared());
    rr2_command(11U, "START", NULL); Robot_Update();
    assert(Mission_GetState() == INITIAL_SCAN && Mission_GetSectionLimit() == 2);
    return;
  }
  if (!strcmp(name, "uart_v2_prepare_default_after_test")) {
    /* 인자 없는 PREPARE와 CANCEL은 기본 구간 수(ENCODER_SECTION_LIMIT)다. 시험 PREPARE 뒤 CANCEL하면 다음 START는 기본 구간 수다. */
    rr2_capability("MISSION");
    rr2_command(1U, "PREPARE", "2"); Robot_Update();
    rr2_command(2U, "CANCEL_PREPARE", NULL); Robot_Update();
    rr2_command(3U, "START", NULL); Robot_Update();
    assert(Mission_GetState() == INITIAL_SCAN && Mission_GetSectionLimit() == ENCODER_SECTION_LIMIT);
    return;
  }
  if (!strcmp(name, "uart_v2_mission_prepare")) {
    rr2_capability("MISSION");
    rr2_command(1U, "PREPARE", NULL); Robot_Update();
    assert(strstr(host_output, "RR2 ACK " SID " 1 PREPARE ACCEPTED"));
    assert(Mission_GetState() == MISSION_IDLE);
    rr2_command(2U, "CANCEL_PREPARE", NULL); Robot_Update();
    assert(strstr(host_output, "RR2 ACK " SID " 2 CANCEL_PREPARE ACCEPTED"));
    assert(Mission_GetState() == MISSION_IDLE);
    rr2_command(3U, "PREPARE", NULL); Robot_Update();
    assert(strstr(host_output, "RR2 ACK " SID " 3 PREPARE ACCEPTED"));
    rr2_command(4U, "START", NULL); Robot_Update();
    assert(strstr(host_output, "RR2 ACK " SID " 4 START ACCEPTED"));
    assert(Mission_GetState() == INITIAL_SCAN);
    assert(strstr(host_output, "MISSION_STARTED"));
    return;
  }
  /* These scenarios own their capability handshake and must not pass through
   * the generic mission START path below. */
  if (!strcmp(name, "uart_v2_legacy_isolation")) {
    static const char *blocked[] = {
      "FRONT_CLEANER_PWM_12_5\r\n", "FRONT_CLEANER_PWM_25\r\n",
      "FRONT_PUMP_ON\r\n", "SIDE_CLEANER_PWM_12_5\r\n",
      "SIDE_CLEANER_PWM_25\r\n", "SIDE_PUMP_ON\r\n",
      "FRONT_CLEANER_OFF\r\n", "FRONT_PUMP_OFF\r\n",
      "SIDE_CLEANER_OFF\r\n", "SIDE_PUMP_OFF\r\n"
    };
    rr2_capability("MISSION");
    assert(UART_V2Active());
    for (unsigned i = 0U; i < sizeof(blocked) / sizeof(blocked[0]); ++i) {
      host_line(blocked[i]);
      assert(UART_TakePendingCommand() == UART_COMMAND_NONE);
      assert(Actuators_GetOutputMask() == 0U);
    }
    host_line("ACTUATOR_TEST_START\r\n");
    assert(UART_TakePendingCommand() == UART_COMMAND_NONE);
    host_line("DBG1_HELLO " SID " 7\r\n");
    assert(Mission_GetState() == MISSION_IDLE);
    assert(strstr(host_output, "DBG1_READY") == NULL);
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_mission_accessories")) {
    rr2_capability("MISSION");
    Mission_SetState(REALTIME_CLEANING);
    rr2_command(1U, "FRONT_CLEANER_PWM_12_5", NULL); Robot_Update();
    assert(TIM1->CCR4 == FRONT_DC_MOTOR_PWM_DUTY_12_5 && (Actuators_GetOutputMask() & 1U));
    rr2_command(2U, "FRONT_CLEANER_PWM_25", NULL); Robot_Update();
    assert(TIM1->CCR4 == DC_MOTOR_PWM_DUTY_25);
    rr2_command(3U, "FRONT_PUMP_ON", NULL); Robot_Update();
    assert(TIM1->CCR1 == WATER_PUMP_PWM_DUTY && (Actuators_GetOutputMask() & 4U));
    rr2_command(4U, "SIDE_CLEANER_PWM_12_5", NULL); Robot_Update();
    assert(TIM1->CCR2 == SIDE_DC_MOTOR_PWM_DUTY_12_5 && (Actuators_GetOutputMask() & 2U));
    rr2_command(5U, "SIDE_CLEANER_PWM_25", NULL); Robot_Update();
    assert(TIM1->CCR2 == DC_MOTOR_PWM_DUTY_25);
    rr2_command(6U, "SIDE_PUMP_ON", NULL); Robot_Update();
    assert(TIM1->CCR3 == WATER_PUMP_PWM_DUTY && Actuators_GetOutputMask() == 15U);
    rr2_command(7U, "FRONT_CLEANER_OFF", NULL); Robot_Update();
    assert(!(Actuators_GetOutputMask() & 1U));
    rr2_command(8U, "FRONT_PUMP_OFF", NULL); Robot_Update();
    assert(!(Actuators_GetOutputMask() & 4U));
    rr2_command(9U, "SIDE_CLEANER_OFF", NULL); Robot_Update();
    assert(!(Actuators_GetOutputMask() & 2U));
    rr2_command(10U, "SIDE_PUMP_OFF", NULL); Robot_Update();
    assert(Actuators_GetOutputMask() == 0U);
    return;
  }
  if (!strcmp(name, "uart_v2_debug_actuator")) {
    rr2_capability("ACTUATOR_DEBUG_ALL");
    assert(Mission_GetState() == DEBUG_TEST);
    rr2_command(1U, "SET", "1 5000 15");
    assert(Drive_GetDirection() == MOTOR_DIRECTION_FORWARD);
    assert(Actuators_GetOutputMask() == 15U);
    assert(TIM2->CCR2 != 0U || TIM2->CCR1 != 0U);
    rr2_command(2U, "STOP", NULL);
    assert(Drive_GetDirection() == MOTOR_DIRECTION_STOPPED);
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_debug_sessions_end_idle")) {
    /* A debug host that leaves (heartbeat and command timeouts) ends its
     * session: outputs off, IDLE, and a new MISSION session without reset. */
    static const char *capabilities[] = {
      "SENSOR_DEBUG", "ACTUATOR_DEBUG_ENCODER", "ACTUATOR_DEBUG_DC",
      "ACTUATOR_DEBUG_PUMP", "ACTUATOR_DEBUG_ALL"
    };
    static const char *outputs[] = {NULL, "1 5000 0", "0 0 3", "0 0 12", "1 5000 15"};
    for (unsigned i = 0U; i < 5U; ++i) {
      host_reset();
      rr2_capability(capabilities[i]);
      assert(Mission_GetState() == DEBUG_TEST && UART_V2Active());
      if (outputs[i] != NULL) {
        rr2_command(1U, "SET", outputs[i]);
        assert(Drive_GetDirection() != MOTOR_DIRECTION_STOPPED ||
               Actuators_GetOutputMask() != 0U);
      }
      host_time(700U);
      Robot_Update();
      Robot_Update();
      assert(Mission_GetState() == MISSION_IDLE && !UART_V2Active());
      all_off();
      rr2("RR2 HELLO fedcba9876543210fedcba9876543210 MISSION");
      assert(strstr(host_output, "RR2 READY fedcba9876543210fedcba9876543210 2 MISSION"));
      assert(UART_V2Active() && Mission_GetState() == MISSION_IDLE);
    }
    /* A real link error inside a debug session still latches FAULT. */
    host_reset();
    rr2_capability("ACTUATOR_DEBUG_ALL");
    rr2_command(1U, "SET", "1 5000 15");
    Robot_OnUartError(&huart2);
    all_off();
    Robot_Update();
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_debug_capability_gates")) {
    host_reset(); rr2_capability("ACTUATOR_DEBUG_ENCODER");
    rr2_command(1U, "SET", "1 5000 0");
    assert(Drive_GetDirection() == MOTOR_DIRECTION_FORWARD);
    rr2_command(2U, "SET", "0 0 4");
    assert(strstr(host_output, "RR2 NACK " SID " 2 SET CAPABILITY"));
    assert(Actuators_GetOutputMask() == 0U);
    rr2_command(3U, "STOP", NULL);
    assert(Drive_GetDirection() == MOTOR_DIRECTION_STOPPED);
    all_off();

    host_reset(); rr2_capability("ACTUATOR_DEBUG_DC");
    rr2_command(1U, "SET", "0 0 3");
    assert(Actuators_GetOutputMask() == 3U);
    rr2_command(2U, "SET", "0 0 4");
    assert(strstr(host_output, "RR2 NACK " SID " 2 SET CAPABILITY"));
    assert(Actuators_GetOutputMask() == 3U);
    rr2_command(3U, "STOP", NULL);
    assert(Actuators_GetOutputMask() == 0U);
    all_off();

    host_reset(); rr2_capability("ACTUATOR_DEBUG_PUMP");
    rr2_command(1U, "SET", "0 0 12");
    assert(Actuators_GetOutputMask() == 12U);
    rr2_command(2U, "SET", "0 0 1");
    assert(strstr(host_output, "RR2 NACK " SID " 2 SET CAPABILITY"));
    assert(Actuators_GetOutputMask() == 12U);
    rr2_command(3U, "STOP", NULL);
    assert(Actuators_GetOutputMask() == 0U);
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_sensor_status")) {
    uint8_t red = 0U, green = 0U, blue = 0U;
    rr2_capability("SENSOR_DEBUG");
    assert(Mission_GetState() == DEBUG_TEST);
    rr2_command(1U, "LED", "manual 17 34 51");
    assert(Debug_GetLedOverride(&red, &green, &blue));
    assert(red == 17U && green == 34U && blue == 51U);
    host_time(100U); Robot_Update();
    assert_valid_status();
    assert(strstr(host_output, " ULTRASONIC ") != NULL);
    assert(count_text(host_output, "RR2 ACK ") == 1U);
    rr2_command(2U, "SET", "1 5000 0");
    assert(strstr(host_output, "RR2 NACK " SID " 2 SET CAPABILITY"));
    assert(Drive_GetDirection() == MOTOR_DIRECTION_STOPPED);
    return;
  }
  if (!strcmp(name, "uart_v2_event_retry_ack")) {
    rr2_capability("MISSION");
    Mission_SetState(INITIAL_SCAN);
    UART_Update();
    const char *first = strstr(host_output, "RR2 EVENT ");
    assert(first != NULL && strstr(first, "MISSION_STARTED") != NULL);
    char saved[128];
    const char *end = strstr(first, "\r\n");
    assert(end != NULL && (size_t)(end - first) < sizeof(saved));
    memcpy(saved, first, (size_t)(end - first)); saved[end - first] = '\0';
    host_time(100U); UART_Update();
    const char *second = strstr(first + strlen(saved), "RR2 EVENT ");
    assert(second != NULL);
    assert(strncmp(saved, second, strlen(saved)) == 0);
    snprintf(body, sizeof(body), "RR2 EVENT_ACK %s 1 ACCEPTED", session);
    rr2(body);
    unsigned before = count_text(host_output, "RR2 EVENT ");
    host_time(200U); UART_Update();
    assert(count_text(host_output, "RR2 EVENT ") == before);
    return;
  }
  if (!strcmp(name, "uart_v2_obstacle_event")) {
    rr2_capability("MISSION");
    rr2_command(1U, "START", NULL);
    assert(Mission_GetState() == INITIAL_SCAN);
    snprintf(body, sizeof(body), "RR2 EVENT_ACK %s 1 ACCEPTED", session);
    rr2(body);
    for (unsigned i = 0U; i < 16U; ++i) {
      ultrasonic_pulse(584U);
      Robot_Update();
    }
    assert(Ultrasonic_IsHeld());
    const char *first = strstr(host_output,
        "RR2 EVENT " SID " 2 OBSTACLE_HOLD");
    assert(first != NULL);
    char saved[128];
    const char *end = strstr(first, "\r\n");
    assert(end != NULL && (size_t)(end - first) < sizeof(saved));
    memcpy(saved, first, (size_t)(end - first)); saved[end - first] = '\0';
    host_time(100U); Robot_Update();
    const char *second = strstr(first + strlen(saved), "RR2 EVENT ");
    assert(second != NULL && strncmp(saved, second, strlen(saved)) == 0);
    snprintf(body, sizeof(body), "RR2 EVENT_ACK %s 2 ACCEPTED", session);
    rr2(body);
    unsigned before = count_text(host_output, "OBSTACLE_HOLD");
    host_time(200U); Robot_Update();
    assert(count_text(host_output, "OBSTACLE_HOLD") == before);

    for (unsigned i = 0U; i < 16U; ++i) {
      ultrasonic_pulse(2000U);
      Robot_Update();
    }
    assert(!Ultrasonic_IsHeld());
    assert(strstr(host_output, "RR2 EVENT " SID " 3 OBSTACLE_CLEAR") != NULL);
    Robot_EmergencyStop();
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_event_fail_safe")) {
    rr2_capability("MISSION");
    Mission_SetState(INITIAL_SCAN);
    UART_Update();
    host_time(100U); UART_Update();
    host_time(200U); UART_Update();
    host_time(300U); UART_Update();
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();

    host_reset(); rr2_capability("MISSION");
    Mission_SetState(INITIAL_SCAN);
    Mission_SetState(WAIT_CAPTURE_ACK);
    Mission_SetState(RETURNING);
    Mission_SetState(REALTIME_CLEANING);
    Mission_SetState(RESCAN_RETURN);
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    return;
  }
  snprintf(body, sizeof(body), "RR2 HELLO %s MISSION", session);
  rr2(body);
  assert(strstr(host_output, "RR2 READY " SID " 2 "));
  for (unsigned i=0; i<20U; ++i) ultrasonic_pulse(2000U);
  if (!strcmp(name, "uart_v2_legacy_start")) {
    host_line("START\r\n");
    assert(Mission_GetState() == MISSION_IDLE);
    return;
  }
  snprintf(body, sizeof(body), "RR2 CMD %s 1 START", session);
  rr2(body);
  assert(Mission_GetState() == INITIAL_SCAN);
  assert(strstr(host_output, "RR2 ACK " SID " 1 START ACCEPTED 1 "));
  if (!strcmp(name, "uart_v2_duplicate")) {
    size_t before = strlen(host_output);
    rr2(body);
    assert(Mission_GetState() == INITIAL_SCAN);
    assert(strstr(host_output + before, "RR2 ACK " SID " 1 START ACCEPTED 1 "));
    snprintf(body, sizeof(body), "RR2 CMD %s 1 CAPTURE_OK", session);
    rr2(body);
    assert(strstr(host_output, "RR2 NACK " SID " 1 CAPTURE_OK CONFLICT"));
    return;
  }
  if (!strcmp(name, "uart_v2_capture")) {
    Robot_OnEncoderUpdate(&htim3); Robot_Update();
    assert(Mission_GetState() == WAIT_CAPTURE_ACK); all_off();
    /* A raw legacy ACK cannot release a reliable session. */
    host_line("CAPTURE_OK\r\n");
    assert(Mission_GetState() == WAIT_CAPTURE_ACK);
    snprintf(body, sizeof(body), "RR2 CMD %s 2 CAPTURE_OK", session);
    rr2(body);
    assert(Mission_GetState() == INITIAL_SCAN);
    assert(strstr(host_output, "RR2 ACK " SID " 2 CAPTURE_OK ACCEPTED 1 "));
    rr2(body);
    assert(Mission_GetState() == INITIAL_SCAN);
    snprintf(body, sizeof(body), "RR2 CMD %s 1 CAPTURE_OK", session);
    rr2(body);
    assert(strstr(host_output, "RR2 NACK " SID " 1 CAPTURE_OK STALE"));
    return;
  }
  if (!strcmp(name, "uart_v2_burst")) {
    /* A pending heartbeat followed by a mission command is normal on Jetson. */
    host_reset();
    snprintf(body, sizeof(body), "RR2 HELLO %s MISSION", session);
    rr2(body);
    snprintf(body, sizeof(body), "RR2 PING %s 1", session);
    rr2_feed(body);
    snprintf(body, sizeof(body), "RR2 CMD %s 2 START", session);
    rr2_feed(body);
    host_drain();
    assert(Mission_GetState() == INITIAL_SCAN);
    assert(strstr(host_output, "RR2 ACK " SID " 1 PING ACCEPTED"));
    assert(strstr(host_output, "RR2 ACK " SID " 2 START ACCEPTED"));
    return;
  }
  if (!strcmp(name, "uart_v2_late_command")) {
    host_reset();
    snprintf(body, sizeof(body), "RR2 HELLO %s MISSION", session);
    rr2(body);
    host_time(700U);
    snprintf(body, sizeof(body), "RR2 CMD %s 1 START", session);
    rr2_feed(body);
    Robot_Update();
    /* The idle session already closed; the late START is ignored, never driven. */
    assert(Mission_GetState() == MISSION_IDLE && !UART_V2Active());
    assert(!strstr(host_output, "START ACCEPTED")); all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_uart_fault")) {
    rr2_capability("MISSION");
    rr2_command(1UL, "START", NULL);
    assert(Mission_GetState() == INITIAL_SCAN);
    Robot_OnUartError(&huart2);
    /* The IRQ stops outputs immediately; protocol/session teardown is applied
     * by the main-loop dispatcher rather than from the USART callback. */
    all_off();
    UART_Update();
    assert(Mission_GetState() == MISSION_FAULT); all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_fault_mid_robot_update")) {
    uint32_t writes_before;

    /* Error at the actuator mailbox's IRQ restore, after Ultrasonic_Update's
     * earlier restore. Mission_Update must not restart stale RETURNING motion. */
    rr2_capability("MISSION");
    rr2_command(1UL, "START", NULL);
    Mission_SetState(RETURNING);
    Robot_OnEncoderUpdate(&htim3);
    assert(Mission_HasEncoderEvent());
    Robot_Update();
    assert(Mission_GetState() == RETURNING);
    assert(!Mission_HasEncoderEvent());
    assert(TIM2->CCR1 != 0U || TIM2->CCR4 != 0U);
    Robot_OnEncoderUpdate(&htim3);
    assert(Mission_HasEncoderEvent());
    drive_off();
    writes_before = host_motor_accessory_nonzero_writes;
    Actuators_QueueCommand(ACTUATOR_FRONT_PUMP, ACTUATOR_COMMAND_OFF);
    UART_SendText("TX_MUST_BE_DROPPED\r\n");
    host_inject_uart_error_after_irq_restores(1U);
    Robot_Update();
    assert(host_mission_state_at_uart_error == MISSION_FAULT);
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    assert(host_motor_accessory_nonzero_writes == writes_before);
    assert(strstr(host_output, "TX_MUST_BE_DROPPED") == NULL);

    /* Inject at the same mailbox boundary with an allowed pending ON. */
    host_reset();
    rr2_capability("MISSION");
    rr2_command(1UL, "START", NULL);
    Mission_SetState(REALTIME_CLEANING);
    Actuators_QueueCommand(ACTUATOR_FRONT_PUMP, ACTUATOR_COMMAND_ON);
    writes_before = host_motor_accessory_nonzero_writes;
    UART_SendText("TX_ACCESSORY_MUST_BE_DROPPED\r\n");
    host_inject_uart_error_after_irq_restores(1U);
    Robot_Update();
    assert(host_mission_state_at_uart_error == MISSION_FAULT);
    assert(Mission_GetState() == MISSION_FAULT);
    all_off();
    assert(host_motor_accessory_nonzero_writes == writes_before);
    assert(strstr(host_output, "TX_ACCESSORY_MUST_BE_DROPPED") == NULL);
    return;
  }
  if (!strcmp(name, "uart_v2_tx_fault_race")) {
    rr2_capability("MISSION");
    UART_SendText("TX_PENDING_FIRST\r\n");
    UART_SendText("TX_PENDING_SECOND\r\n");
    /* Deliver a pending USART error at the exact IRQ-restore boundary after
     * the main loop dequeues an item, before its blocking TX call can begin. */
    host_inject_uart_error_after_next_irq_restore();
    UART_Update();
    assert(Mission_GetState() == MISSION_FAULT);
    assert(strstr(host_output, "TX_PENDING_FIRST") == NULL);
    assert(strstr(host_output, "TX_PENDING_SECOND") == NULL);
    /* Main-context teardown leaves the queue reusable; no uint8 underflow may
     * expose stale entries or prevent a later safe diagnostic response. */
    UART_SendText("TX_AFTER_FAULT\r\n");
    UART_Update();
    assert(count_text(host_output, "TX_AFTER_FAULT") == 1U);
    assert(strstr(host_output, "TX_PENDING_SECOND") == NULL);
    all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_heartbeat")) {
    host_time(500);
    snprintf(body, sizeof(body), "RR2 PING %s 2", session);
    rr2(body);
    assert(strstr(host_output, "RR2 ACK " SID " 2 PING ACCEPTED"));
    host_time(1000); Robot_Update();
    assert(Mission_GetState() == INITIAL_SCAN);
    host_time(1100); Robot_Update();
    assert(Mission_GetState() == MISSION_FAULT); all_off();
    return;
  }
  if (!strcmp(name, "uart_v2_capture_timeout")) {
    Robot_OnEncoderUpdate(&htim3); Robot_Update();
    assert(Mission_GetState() == WAIT_CAPTURE_ACK); all_off();
    for (uint32_t tick = 500; tick < 30000; tick += 500) {
      host_time(tick);
      snprintf(body, sizeof(body), "RR2 PING %s %lu", session, (unsigned long)(tick / 500 + 1));
      rr2(body);
    }
    host_time(30000); Robot_Update();
    assert(Mission_GetState() == MISSION_FAULT); all_off();
    return;
  }
  assert(!"unknown uart v2 scenario");
}
