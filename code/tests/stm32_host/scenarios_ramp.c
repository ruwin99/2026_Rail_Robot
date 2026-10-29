#include "checks.h"
#include "drive_sync_config.h"

/* 출발 가속(soft start) 시나리오. 운영 설정 DRIVE_SOFT_START_MS(500 ms)로 빌드해 실행한다.
 * 전진 PWM1=TIM2 CCR2(오른쪽), PWM2=CCR3(왼쪽). 후진은 CCR1/CCR4. */
#define BASE MISSION_FORWARD_PWM_DUTY
#define FLOOR(b) (((uint32_t)(b) * DRIVE_SOFT_START_FLOOR_PERCENT + 50U) / 100U)

static unsigned now_ms;
/* control 시나리오와 같은 방식: main 생존 표시 후 엔코더 값을 두고 SysTick 1회를 실행한다. */
static void tick_to(unsigned t, unsigned c1, unsigned c2) {
  now_ms = t; host_time(t); Drive_MarkMainAlive(); TIM3->CNT = c1; TIM4->CNT = c2; DriveSync_Tick1ms();
}
/* 엔코더는 펌웨어가 구간 시작 때 TIM3만 0으로 되돌리므로 두 값을 따로 이어간다.
 * 전진: TIM3 증가, TIM4 감소. 후진: TIM3 감소, TIM4 증가. 10 ms마다 5 count. */
static unsigned c3, c4;
static void advance(unsigned t, int direction) {
  c3 = (c3 + (direction > 0 ? 5U : 596U)) % 601U;
  c4 = (c4 + (direction > 0 ? 596U : 5U)) % 601U;
  tick_to(t, c3, c4);
}
static void forward_steps(unsigned from, unsigned to, unsigned *unused) {
  (void)unused;
  for (unsigned t = from; t <= to; t += 10U) advance(t, MOTOR_DIRECTION_FORWARD);
}
static void start_forward(unsigned t) {
  host_reset(); Mission_SetState(ACTUATOR_TEST); now_ms = t; host_time(t); Drive_MarkMainAlive();
  c3 = c4 = 0U; TIM3->CNT = 0U; TIM4->CNT = 0U;
  Motor_Run(MOTOR_DIRECTION_FORWARD, BASE);
}

void scenario_ramp(const char *name) {
  unsigned c = 0U;
  if (!strcmp(name, "ramp_start_curve")) {
    start_forward(1000U);
    /* 출발 직후에는 시작값(기준의 FLOOR_PERCENT)이고 반대 방향 채널은 계속 0이다. */
    assert(FLOOR(BASE) > 0U && TIM2->CCR2 == FLOOR(BASE) && TIM2->CCR3 == FLOOR(BASE));
    assert(TIM2->CCR1 == 0U && TIM2->CCR4 == 0U);
    uint32_t previous = FLOOR(BASE);
    for (unsigned t = 1010U; t < 1500U; t += 10U) {
      advance(t, MOTOR_DIRECTION_FORWARD);
      /* 가속 중에는 PID 보정 없이 두 바퀴가 같은 곡선 값이고 감소하지 않으며 기준을 넘지 않는다. */
      assert(TIM2->CCR2 == TIM2->CCR3 && TIM2->CCR3 >= previous && TIM2->CCR3 < BASE);
      assert(TIM2->CCR1 == 0U && TIM2->CCR4 == 0U);
      /* smoothstep(0.5)=0.5: 시작값과 기준의 정확히 중간 */
      if (t == 1250U) assert(TIM2->CCR3 == FLOOR(BASE) + (BASE - FLOOR(BASE) + 1U) / 2U);
      /* 처음 50 ms는 시작값 근처에서 천천히 오른다(전체 상승폭의 3% 미만). */
      if (t == 1050U) assert(TIM2->CCR3 > FLOOR(BASE) && TIM2->CCR3 - FLOOR(BASE) < (BASE - FLOOR(BASE)) * 3U / 100U);
      previous = TIM2->CCR3;
    }
    /* 500 ms부터 기준 PWM, 이후 PID 보정은 slew 제한(표본당 20) 안에서 시작한다. */
    advance(1500U, MOTOR_DIRECTION_FORWARD);
    assert(TIM2->CCR2 + TIM2->CCR3 == 2U * BASE);
    assert(TIM2->CCR3 >= BASE - 20U && TIM2->CCR3 <= BASE + 20U);
    forward_steps(1510U, 1700U, &c);
    assert(TIM2->CCR2 + TIM2->CCR3 == 2U * BASE && Mission_GetState() == ACTUATOR_TEST);
  } else if (!strcmp(name, "ramp_stall_deferred_then_detected")) {
    /* 바퀴가 전혀 돌지 않아도 가속 중(500 ms)에는 정지 폴트가 나지 않는다. */
    start_forward(1000U);
    for (unsigned t = 1010U; t <= 1490U; t += 10U) tick_to(t, 0U, 0U);
    assert(Mission_GetState() == ACTUATOR_TEST && TIM2->CCR3 > 0U);
    /* 가속이 끝나면 정지 판정이 다시 동작해 500 ms 안에 폴트로 멈춘다. */
    unsigned t = 1500U;
    while (Mission_GetState() != MISSION_FAULT && t <= 2100U) { tick_to(t, 0U, 0U); t += 10U; }
    assert(Mission_GetState() == MISSION_FAULT && t - 10U >= 1990U && t - 10U <= 2010U);
    drive_off();
  } else if (!strcmp(name, "ramp_stop_is_immediate")) {
    start_forward(1000U);
    forward_steps(1010U, 1200U, &c);
    assert(TIM2->CCR3 > 0U && TIM2->CCR3 < BASE);
    Motor_Stop();
    drive_off();
    forward_steps(1210U, 1400U, &c); /* 정지 뒤 SysTick이 다시 켜지 않는다. */
    drive_off();
  } else if (!strcmp(name, "ramp_section_transition_keeps_speed")) {
    /* 다 가속한 뒤 구간 전환(정지 직후 같은 방향 출발)은 다시 0부터 가속하지 않는다. */
    start_forward(1000U);
    forward_steps(1010U, 1800U, &c);
    assert(TIM2->CCR2 + TIM2->CCR3 == 2U * BASE);
    host_time(1805U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_FORWARD, BASE);
    forward(BASE, BASE);
  } else if (!strcmp(name, "ramp_section_transition_mid_ramp_continues")) {
    /* 가속 도중 구간 전환이면 0으로 돌아가거나 기준으로 튀지 않고 같은 곡선을 이어간다. */
    start_forward(1000U);
    forward_steps(1010U, 1200U, &c);
    uint32_t before = TIM2->CCR3;
    host_time(1205U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_FORWARD, BASE);
    assert(TIM2->CCR3 >= before && TIM2->CCR3 < BASE && TIM2->CCR2 == TIM2->CCR3);
    c3 = 0U; /* 구간 시작이 TIM3를 0으로 되돌렸다. */
    forward_steps(1215U, 1490U, &c);
    assert(TIM2->CCR3 < BASE);
    forward_steps(1500U, 1600U, &c);
    assert(TIM2->CCR2 + TIM2->CCR3 == 2U * BASE);
  } else if (!strcmp(name, "ramp_restart_after_real_stop")) {
    /* 50 ms보다 길게 멈춘 뒤 출발하면(촬영 대기, 장애물 정지 뒤) 다시 시작값에서 가속한다. */
    start_forward(1000U);
    forward_steps(1010U, 1800U, &c);
    Motor_Stop();
    host_time(1900U); Drive_MarkMainAlive();
    Motor_Run(MOTOR_DIRECTION_FORWARD, BASE);
    assert(TIM2->CCR2 == FLOOR(BASE) && TIM2->CCR3 == FLOOR(BASE));
    forward_steps(1910U, 2150U, &c);
    assert(TIM2->CCR3 > 0U && TIM2->CCR3 < BASE);
  } else if (!strcmp(name, "ramp_direction_change_starts_from_floor")) {
    /* 방향이 바뀌면 직후라도 시작값에서 가속한다(후진 채널 CCR1/CCR4). */
    start_forward(1000U);
    forward_steps(1010U, 1800U, &c);
    Motor_Stop();
    host_time(1805U); Drive_MarkMainAlive();
    Motor_Run(MOTOR_DIRECTION_REVERSE, MISSION_REVERSE_PWM_DUTY);
    assert(TIM2->CCR1 == FLOOR(MISSION_REVERSE_PWM_DUTY) && TIM2->CCR4 == FLOOR(MISSION_REVERSE_PWM_DUTY));
    assert(TIM2->CCR2 == 0U && TIM2->CCR3 == 0U);
    for (unsigned t = 1815U; t <= 2055U; t += 10U) advance(t, MOTOR_DIRECTION_REVERSE);
    assert(TIM2->CCR4 > FLOOR(MISSION_REVERSE_PWM_DUTY) && TIM2->CCR4 < MISSION_REVERSE_PWM_DUTY && TIM2->CCR1 == TIM2->CCR4);
    assert(TIM2->CCR2 == 0U && TIM2->CCR3 == 0U);
  } else if (!strcmp(name, "ramp_carry_helper_table")) {
    /* 구간 위치 계산: (시작 CNT, 현재 CNT, 방향) -> 새 CNT. ARR=600. */
    const struct { unsigned start, current; int direction; unsigned expected; } cases[] = {
      {0U, 25U, 1, 25U},      /* 전진 경계 뒤 25 count 관성 -> 다음 구간에 넣음 */
      {0U, 590U, 1, 0U},      /* 전진 뒤 0 아래로 11 count 밀림 -> 구간 시작 */
      {600U, 580U, -1, 580U}, /* 후진 경계 뒤 20 count 관성 */
      {600U, 5U, -1, 600U},   /* 후진 뒤 반대로 밀림 -> 구간 시작(ARR) */
      {300U, 320U, 1, 320U},  /* 장애물 정지 중 20 count 더 굴러감 */
      {595U, 10U, 1, 600U},   /* 장애물 정지 중 경계를 넘음 -> 경계 직전, 다음 count에 구간 종료 */
      {10U, 590U, -1, 0U},    /* 후진 장애물 정지 중 경계를 넘음 -> 경계 직전 */
      {5U, 598U, 1, 0U},      /* 구간 시작 직후 뒤로 밀림 -> 시작으로 고정 */
      {300U, 300U, 1, 300U}, {300U, 300U, -1, 300U},
      {123U, 500U, 1, 123U},  /* 관성으로 볼 수 없는 큰 변화(377 count)는 넣지 않는다 */
      {0U, 151U, 1, 0U}, {0U, 150U, 1, 150U}, /* 상한 150 count 경계 */
    };
    for (unsigned n = 0U; n < sizeof(cases) / sizeof(cases[0]); ++n)
      assert(Drive_CarrySectionCount(cases[n].start, cases[n].current, cases[n].direction, 600U) == cases[n].expected);
  } else if (!strcmp(name, "ramp_carry_same_direction_section")) {
    /* ISR이 경계에서 멈춘 뒤 관성으로 25 count 굴러가면 다음 같은 방향 구간은 25에서 시작한다. */
    start_forward(1000U);
    forward_steps(1010U, 1800U, &c);
    Motor_Stop(); Drive_MarkSectionBoundary(MOTOR_DIRECTION_FORWARD);
    TIM3->CNT = 25U;
    host_time(5000U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_FORWARD, BASE);
    assert(TIM3->CNT == 25U);
    /* 표시는 한 번만 쓴다. 다음 구간 시작은 다시 0이다. */
    Motor_Stop(); TIM3->CNT = 40U;
    host_time(9000U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_FORWARD, BASE);
    assert(TIM3->CNT == 0U);
    /* 후진도 같다: 경계(ARR)에서 20 count 더 내려가면 580에서 시작한다. */
    Motor_Stop(); Drive_MarkSectionBoundary(MOTOR_DIRECTION_REVERSE); TIM3->CNT = 580U;
    host_time(13000U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_REVERSE, MISSION_REVERSE_PWM_DUTY);
    assert(TIM3->CNT == 580U);
  } else if (!strcmp(name, "ramp_carry_direction_change_and_mission_reset")) {
    start_forward(1000U);
    /* 방향이 바뀌면 같은 방향 이월 대신 첫 구간을 관성만큼 늘린다: 전진 관성 25 -> 후진 CNT 24,
     * 25 count 뒤 첫 경계 이벤트는 보정 이벤트(preroll)로 소비된다. */
    Motor_Stop(); Drive_MarkSectionBoundary(MOTOR_DIRECTION_FORWARD); TIM3->CNT = 25U;
    host_time(5000U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_REVERSE, MISSION_REVERSE_PWM_DUTY);
    assert(TIM3->CNT == 24U && Drive_ConsumeSectionPreroll() == 1 && Drive_ConsumeSectionPreroll() == 0);
    /* 후진 관성 15(CNT 585) -> 전진 CNT 586(=600-14), 15 count 뒤 보정 이벤트. */
    Motor_Stop(); Drive_MarkSectionBoundary(MOTOR_DIRECTION_REVERSE); TIM3->CNT = 585U;
    host_time(5500U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_FORWARD, BASE);
    assert(TIM3->CNT == 586U && Drive_ConsumeSectionPreroll() == 1);
    /* 관성이 0이면 보정 구간 없이 기존처럼 0/ARR에서 시작한다. */
    Motor_Stop(); Drive_MarkSectionBoundary(MOTOR_DIRECTION_FORWARD); TIM3->CNT = 0U;
    host_time(6000U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_REVERSE, MISSION_REVERSE_PWM_DUTY);
    assert(TIM3->CNT == 600U && Drive_ConsumeSectionPreroll() == 0);
    /* 반대로 밀린 경우(전진 뒤 CNT 590)도 관성 0으로 본다. */
    Motor_Stop(); Drive_MarkSectionBoundary(MOTOR_DIRECTION_FORWARD); TIM3->CNT = 590U;
    host_time(6500U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_REVERSE, MISSION_REVERSE_PWM_DUTY);
    assert(TIM3->CNT == 600U && Drive_ConsumeSectionPreroll() == 0);
    /* 경계 표시가 없으면(임무 첫 구간) 이월하지 않는다. */
    Motor_Stop(); TIM3->CNT = 25U;
    host_time(9000U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_FORWARD, BASE);
    assert(TIM3->CNT == 0U);
    /* 고장·대기·시험 전환은 이전 표시를 지운다. */
    Motor_Stop(); Drive_MarkSectionBoundary(MOTOR_DIRECTION_FORWARD);
    Mission_SetState(MISSION_IDLE); Mission_SetState(ACTUATOR_TEST);
    TIM3->CNT = 25U; host_time(13000U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_FORWARD, BASE);
    assert(TIM3->CNT == 0U);
    Motor_Stop(); Drive_MarkSectionBoundary(MOTOR_DIRECTION_FORWARD); Mission_LatchFault();
    host_reset(); Mission_SetState(ACTUATOR_TEST); TIM3->CNT = 25U; host_time(17000U); Drive_MarkMainAlive();
    Motor_StartNextEncoderSection(MOTOR_DIRECTION_FORWARD, BASE);
    assert(TIM3->CNT == 0U);
  } else assert(!"unknown ramp scenario");
}
