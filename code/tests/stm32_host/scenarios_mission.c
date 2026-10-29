#include "checks.h"
static unsigned captures_sent(void) {
  unsigned count=0; const char *at=host_output;
  while ((at=strstr(at,"CAMERA_CAPTURE\r\n"))!=NULL) { ++count; ++at; }
  return count;
}
static void pause_to(MissionState target) {
  uint32_t started=host_tick; MissionState previous=Mission_GetState();
  assert(Mission_IsTransitionPending()); all_off(); assert(!(TIM3->DIER & TIM_IT_UPDATE));
  host_time(started+999U); Robot_Update();
  assert(Mission_GetState()==previous && Mission_IsTransitionPending()); all_off();
  host_time(started+1000U); Robot_Update();
  assert(Mission_GetState()==target && !Mission_IsTransitionPending() && host_primask==0);
}
static void captures(MissionState scan) {
  for (int section=1;section<=ENCODER_SECTION_LIMIT;++section) {
    unsigned previous_captures=captures_sent();
    assert(Mission_GetState()==scan); forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
    Robot_OnEncoderUpdate(&htim4); assert(!Mission_HasEncoderEvent());
    Robot_OnEncoderUpdate(&htim3); drive_off(); assert(Mission_HasEncoderEvent());
    assert(Mission_GetSectionCount()==section-1); Robot_Update();
    assert(Mission_GetState()==WAIT_CAPTURE_ACK && Mission_GetSectionCount()==section);
    assert(captures_sent()==previous_captures+1);
    if (section==ENCODER_SECTION_LIMIT) {
      /* A phase transition clears both UART and accessory pending ON commands. */
      host_feed("FRONT_PUMP_ON\r\n");
      Actuators_QueueCommand(ACTUATOR_FRONT_CLEANER,ACTUATOR_COMMAND_ON);
    }
    host_line("CAPTURE_OK\r\n");
    if (section<ENCODER_SECTION_LIMIT) assert(Mission_GetState()==scan && !Mission_IsTransitionPending());
  }
  assert(Mission_IsTransitionPending());
}
static void prepare_pause(void) { host_reset(); host_line("START\r\n"); captures(INITIAL_SCAN); }
static void continuous(MissionState state) {
  for (int i=1;i<=ENCODER_SECTION_LIMIT;++i) {
    Robot_OnEncoderUpdate(&htim4); assert(Mission_GetSectionCount()==i-1);
    Robot_OnEncoderUpdate(&htim3); assert(Mission_GetSectionCount()==i);
    if (i<ENCODER_SECTION_LIMIT) {
      assert(!Mission_HasEncoderEvent());
      if (state==REALTIME_CLEANING) forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY); else reverse(right_reverse_base_pwm(MISSION_REVERSE_PWM_DUTY),MISSION_REVERSE_PWM_DUTY);
      Robot_Update(); assert(!Mission_IsTransitionPending());
    } else { assert(Mission_HasEncoderEvent()); all_off(); Robot_Update(); }
  }
}
/* 전체 임무에서 매 정지 뒤 관성으로 TIM3가 더 굴러간 상황. 같은 단계 안의 다음 구간은 굴러간 만큼에서
 * 시작하고, 마지막 구간 뒤 단계가 바뀌면(방향 전환·연속 주행) 구간 수 0, CNT 0/600에서 다시 센다. */
static void coast_captures(MissionState scan, unsigned coast) {
  for (int section=1;section<=ENCODER_SECTION_LIMIT;++section) {
    assert(Mission_GetState()==scan && Mission_GetSectionCount()==section-1);
    Robot_OnEncoderUpdate(&htim3); drive_off();      /* 경계에서 ISR 정지 */
    TIM3->CNT=coast;                                 /* 경계 뒤 관성 */
    Robot_Update(); assert(Mission_GetState()==WAIT_CAPTURE_ACK && Mission_GetSectionCount()==section);
    host_line("CAPTURE_OK\r\n");
    if (section<ENCODER_SECTION_LIMIT) assert(Mission_GetState()==scan && TIM3->CNT==coast); /* 같은 방향: 이월 */
  }
  assert(Mission_IsTransitionPending());
}
/* 방향 전환 단계의 첫 경계 이벤트는 보정 이벤트다: 구간으로 세지 않고 주행을 유지한다. */
static void preroll_event(void) {
  Robot_OnEncoderUpdate(&htim3);
  assert(!Mission_HasEncoderEvent() && Mission_GetSectionCount()==0 && (TIM3->DIER & TIM_IT_UPDATE));
}
static void coast_continuous(MissionState state, unsigned coast) {
  for (int i=1;i<=ENCODER_SECTION_LIMIT;++i) {
    Robot_OnEncoderUpdate(&htim3); assert(Mission_GetSectionCount()==i);
    if (i<ENCODER_SECTION_LIMIT) Robot_Update(); else { all_off(); TIM3->CNT=coast; Robot_Update(); }
  }
  assert(Mission_IsTransitionPending());
}
/* 재검사 마지막 촬영 뒤 출발점으로 연속 후진 복귀하고 나서야 임무를 마친다(DONE).
 * 재검사 완료(RESCAN_DONE)는 복귀 출발 때 알린다. 방향 전환 관성이 있으면 첫 경계는 보정 이벤트다. */
static void final_return(int sections) {
  pause_to(FINAL_RETURN);
  assert(Drive_GetDirection()==-1 && Mission_GetSectionCount()==0 && (TIM3->DIER & TIM_IT_UPDATE));
  assert(strstr(host_output,"RESCAN_DONE\r\n") && !strstr(host_output,"\nDONE\r\n"));
  reverse(right_reverse_base_pwm(MISSION_REVERSE_PWM_DUTY),MISSION_REVERSE_PWM_DUTY);
  int calls=0;
  while (!Mission_HasEncoderEvent()) {
    assert(++calls<=sections+1);
    Robot_OnEncoderUpdate(&htim3);
    if (!Mission_HasEncoderEvent()) { reverse(right_reverse_base_pwm(MISSION_REVERSE_PWM_DUTY),MISSION_REVERSE_PWM_DUTY); Robot_Update(); }
  }
  assert(Mission_GetSectionCount()==sections && calls>=sections);
  all_off(); Robot_Update(); assert(!strstr(host_output,"\nDONE\r\n"));
  pause_to(MISSION_DONE); assert(strstr(host_output,"\nDONE\r\n"));
}
void scenario_mission(const char *name) {
  if (!strcmp(name,"mission_two_sections")) {
    /* 시험 모드: 모든 단계가 2구간이고 흐름(촬영·복귀·실시간·재검사 복귀·재검사, 1초 정지)은 같다. */
    host_reset();
    assert(!Mission_ApplyReliablePrepare(1,0) && !Mission_ApplyReliablePrepare(1,ENCODER_SECTION_LIMIT+1));
    assert(Mission_ApplyReliablePrepare(1,2)); host_line("START\r\n");
    assert(Mission_GetState()==INITIAL_SCAN && Mission_GetSectionLimit()==2);
    for (int section=1;section<=2;++section) {
      forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY); Robot_OnEncoderUpdate(&htim3); drive_off(); Robot_Update();
      assert(Mission_GetState()==WAIT_CAPTURE_ACK && Mission_GetSectionCount()==section);
      host_line("CAPTURE_OK\r\n");
    }
    pause_to(RETURNING);
    for (int i=1;i<=2;++i) {
      reverse(right_reverse_base_pwm(MISSION_REVERSE_PWM_DUTY),MISSION_REVERSE_PWM_DUTY); Robot_OnEncoderUpdate(&htim3); drive_off(); Robot_Update();
      assert(Mission_GetSectionCount()==i);
    }
    pause_to(REALTIME_CLEANING);
    Robot_OnEncoderUpdate(&htim3); assert(Mission_GetSectionCount()==1 && !Mission_HasEncoderEvent());
    forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
    Robot_OnEncoderUpdate(&htim3); assert(Mission_GetSectionCount()==2 && Mission_HasEncoderEvent());
    all_off(); Robot_Update(); pause_to(RESCAN_RETURN);
    Robot_OnEncoderUpdate(&htim3); assert(!Mission_HasEncoderEvent());
    Robot_OnEncoderUpdate(&htim3); assert(Mission_HasEncoderEvent()); all_off(); Robot_Update();
    pause_to(RESCAN);
    for (int section=1;section<=2;++section) {
      forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY); Robot_OnEncoderUpdate(&htim3); drive_off(); Robot_Update();
      assert(Mission_GetState()==WAIT_CAPTURE_ACK && Mission_GetSectionCount()==section);
      host_line("CAPTURE_OK\r\n");
    }
    final_return(2); all_off(); Robot_Update(); assert(Mission_GetState()==MISSION_IDLE);
    /* 다음 임무는 PREPARE 인자가 없으면 기본 구간 수(ENCODER_SECTION_LIMIT)다. */
    host_line("START\r\n"); assert(Mission_GetState()==INITIAL_SCAN && Mission_GetSectionLimit()==ENCODER_SECTION_LIMIT);
    /* 시험 PREPARE 뒤 START 없이 다른 상태를 거치면 요청값은 남지 않는다. */
    host_reset(); assert(Mission_ApplyReliablePrepare(1,2));
    Mission_SetState(ACTUATOR_TEST); Mission_SetState(MISSION_IDLE); host_line("START\r\n");
    assert(Mission_GetState()==INITIAL_SCAN && Mission_GetSectionLimit()==ENCODER_SECTION_LIMIT);
  } else
  if (!strcmp(name,"mission_coast_phase_reset")) {
    /* 단계가 바뀌면 구간 수는 0부터 다시 세고, CNT는 이전 방향 관성(coast)만큼 첫 구간을 늘린 값에서 시작한다.
     * coast 뒤의 첫 경계 이벤트는 보정 이벤트로 소비되고 구간으로 세지 않는다. */
    host_reset(); host_line("START\r\n"); assert(TIM3->CNT==0 && Mission_GetSectionCount()==0);
    coast_captures(INITIAL_SCAN, 20U);
    pause_to(RETURNING);                       /* 전진 관성 20 -> 후진 CNT 19 */
    assert(Mission_GetSectionCount()==0 && TIM3->CNT==19U && Drive_GetDirection()==-1);
    preroll_event(); reverse(right_reverse_base_pwm(MISSION_REVERSE_PWM_DUTY),MISSION_REVERSE_PWM_DUTY);
    for (int i=1;i<=ENCODER_SECTION_LIMIT;++i) {
      Robot_OnEncoderUpdate(&htim3); drive_off(); TIM3->CNT=585U; Robot_Update();
      assert(Mission_GetSectionCount()==i);
      if (i<ENCODER_SECTION_LIMIT) { assert(TIM3->CNT==585U); reverse(right_reverse_base_pwm(MISSION_REVERSE_PWM_DUTY),MISSION_REVERSE_PWM_DUTY); } /* 같은 방향: 이월 */
    }
    pause_to(REALTIME_CLEANING);               /* 후진 관성 15 -> 전진 CNT 586 */
    assert(Mission_GetSectionCount()==0 && TIM3->CNT==586U && Drive_GetDirection()==1);
    preroll_event(); forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
    coast_continuous(REALTIME_CLEANING, 30U);
    pause_to(RESCAN_RETURN);                   /* 전진 관성 30 -> 후진 CNT 29 */
    assert(Mission_GetSectionCount()==0 && TIM3->CNT==29U && Drive_GetDirection()==-1);
    preroll_event();
    coast_continuous(RESCAN_RETURN, 570U);
    pause_to(RESCAN);                          /* 후진 관성 31 -> 전진 CNT 571 */
    assert(Mission_GetSectionCount()==0 && TIM3->CNT==571U && Drive_GetDirection()==1);
    preroll_event();
    coast_captures(RESCAN, 12U);
    pause_to(FINAL_RETURN);                    /* 전진 관성 12 -> 후진 CNT 11 */
    assert(Mission_GetSectionCount()==0 && TIM3->CNT==11U && Drive_GetDirection()==-1);
    preroll_event();
    coast_continuous(FINAL_RETURN, 590U);
    pause_to(MISSION_DONE); all_off(); Robot_Update(); assert(Mission_GetState()==MISSION_IDLE);
    /* 다음 임무 첫 구간은 이전 임무의 관성을 쓰지 않는다. */
    TIM3->CNT=40U; host_line("START\r\n");
    assert(Mission_GetState()==INITIAL_SCAN && TIM3->CNT==0 && Mission_GetSectionCount()==0);
    Robot_OnEncoderUpdate(&htim3); assert(Mission_HasEncoderEvent()); /* 보정 이벤트가 남아 있지 않다 */
  } else
  if (!strcmp(name,"mission_full")) {
    prepare_pause(); assert(strstr(host_output,"STARTED\r\n")); pause_to(RETURNING);
    assert(Mission_GetSectionCount()==0 && TIM3->CNT==600 && Drive_GetDirection()==-1);
    for (int i=1;i<=ENCODER_SECTION_LIMIT;++i) {
      reverse(right_reverse_base_pwm(MISSION_REVERSE_PWM_DUTY),MISSION_REVERSE_PWM_DUTY); Robot_OnEncoderUpdate(&htim3); drive_off(); Robot_Update();
      assert(Mission_GetSectionCount()==i); if (i<ENCODER_SECTION_LIMIT) assert(!Mission_IsTransitionPending());
    }
    pause_to(REALTIME_CLEANING); forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
    host_line("FRONT_PUMP_ON\r\nSIDE_CLEANER_PWM_25\r\n"); assert(Actuators_GetOutputMask()==6);
    continuous(REALTIME_CLEANING); pause_to(RESCAN_RETURN);
    continuous(RESCAN_RETURN); pause_to(RESCAN); captures(RESCAN);
    pause_to(FINAL_RETURN);
    assert(strstr(host_output,"RESCAN_DONE\r\n") && !strstr(host_output,"\nDONE\r\n"));
    continuous(FINAL_RETURN); pause_to(MISSION_DONE);
    assert(strstr(host_output,"RETURN_START\r\n") && strstr(host_output,"REALTIME_START\r\n"));
    assert(strstr(host_output,"RESCAN_RETURN_START\r\n") && strstr(host_output,"RESCAN_START\r\n"));
    assert(strstr(host_output,"\nDONE\r\n")); all_off(); Robot_Update(); assert(Mission_GetState()==MISSION_IDLE);
  } else if (!strcmp(name,"mission_pending")) {
    prepare_pause(); uint32_t started=host_tick;
    host_time(started+500); host_line("FRONT_PUMP_ON\r\nSTART\r\nCAPTURE_OK\r\n"); all_off();
    assert(UART_TakePendingCommand()==UART_COMMAND_NONE); host_line("FRONT_PUMP_OFF\r\n"); all_off();
    host_time(started+999); Robot_Update(); assert(Mission_IsTransitionPending());
    host_time(started+1000); Robot_Update(); assert(Mission_GetState()==RETURNING);
  } else if (!strcmp(name,"mission_faults")) {
    for (int kind=0;kind<3;++kind) {
      prepare_pause();
      if (kind==0) Mission_LatchFault();
      else if (kind==1) Robot_OnUartError(&huart2);
      else { host_time(host_tick+1000); host_fault_on_tick=1; Robot_Update(); }
      host_time(host_tick+2000); Robot_Update();
      assert(Mission_GetState()==MISSION_FAULT && !Mission_IsTransitionPending()); all_off();
      for (uint32_t mask=0;mask<=1;++mask) {
        host_primask=mask; Mission_SetState(REALTIME_CLEANING);
        assert(host_primask==mask && Mission_GetState()==MISSION_FAULT); all_off();
      }
      host_primask=0;
    }
  } else if (!strcmp(name,"mission_wrap")) {
    host_reset(); host_time(UINT32_MAX-500U); host_line("START\r\n");
    captures(INITIAL_SCAN); pause_to(RETURNING); reverse(right_reverse_base_pwm(MISSION_REVERSE_PWM_DUTY),MISSION_REVERSE_PWM_DUTY);
  } else if (!strcmp(name,"mission_legacy_link")) {
    /* Legacy START without RR2: LINK_OK keeps driving, 1 s of silence stops and latches FAULT. */
    host_reset(); host_line("START\r\n"); assert(Mission_GetState()==INITIAL_SCAN);
    forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
    for (uint32_t t=200U;t<=3000U;t+=200U) { host_time(t); Robot_Update(); }
    assert(Mission_GetState()==INITIAL_SCAN); forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
    host_legacy_link=0;
    host_time(3999U); Robot_Update(); assert(Mission_GetState()==INITIAL_SCAN);
    host_time(4000U); Robot_Update(); assert(Mission_GetState()==MISSION_FAULT); all_off();
    assert(!(TIM3->DIER & TIM_IT_UPDATE));
    /* A late keepalive never revives the latched fault. */
    host_line("LINK_OK\r\n"); host_time(4200U); Robot_Update();
    assert(Mission_GetState()==MISSION_FAULT); all_off();
    /* IDLE and the stopped capture-ACK wait are not supervised by this rule. */
    host_reset(); host_legacy_link=0; host_time(10000U); Robot_Update();
    assert(Mission_GetState()==MISSION_IDLE);
    host_reset(); host_line("START\r\n");
    Robot_OnEncoderUpdate(&htim4); Robot_OnEncoderUpdate(&htim3); Robot_Update();
    assert(Mission_GetState()==WAIT_CAPTURE_ACK); drive_off();
    host_legacy_link=0; host_time(host_tick+5000U); Robot_Update();
    assert(Mission_GetState()==WAIT_CAPTURE_ACK); drive_off();
  } else assert(!"unknown mission scenario");
}
