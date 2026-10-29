#include "checks.h"
#include "ultrasonic.h"
#include "rgb_pwm.h"
#include "../fixtures/rr2_vectors.h"

static void pulse(unsigned us) {
  unsigned i;
  /* One fresh 60 ms ping; TIM2 IRQ supplies the trigger/timeout state machine. */
  for(i=0;i<1201 && !(GPIOB->ODR & GPIO_PIN_8);++i) Ultrasonic_Tick50us();
  assert(GPIOB->ODR & GPIO_PIN_8); Ultrasonic_Tick50us();
  Ultrasonic_OnEcho(1); DWT->CYCCNT += us*72U; Ultrasonic_OnEcho(0);
}
static void pulses(unsigned us, unsigned count) {
  for(unsigned i=0;i<count;++i) { pulse(us); Robot_Update(); }
}
void scenario_ultrasonic(void) {
  unsigned i; uint32_t logical;
  host_reset(); host_line("START\r\n");
  TIM3->CNT=123; TIM4->CNT=234;
  RGB_Init();
  /* 정지 거리 10 cm(584 us). 이전 기준 25 cm(1458 us) 물체는 더 이상 정지시키지 않는다. */
  pulses(1458,3); assert(!Ultrasonic_IsHeld());
  pulses(584,2); assert(!Ultrasonic_IsHeld());
  pulse(584); assert(Ultrasonic_IsHeld()); drive_off(); Robot_Update();
  assert(Mission_GetState()==INITIAL_SCAN);
  Robot_Update();
  for(unsigned tick=0;tick<255;++tick) {
    RGB_Tick50us();
    assert(GPIOB->BSRR==(RGB_PWM_ACTIVE_LOW ? (24U | 32U<<16) : (32U | 24U<<16)));
  }
  logical=Ultrasonic_OperationalTick();
  host_time(5000); Robot_Update();
  assert(Ultrasonic_OperationalTick()==logical); drive_off();
  Robot_OnEncoderUpdate(&htim3); assert(!Mission_HasEncoderEvent());
  Motor_Run(1,1600); Front_Pump_Start(); drive_off();
  assert(TIM1->CCR2==0);
  assert(strstr(host_output,"OBSTACLE_HOLD\r\n"));
  TIM3->CNT=500; TIM4->CNT=550;
  for(i=0;i<15;++i) { pulse(585); Robot_Update(); assert(Ultrasonic_IsHeld()); drive_off(); }
  pulse(585); Robot_Update(); assert(!Ultrasonic_IsHeld());
  assert(TIM3->CNT==123 && TIM4->CNT==234); forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
  assert(strstr(host_output,"OBSTACLE_CLEAR\r\n"));
  pulses(584,20); assert(Ultrasonic_IsHeld()); Robot_EmergencyStop();
  pulses(2000,20);
  assert(Mission_GetState()==MISSION_FAULT); all_off();
}

static void clear_window(void) { pulses(2000,20); }
static void no_echo_sample(void) {
  while (!(GPIOB->ODR&GPIO_PIN_8)) Ultrasonic_Tick50us();
  Ultrasonic_Tick50us();
  for(unsigned i=0;i<600;++i) Ultrasonic_Tick50us();
  Robot_Update();
}
static unsigned events(const char *token) {
  unsigned count=0; const char *at=host_output;
  while ((at=strstr(at,token))!=NULL) { ++count; at+=strlen(token); }
  return count;
}
void scenario_ultrasonic_prestart(void) {
  host_reset(); RGB_Init();
  pulses(467,20);
  assert(Mission_GetState()==MISSION_IDLE);
  assert(!Ultrasonic_IsHeld() && !Ultrasonic_HasObstacle());
  assert(events("OBSTACLE_HOLD\r\n")==0);
  host_time(250); Robot_Update();
  unsigned blue_on=0;
  for(unsigned i=0;i<255;++i) {
    RGB_Tick50us();
    blue_on += (RGB_PWM_ACTIVE_LOW ? (GPIOB->BSRR >> 16) : GPIOB->BSRR) & 8U ? 1U : 0U;
  }
  assert(blue_on==0U); /* No status LED before the MISSION session. */
  host_line("START\r\n");
  assert(Mission_GetState()==INITIAL_SCAN);
  assert(Ultrasonic_IsHeld() && Ultrasonic_HasObstacle());
  drive_off();
  pulses(2000,16);
  assert(!Ultrasonic_IsHeld());
  forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
  assert(events("OBSTACLE_HOLD\r\n")==1 && events("OBSTACLE_CLEAR\r\n")==1);

  host_reset(); RGB_Init();
  assert(!Ultrasonic_IsHeld());
  host_line("START\r\n");
  assert(Mission_GetState()==INITIAL_SCAN);
  forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);

  host_reset(); Ultrasonic_Init();
  host_line("START\r\n");
  assert(Mission_GetState()==INITIAL_SCAN && Ultrasonic_IsHeld());
  drive_off();
  pulses(2000,19); assert(Ultrasonic_IsHeld()); drive_off();
  pulses(2000,1); assert(!Ultrasonic_IsHeld());
  forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);

  host_reset(); Ultrasonic_Init();
  host_line("ACTUATOR_TEST_START\r\n");
  assert(Mission_GetState()==ACTUATOR_TEST && Ultrasonic_IsHeld());
  host_line("FRONT_PUMP_ON\r\n"); all_off();
  pulses(2000,20); assert(!Ultrasonic_IsHeld());
  host_line("FRONT_PUMP_ON\r\n");
  assert(Actuators_GetOutputMask() & 4U);

  host_reset(); Ultrasonic_Init();
  host_line("START\r\n"); assert(Ultrasonic_IsHeld());
  Mission_SetState(MISSION_IDLE); assert(!Ultrasonic_IsHeld());
  pulses(2000,20);
  host_line("START\r\n");
  TIM3->CNT=123U;
  pulses(467,16); assert(Ultrasonic_IsHeld());
  pulses(2000,16); assert(!Ultrasonic_IsHeld());
  assert(TIM3->CNT==123U); forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
}
void scenario_ultrasonic_states(void) {
  const MissionState states[]={MISSION_IDLE,INITIAL_SCAN,WAIT_CAPTURE_ACK,RETURNING,
    REALTIME_CLEANING,ACTUATOR_TEST,DEBUG_TEST,RESCAN_RETURN,RESCAN,MISSION_DONE,MISSION_FAULT,FINAL_RETURN};
  for(unsigned n=0;n<sizeof(states)/sizeof(states[0]);++n) {
    host_reset(); Drive_SetDirection(1);
    if(states[n]==WAIT_CAPTURE_ACK) Mission_SetState(INITIAL_SCAN);
    Mission_SetState(states[n]);
    if(states[n]==REALTIME_CLEANING || states[n]==ACTUATOR_TEST || states[n]==DEBUG_TEST) {
      Front_Pump_Start(); Side_Cleaner_Run(900);
    }
    uint32_t drive_before=TIM2->CCR1+TIM2->CCR2+TIM2->CCR3+TIM2->CCR4;
    for(unsigned i=0;i<16;++i) pulse(584);
    all_off();
    Robot_Update(); assert(Mission_GetState()==states[n]);
    host_time(4000); Robot_Update(); assert(Mission_GetState()==states[n]); all_off();
    host_feed("FRONT_PUMP_ON\r\n");
    clear_window();
    assert(!Ultrasonic_IsHeld());
    assert(Actuators_GetOutputMask()==0 && TIM1->CCR1==0 && TIM1->CCR2==0 && TIM1->CCR3==0 && TIM1->CCR4==0);
    if(drive_before && states[n]!=DEBUG_TEST && states[n]!=ACTUATOR_TEST) assert(TIM2->CCR1+TIM2->CCR2+TIM2->CCR3+TIM2->CCR4>0);
    else drive_off();
    if(states[n]==MISSION_IDLE || states[n]==MISSION_FAULT)
      assert(events("OBSTACLE_HOLD\r\n")==0 && events("OBSTACLE_CLEAR\r\n")==0);
    else
      assert(events("OBSTACLE_HOLD\r\n")==1 && events("OBSTACLE_CLEAR\r\n")==1);
    if(states[n]==REALTIME_CLEANING || states[n]==ACTUATOR_TEST) {
      host_line("FRONT_PUMP_ON\r\n"); assert(Actuators_GetOutputMask()&4U);
    }
  }
  host_reset(); host_line("START\r\n");
  for(unsigned i=0;i<ENCODER_SECTION_LIMIT;++i) { Robot_OnEncoderUpdate(&htim3); Robot_Update(); host_line("CAPTURE_OK\r\n"); }
  assert(Mission_IsTransitionPending()); host_time(250); Robot_Update();
  pulses(584,16); host_time(5250); Robot_Update();
  assert(Mission_IsTransitionPending() && Mission_GetState()==WAIT_CAPTURE_ACK);
  clear_window(); assert(Mission_IsTransitionPending());
  host_time(5999); Robot_Update(); assert(Mission_IsTransitionPending());
  host_time(6000); Robot_Update(); assert(Mission_GetState()==RETURNING);
  host_reset(); host_line("START\r\n"); Robot_OnEncoderUpdate(&htim3); Robot_Update();
  pulses(584,16); host_feed("CAPTURE_OK\r\n"); Robot_Update();
  assert(Mission_GetState()==WAIT_CAPTURE_ACK); clear_window();
  assert(Mission_GetState()==INITIAL_SCAN); forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
}
void scenario_ultrasonic_edges(void) {
  host_reset(); Ultrasonic_Init();
  assert(!Ultrasonic_IsHeld() && !Ultrasonic_HasObstacle());
  Ultrasonic_Arm(); assert(Ultrasonic_IsHeld() && !Ultrasonic_HasObstacle());
  /* 운용 정책: NO_ECHO는 장애물 근거가 아니며 시작 창의 원거리 결과로 센다. */
  for(unsigned n=0;n<19;++n) no_echo_sample();
  assert(Ultrasonic_IsHeld());
  no_echo_sample();
  assert(!Ultrasonic_IsHeld() && !strstr(host_output,"OBSTACLE_"));
  Ultrasonic_OnEcho(0); assert(!Ultrasonic_IsHeld());
  /* 정지는 근거리 연속 3회다. 원거리나 NO_ECHO가 끼면 연속 횟수가 다시 시작한다. */
  pulses(584,2); pulses(585,1); pulses(584,2); assert(!Ultrasonic_IsHeld());
  no_echo_sample(); pulses(584,2); assert(!Ultrasonic_IsHeld());
  pulses(467,1); assert(Ultrasonic_IsHeld());
  assert(events("OBSTACLE_HOLD\r\n")==1);
  /* 해제 기준은 기존과 같다: 정지 이후 원거리 16/20. NO_ECHO도 해제 근거로 센다. */
  pulses(2000,14); assert(Ultrasonic_IsHeld());
  DWT->CYCCNT=UINT32_MAX-100U; pulses(2000,1); assert(Ultrasonic_IsHeld());
  no_echo_sample(); assert(!Ultrasonic_IsHeld());
  assert(events("OBSTACLE_CLEAR\r\n")==1);
  /* 근거리 연속 3회는 창의 원거리 개수와 무관하게 즉시 정지한다. */
  host_reset(); Ultrasonic_Arm(); assert(!Ultrasonic_IsHeld());
  for(unsigned i=0;i<6;++i) { pulses(584,2); pulses(2000,1); }
  assert(!Ultrasonic_IsHeld()); /* 근거리 12, 원거리 6이지만 연속 3회 없음 */
  pulses(584,3); assert(Ultrasonic_IsHeld());
  for(unsigned i=0;i<4;++i) { pulses(2000,1); pulses(584,1); }
  pulses(2000,11); assert(Ultrasonic_IsHeld()); /* 원거리 15, 근거리 5 */
  pulses(2000,1); assert(!Ultrasonic_IsHeld()); /* 원거리 16, 근거리 4 */
  /* Explicit stop/state cancellation during hold must invalidate drive resume. */
  host_line("START\r\n"); pulses(584,20); Mission_SetState(MISSION_IDLE); clear_window(); drive_off();
  host_reset(); host_line("START\r\n"); host_time(UINT32_MAX-500U);
  pulses(584,16); uint32_t frozen=Ultrasonic_OperationalTick();
  host_time(100U); Robot_Update(); assert(Ultrasonic_OperationalTick()==frozen);
  clear_window(); assert(Ultrasonic_OperationalTick()==frozen);
  /* Model a section already selected by the FSM before echo preemption. */
  pulses(584,20); Motor_StartNextEncoderSection(1,1600); drive_off();
  clear_window(); assert(TIM3->CNT==0); forward(right_base_pwm(1600),1600);
  pulses(584,20); Motor_Run(0,0); clear_window(); drive_off();
  pulses(467,16); assert(Ultrasonic_IsHeld()); drive_off();
}
void scenario_ultrasonic_mission_gate(void) {
  host_reset(); RGB_Init();
  pulses(467,20); /* A near window collected before rail_mission connects. */
  assert(Mission_GetState()==MISSION_IDLE);
  assert(!Ultrasonic_IsHeld() && !Ultrasonic_HasObstacle());

  host_line(RR2_VECTOR_HELLO_WIRE);
  assert(UART_V2Active() && UART_RR2GetCapability()==UART_RR2_CAP_MISSION);
  host_line(RR2_VECTOR_CMD_WIRE);
  assert(Mission_GetState()==INITIAL_SCAN);
  /* Old near samples are discarded at MISSION-session activation. Until a
   * fresh far window arrives, the fail-safe unknown-range inhibit remains. */
  assert(Ultrasonic_IsHeld() && !Ultrasonic_HasObstacle());
  drive_off();
  pulses(2000,20);
  assert(!Ultrasonic_IsHeld() && !Ultrasonic_HasObstacle());
  forward(right_base_pwm(MISSION_FORWARD_PWM_DUTY),MISSION_FORWARD_PWM_DUTY);
  pulses(584,20);
  assert(Ultrasonic_IsHeld() && Ultrasonic_HasObstacle());
  drive_off();
}
