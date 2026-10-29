#include "checks.h"
#include "rgb_pwm.h"

static unsigned sample(void) {
  unsigned set, reset, previous=GPIOB->ODR;
  RGB_Tick50us();
  set = GPIOB->BSRR & 0xffffU;
  reset = GPIOB->BSRR >> 16;
  assert((set | reset) == 56U && (set & reset) == 0U);
  GPIOB->ODR = (GPIOB->ODR | set) & ~reset;
  assert((GPIOB->ODR & ~56U) == (previous & ~56U));
  return (GPIOB->ODR ^ (RGB_PWM_ACTIVE_LOW ? 56U : 0U)) & 56U;
}

void scenario_rgb(void) {
  unsigned d, bit, tick, out, expected;
  host_reset(); GPIOB->ODR=0x6000U; RGB_Init();
  assert(GPIOB->BSRR==(RGB_PWM_ACTIVE_LOW ? 56U : 56U<<16));
  for (d=0;d<256;++d) {
    RGB_Set((uint8_t)d,(uint8_t)(255-d),85);
    for (bit=0;bit<8;++bit) {
      expected=((d&(1U<<bit)) ? 32U : 0U) |
               (((255-d)&(1U<<bit)) ? 16U : 0U) |
               ((85U&(1U<<bit)) ? 8U : 0U);
      for (tick=0;tick<(1U<<bit);++tick) assert(sample()==expected);
    }
  }
  RGB_Set(255,0,0); assert(sample()==32U);
  RGB_Set(0,255,0); RGB_Set(0,0,255);
  for (tick=1;tick<255;++tick) assert(sample()==32U);
  for (tick=0;tick<255;++tick) assert(sample()==8U);
  RGB_Set(0,0,0);
  for (tick=0;tick<255;++tick) { out=sample(); assert(out==0); }
}

static void colour(unsigned r,unsigned g,unsigned b) {
  unsigned n, out, cr=0,cg=0,cb=0;
  for(n=0;n<255;++n) { out=sample(); cr+=(out&32U)!=0; cg+=(out&16U)!=0; cb+=(out&8U)!=0; }
  if (cr!=r || cg!=g || cb!=b)
    fprintf(stderr,"RGB expected %u,%u,%u got %u,%u,%u at tick %lu state %d prepared %d\n",
            r,g,b,cr,cg,cb,(unsigned long)host_tick,(int)Mission_GetState(),Mission_IsPrepared());
  assert(cr==r && cg==g && cb==b);
}
void scenario_rgb_status(void) {
  unsigned i;
  host_reset(); GPIOB->ODR=0x6000U; RGB_Init(); Robot_Update(); colour(0,0,0);
  host_time(250); Robot_Update(); colour(0,0,0);
  host_time(500); Robot_Update(); colour(0,0,0);
  host_time(750); Robot_Update(); colour(0,0,0);
  host_time(1000); Robot_Update(); colour(0,0,0);
  host_line("MISSION_PREPARE\r\n");
  host_time(1250); Robot_Update(); colour(0,0,0);
  host_time(1500); Robot_Update(); colour(0,0,0);
  host_line("MISSION_CANCEL_PREPARE\r\n"); colour(0,0,0);
  host_time(2000); Robot_Update(); colour(0,0,0);
  host_line("MISSION_PREPARE\r\n");
  host_line("START\r\n"); colour(0,0,0);
  host_time(2250); Robot_Update(); colour(0,127,0);
  for(i=0;i<ENCODER_SECTION_LIMIT;++i) { Robot_OnEncoderUpdate(&htim3); Robot_Update(); host_line("CAPTURE_OK\r\n"); }
  assert(Mission_IsTransitionPending()); colour(0,0,0);
  host_time(2500); Robot_Update(); colour(0,0,127);
  Mission_LatchFault(); Robot_Update(); colour(255,0,0);
  host_time(2999); Robot_Update(); colour(255,0,0);
  host_time(3000); Robot_Update(); colour(0,0,0);
  host_time(3500); Robot_Update(); colour(255,0,0);
  host_reset(); RGB_Init(); host_line("START\r\n"); Front_Pump_Start();
  host_time(123); Robot_EmergencyStop();
  assert(Mission_GetState()==MISSION_FAULT && host_primask==0); all_off();
  colour(255,0,0);
  host_time(623); Robot_Update(); colour(0,0,0);
  Mission_SetState(INITIAL_SCAN); assert(Mission_GetState()==MISSION_FAULT); all_off();
  host_primask=1; Robot_EmergencyStop(); assert(host_primask==1); host_primask=0;
  host_reset(); RGB_Init(); host_time(UINT32_MAX-249U); Robot_Init();
  colour(0,0,0); host_time(0); Robot_Update(); colour(0,0,0);
  RGB_ForceRed();
  assert(GPIOB->BSRR==(RGB_PWM_ACTIVE_LOW ? (24U | 32U<<16) : (32U | 24U<<16)));
  RGB_ForceOff();
  assert(GPIOB->BSRR==(RGB_PWM_ACTIVE_LOW ? 56U : 56U<<16));
}

void scenario_rgb_mission_gate(void) {
  host_reset(); RGB_Init();
  host_line("MISSION_PREPARE\r\n");
  Robot_Update();
  /* Prepared/ready indicators stay dark before a MISSION session exists. */
  colour(0,0,0);
  Mission_LatchFault(); Robot_Update();
  /* A latched fault remains visible even before a dashboard session. */
  colour(255,0,0);
}

static uint16_t rgb_fixture_crc(const char *data) {
  uint16_t crc = 0xffffU;
  while (*data) {
    crc ^= (uint16_t)(uint8_t)*data++ << 8;
    for (unsigned bit = 0; bit < 8; ++bit)
      crc = (uint16_t)((crc & 0x8000U) ? (crc << 1) ^ 0x1021U : crc << 1);
  }
  return crc;
}

static void rgb_rr2(const char *body) {
  char frame[150];
  snprintf(frame, sizeof(frame), "%s %04X\r\n", body, rgb_fixture_crc(body));
  host_line(frame);
}

void scenario_rgb_idle_session(void) {
  uint32_t epoch;
  host_reset(); RGB_Init(); Robot_Update(); colour(0,0,0);
  rgb_rr2("RR2 HELLO " SID " MISSION");
  assert(UART_V2Active() && Mission_GetState()==MISSION_IDLE && !Mission_IsPrepared());
  /* Connected, waiting: slow 2 s blue breathing (half the 1 s pause-breathing slope). */
  epoch=host_tick;
  host_time(epoch+250U); Robot_Update(); colour(0,0,63);
  host_time(epoch+500U); Robot_Update(); colour(0,0,127);
  /* PREPARE (START requested, readiness checks running): fast 200 ms on/off blue blink. */
  host_time(epoch+550U); rgb_rr2("RR2 CMD " SID " 1 PREPARE");
  assert(Mission_IsPrepared());
  host_time(epoch+650U); Robot_Update(); colour(0,0,255);
  host_time(epoch+850U); Robot_Update(); colour(0,0,0);
  host_time(epoch+950U); Robot_Update(); colour(0,0,255);
  /* CANCEL_PREPARE returns to the slow waiting indicator from a fresh phase. */
  host_time(epoch+1000U); rgb_rr2("RR2 CMD " SID " 2 CANCEL_PREPARE");
  assert(!Mission_IsPrepared());
  host_time(epoch+1250U); Robot_Update(); colour(0,0,63);
  /* Without PING the clean IDLE session ends (no FAULT) and the indicator goes dark again. */
  host_time(epoch+1700U); Robot_Update();
  assert(!UART_V2Active() && Mission_GetState()==MISSION_IDLE);
  host_time(epoch+1800U); Robot_Update(); colour(0,0,0);
}

/* Literal expectations deliberately do not reproduce the production colour formula. */
static void green_triangle(MissionState state) {
  uint32_t epoch=host_tick;
  assert(Mission_GetState()==state && !Mission_IsTransitionPending());
  colour(0,0,0);
  host_time(epoch+250U); Robot_Update(); colour(0,127,0);
  host_time(epoch+500U); Robot_Update(); colour(0,255,0);
  host_time(epoch+750U); Robot_Update(); colour(0,127,0);
  host_time(epoch+1000U); Robot_Update(); colour(0,0,0);
  assert(Mission_GetState()==state);
  /* Enter capture/section transitions away from zero to detect incorrect resets. */
  host_time(epoch+1250U); Robot_Update(); colour(0,127,0);
}

static void blue_pause(MissionState target) {
  uint32_t epoch=host_tick;
  assert(Mission_IsTransitionPending()); colour(0,0,0);
  host_time(epoch+250U); Robot_Update(); colour(0,0,127);
  host_time(epoch+500U); Robot_Update(); colour(0,0,255);
  host_time(epoch+750U); Robot_Update(); colour(0,0,127);
  host_time(epoch+999U); Robot_Update(); colour(0,0,0);
  assert(Mission_IsTransitionPending());
  host_time(epoch+1000U); Robot_Update(); colour(0,0,0);
  assert(Mission_GetState()==target && !Mission_IsTransitionPending());
}

static void rgb_captures(MissionState state) {
  unsigned section;
  for(section=0;section<ENCODER_SECTION_LIMIT;++section) {
    assert(Mission_GetState()==state);
    Robot_OnEncoderUpdate(&htim3); Robot_Update();
    assert(Mission_GetState()==WAIT_CAPTURE_ACK); colour(0,127,0);
    host_line("CAPTURE_OK\r\n");
    if(section+1<ENCODER_SECTION_LIMIT) { assert(Mission_GetState()==state); colour(0,127,0); }
  }
  assert(Mission_IsTransitionPending());
}

static void rgb_sections(MissionState state) {
  unsigned section;
  for(section=0;section<ENCODER_SECTION_LIMIT;++section) {
    assert(Mission_GetState()==state);
    Robot_OnEncoderUpdate(&htim3); Robot_Update();
    if(section+1<ENCODER_SECTION_LIMIT) { assert(!Mission_IsTransitionPending()); colour(0,127,0); }
  }
  assert(Mission_IsTransitionPending());
}

void scenario_rgb_full_mission(void) {
  unsigned i;
  const MissionState test_states[]={ACTUATOR_TEST,DEBUG_TEST};
  host_reset(); RGB_Init(); Robot_Update();
  host_time(137U); host_line("START\r\n");
  green_triangle(INITIAL_SCAN); rgb_captures(INITIAL_SCAN);
  blue_pause(RETURNING); green_triangle(RETURNING); rgb_sections(RETURNING);
  blue_pause(REALTIME_CLEANING); green_triangle(REALTIME_CLEANING); rgb_sections(REALTIME_CLEANING);
  blue_pause(RESCAN_RETURN); green_triangle(RESCAN_RETURN); rgb_sections(RESCAN_RETURN);
  blue_pause(RESCAN); green_triangle(RESCAN); rgb_captures(RESCAN);
  /* 마지막 출발점 복귀도 임무 진행(초록)이며, 앞뒤 1초 정지는 파란색이다. */
  blue_pause(FINAL_RETURN); green_triangle(FINAL_RETURN); rgb_sections(FINAL_RETURN);
  blue_pause(MISSION_DONE);
  Robot_Update(); assert(Mission_GetState()==MISSION_IDLE); colour(0,0,0);
  host_time(host_tick+250U); Robot_Update(); colour(0,0,0);
  for(i=0;i<2;++i) {
    host_reset(); RGB_Init(); Robot_Update();
    host_time(137U); Mission_SetState(test_states[i]); Robot_Update();
    green_triangle(test_states[i]);
  }
}
