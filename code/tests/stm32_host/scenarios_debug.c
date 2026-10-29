#include "checks.h"
static void begin(int capability) {
  char hello[96]; host_reset();
  snprintf(hello,sizeof(hello),"DBG1_HELLO " SID " %d\r\n",capability);
  host_line(hello); assert(Mission_GetState()==DEBUG_TEST);
  assert(strstr(host_output,"DBG1_READY " SID)); all_off();
}
void scenario_debug(const char *name) {
  if (!strcmp(name,"debug_pwm")) {
    begin(7); host_line("DBG1_SET " SID " 1 1 10000 9\r\n");
    forward(right_base_pwm(1600),1600); assert(Actuators_GetOutputMask()==9);
    assert(TIM1->CCR4==FRONT_DC_MOTOR_PWM_DUTY_12_5 && TIM1->CCR3==WATER_PUMP_PWM_DUTY);
    host_time(50); host_line("DBG1_SET " SID " 2 1 5000 6\r\n");
    forward(right_base_pwm(800),800); assert(Actuators_GetOutputMask()==6);
    assert(TIM1->CCR4==0 && TIM1->CCR3==0 && TIM1->CCR1==WATER_PUMP_PWM_DUTY && TIM1->CCR2==SIDE_DC_MOTOR_PWM_DUTY_12_5);
    host_time(100); host_line("DBG1_SET " SID " 3 0 0 0\r\n"); all_off();
  } else if (!strcmp(name,"debug_reverse")) {
    begin(7); host_line("DBG1_SET " SID " 1 1 10000 8\r\n");
    host_time(10); host_line("DBG1_SET " SID " 2 -1 10000 8\r\n");
    drive_off(); assert(Drive_GetDirection()==0 && Actuators_GetOutputMask()==8);
    host_time(109); host_line("DBG1_SET " SID " 3 -1 10000 8\r\n"); drive_off();
    host_time(110); host_line("DBG1_SET " SID " 4 -1 10000 8\r\n"); reverse(right_reverse_base_pwm(1600),1600);
  } else if (!strcmp(name,"debug_expiry")) {
    begin(7); host_line("DBG1_SET " SID " 1 1 10000 15\r\n");
    uint32_t nonzero=host_nonzero_writes;
    host_time(700); host_line("DBG1_SET " SID " 2 1 10000 15\r\n");
    all_off(); assert(Mission_GetState()==MISSION_IDLE && strstr(host_output,"DBG1_TIMEOUT"));
    host_line("DBG1_SET " SID " 3 1 10000 15\r\n"); all_off();
    assert(host_nonzero_writes==nonzero);
  } else if (!strcmp(name,"debug_replay")) {
    begin(7); host_line("DBG1_SET " SID " 1 1 10000 15\r\n");
    uint32_t nonzero=host_nonzero_writes;
    host_time(599); host_line("DBG1_SET " SID " 1 1 10000 15\r\n"); assert(Mission_GetState()==DEBUG_TEST);
    host_time(600); Robot_Update(); assert(Mission_GetState()==MISSION_IDLE); all_off();
    assert(host_nonzero_writes==nonzero);
    begin(7); host_line("DBG1_SET " SID " 1 1 10000 15\r\n");
    host_line("DBG1_CLOSE " SID "\r\n"); all_off(); assert(Mission_GetState()==MISSION_IDLE);
    nonzero=host_nonzero_writes;
    host_line("DBG1_SET " SID " 2 1 10000 15\r\n"); all_off();
    assert(host_nonzero_writes==nonzero);
    Mission_SetState(REALTIME_CLEANING); host_line("DBG1_HELLO " SID " 7\r\n");
    assert(Mission_GetState()==REALTIME_CLEANING && strstr(host_output,"DBG1_REJECT"));
  } else if (!strcmp(name,"debug_malformed")) {
    static const char *bad[]={"DBG1_SET " SID " 1 1 10001 0\r\n",
      "DBG1_SET " SID " 1 1 10000 0 junk\r\n","DBG1_SET " SID " 4294967296 1 10000 0\r\n",
      "DBG1_SET " SID " -1 1 10000 0\r\n","DBG1_SET " SID " 1x 1 10000 0\r\n"};
    begin(2); host_line("DBG1_SET " SID " 1 1 10000 0\r\n");
    assert(Mission_GetState()==MISSION_FAULT); all_off();
    for (unsigned i=0;i<sizeof(bad)/sizeof(bad[0]);++i) {
      begin(7); host_line(bad[i]); all_off(); assert(Mission_GetState()==MISSION_FAULT);
      assert(host_nonzero_writes==0);
    }
    begin(7); host_line("DBG1_SET " SID " 4294967295 1 10000 0\r\n"); forward(right_base_pwm(1600),1600);
  } else if (!strcmp(name,"debug_framing")) {
    begin(7); host_line("DBG1_SET " SID " 1 1 10000 15\r\n");
    uint32_t nonzero=host_nonzero_writes;
    host_feed("DBG1_SET " SID " 2 1 10000 15\r\nDBG1_SET " SID " 3 1 10000 15\r\n");
    host_drain(); all_off(); assert(Mission_GetState()!=DEBUG_TEST);
    assert(host_nonzero_writes==nonzero);
    begin(7); host_feed("D"); for (int i=0;i<120;++i) host_feed("x");
    host_line("\r\n"); all_off(); assert(Mission_GetState()!=DEBUG_TEST);
    assert(host_nonzero_writes==0);
    begin(7); host_line("DBG1_SET " SID " 1 1 10000 15\r\n");
    Robot_OnUartError(&huart2); drive_off();
    nonzero=host_nonzero_writes;
    host_line("DBG1_SET " SID " 2 1 10000 15\r\n"); all_off();
    assert(host_nonzero_writes==nonzero);
    host_reset(); host_line("DBG1_JUNK_START\r\n"); assert(Mission_GetState()==MISSION_IDLE);
    host_line("START\r\n"); assert(Mission_GetState()==INITIAL_SCAN);
  } else assert(!"unknown debug scenario");
}
