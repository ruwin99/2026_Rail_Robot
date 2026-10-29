#include "checks.h"
static void begin(void) {
  host_reset(); host_line("ACTUATOR_TEST_START\r\n");
  assert(Mission_GetState()==ACTUATOR_TEST && strstr(host_output,"ACTUATOR_TEST_READY\r\n")); all_off();
}
void scenario_actuator(const char *name) {
  if (!strcmp(name,"actuator_roles")) {
    begin(); host_line("FRONT_CLEANER_PWM_12_5\r\n");
    assert(TIM1->CCR4==FRONT_DC_MOTOR_PWM_DUTY_12_5 && Actuators_GetOutputMask()==1 && GPIOB->ODR==GPIO_PIN_13);
    host_line("FRONT_CLEANER_PWM_25\r\n"); assert(TIM1->CCR4==DC_MOTOR_PWM_DUTY_25);
    host_line("FRONT_PUMP_ON\r\n"); assert(TIM1->CCR1==WATER_PUMP_PWM_DUTY && GPIOA->ODR==GPIO_PIN_5);
    host_line("SIDE_CLEANER_PWM_12_5\r\n"); assert(TIM1->CCR2==SIDE_DC_MOTOR_PWM_DUTY_12_5 && GPIOC->ODR==GPIO_PIN_13);
    host_line("SIDE_CLEANER_PWM_25\r\n"); assert(TIM1->CCR2==DC_MOTOR_PWM_DUTY_25);
    host_line("SIDE_PUMP_ON\r\n"); assert(TIM1->CCR3==WATER_PUMP_PWM_DUTY && GPIOC->ODR==(GPIO_PIN_13|GPIO_PIN_8));
    assert(Actuators_GetOutputMask()==15); drive_off();
    host_line("FRONT_CLEANER_OFF\r\n"); assert(Actuators_GetOutputMask()==14 && GPIOB->ODR==0);
    host_line("FRONT_PUMP_OFF\r\n"); assert(Actuators_GetOutputMask()==10 && GPIOA->ODR==0);
    host_line("SIDE_CLEANER_OFF\r\n"); assert(Actuators_GetOutputMask()==8 && GPIOC->ODR==GPIO_PIN_8);
    host_line("SIDE_PUMP_OFF\r\n"); all_off(); assert(GPIOC->ODR==0);
  } else if (!strcmp(name,"actuator_handshake")) {
    host_reset(); host_line("FRONT_CLEANER_PWM_25\r\nFRONT_PUMP_ON\r\nSIDE_PUMP_ON\r\n"); all_off();
    host_line("FRONT_PUMP_ON\r\nACTUATOR_TEST_START\r\nSIDE_PUMP_ON\r\n");
    all_off(); assert(Mission_GetState()==ACTUATOR_TEST);
    host_line("SIDE_PUMP_ON\r\n"); assert(Actuators_GetOutputMask()==8);
    host_line("ACTUATOR_TEST_STOP\r\nFRONT_PUMP_ON\r\n"); all_off(); assert(Mission_GetState()==MISSION_IDLE);
    host_line("START\r\n"); host_line("ACTUATOR_TEST_START\r\n"); assert(Mission_GetState()==INITIAL_SCAN);
  } else if (!strcmp(name,"actuator_legacy")) {
    begin(); host_line("CLEANER_ON\r\nCLEANER_PWM_12_5\r\nCLEANER_PWM_25\r\nCLEANER_OFF\r\nPUMP_ON\r\nPUMP_OFF\r\n"); all_off();
  } else if (!strcmp(name,"actuator_gate")) {
    host_reset(); host_line("START\r\n");
    host_line("FRONT_PUMP_ON\r\nSIDE_CLEANER_PWM_25\r\n"); assert(Actuators_GetOutputMask()==0);
    Robot_OnEncoderUpdate(&htim3); Robot_Update(); assert(Mission_GetState()==WAIT_CAPTURE_ACK);
    host_line("FRONT_PUMP_ON\r\n"); all_off();
    host_reset(); Mission_SetState(REALTIME_CLEANING); host_line("FRONT_PUMP_ON\r\n"); assert(Actuators_GetOutputMask()==4);
    for (int i=0;i<ENCODER_SECTION_LIMIT;++i) Robot_OnEncoderUpdate(&htim3);
    all_off(); assert(Mission_HasEncoderEvent()); host_line("FRONT_PUMP_ON\r\n"); all_off();
  } else if (!strcmp(name,"actuator_off")) {
    static const char *on[]={"FRONT_CLEANER_PWM_25\r\n","FRONT_PUMP_ON\r\n","SIDE_CLEANER_PWM_25\r\n","SIDE_PUMP_ON\r\n"};
    static const char *off[]={"FRONT_CLEANER_OFF\r\n","FRONT_PUMP_OFF\r\n","SIDE_CLEANER_OFF\r\n","SIDE_PUMP_OFF\r\n"};
    for (int role=0;role<4;++role) {
      begin(); for (int i=0;i<4;++i) host_line(on[i]);
      host_feed(on[role]); host_feed(off[role]); host_feed(on[role]); host_drain();
      assert(Actuators_GetOutputMask()==(15U & ~(const uint32_t[]){1,4,2,8}[role]));
    }
    begin(); host_line("FRONT_CLEANER_PWM_25\r\nFRONT_CLEANER_PWM_12_5\r\n"); assert(TIM1->CCR4==FRONT_DC_MOTOR_PWM_DUTY_12_5);
    host_line("SIDE_CLEANER_PWM_12_5\r\nSIDE_CLEANER_PWM_25\r\n"); assert(TIM1->CCR2==DC_MOTOR_PWM_DUTY_25);
    for (uint32_t masked=0;masked<=1;++masked) {
      host_primask=masked;
      Actuators_QueueCommand(ACTUATOR_FRONT_PUMP,ACTUATOR_COMMAND_OFF);
      Actuators_QueueCommand(ACTUATOR_FRONT_PUMP,ACTUATOR_COMMAND_ON);
      Actuators_Update(); assert(host_primask==masked && !(Actuators_GetOutputMask() & 4));
    }
    host_primask=0;
  } else if (!strcmp(name,"actuator_watchdogs")) {
    begin(); host_line("FRONT_CLEANER_PWM_12_5\r\n");
    host_time(100); host_line("FRONT_PUMP_ON\r\n"); host_time(200); host_line("SIDE_CLEANER_PWM_25\r\n");
    host_time(300); host_line("SIDE_PUMP_ON\r\n");
    host_time(999); Robot_Update(); assert(Actuators_GetOutputMask()==15);
    host_time(1000); Robot_Update(); assert(Actuators_GetOutputMask()==14);
    host_time(1099); Robot_Update(); assert(Actuators_GetOutputMask()==14);
    host_time(1100); Robot_Update(); assert(Actuators_GetOutputMask()==10);
    host_time(1200); Robot_Update(); assert(Actuators_GetOutputMask()==8);
    host_time(1299); host_line("SIDE_PUMP_ON\r\n");
    host_time(1300); Robot_Update(); assert(Actuators_GetOutputMask()==8);
    host_time(2299); Robot_Update(); all_off();
    host_time(UINT32_MAX-499U); host_line("FRONT_PUMP_ON\r\n");
    host_time(499); Robot_Update(); assert(Actuators_GetOutputMask()==4);
    host_time(500); Robot_Update(); all_off();
  } else if (!strcmp(name,"actuator_uart")) {
    for (int kind=0;kind<3;++kind) {
      begin(); host_line("FRONT_CLEANER_PWM_25\r\nFRONT_PUMP_ON\r\nSIDE_CLEANER_PWM_25\r\nSIDE_PUMP_ON\r\n");
      assert(Actuators_GetOutputMask()==15);
      if (kind==0) Robot_OnUartError(&huart2);
      else if (kind==1) { huart2.ErrorCode=1; host_feed("x"); huart2.ErrorCode=0; }
      else { host_receive_status=HAL_BUSY; host_feed("x"); }
      host_drain(); all_off();
      host_receive_status=HAL_OK; int before=host_receive_calls; Robot_Update();
      if (kind==2) assert(host_receive_calls>before);
      host_line("FRONT_PUMP_ON\r\n"); assert(Actuators_GetOutputMask()==4);
    }
  } else assert(!"unknown actuator scenario");
}
