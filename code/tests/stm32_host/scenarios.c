#include "checks.h"
#include "ultrasonic.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
void scenario_ultrasonic(void);
void scenario_ultrasonic_states(void);
void scenario_ultrasonic_edges(void);
void scenario_ultrasonic_prestart(void);
void scenario_ultrasonic_mission_gate(void);
void scenario_rgb(void);
void scenario_rgb_status(void);
void scenario_rgb_mission_gate(void);
void scenario_rgb_idle_session(void);
void scenario_rgb_full_mission(void);
void scenario_sensor_debug(const char *name);
void scenario_debug_diag(const char *name);
void scenario_control(const char *name);
void scenario_uart_v2(const char *name);
void scenario_fault(const char *name);
void scenario_ramp(const char *name);
/* Interactive host-HAL bridge: stdin carries one RR2 UART line at a time.
 * !ENCODER is a test-only stand-in for a physical encoder interrupt. The
 * compiled robot application still parses, dispatches, and emits every RR2
 * frame. Stdout contains only its UART TX bytes, flushed after each step. */
static void rr2_ultrasonic_pulse(unsigned echo_us) {
  unsigned ticks = 0U;
  while ((GPIOB->ODR & GPIO_PIN_8) == 0U && ticks++ < 1201U)
    Ultrasonic_Tick50us();
  assert((GPIOB->ODR & GPIO_PIN_8) != 0U);
  Ultrasonic_Tick50us();
  Ultrasonic_OnEcho(1);
  DWT->CYCCNT += echo_us * 72U;
  Ultrasonic_OnEcho(0);
}

static void rr2_stdio(void) {
  char input[256];
#ifdef _WIN32
  /* Windows CRT text mode would silently turn UART CRLF into LF. */
  assert(_setmode(_fileno(stdin), _O_BINARY) != -1);
  assert(_setmode(_fileno(stdout), _O_BINARY) != -1);
#endif
  host_reset();
  host_output[0] = '\0';
  while (fgets(input, sizeof(input), stdin) != NULL) {
    if (strncmp(input, "RR2 ", 4U) == 0 &&
        strncmp(input, "RR2 HELLO ", 10U) != 0)
      host_time(host_tick + 10U);
    if (strncmp(input, "!ENCODER", 8U) == 0 && (input[8] == '\n' || input[8] == ' ')) {
      /* "!ENCODER <cnt>": 경계 정지 직후 관성으로 TIM3가 cnt까지 더 굴러간 상황을 만든다. */
      unsigned coast_count = 0U;
      int has_count = input[8] == ' ' && sscanf(input, "!ENCODER %u", &coast_count) == 1;
      Robot_OnEncoderUpdate(&htim3);
      if (has_count) TIM3->CNT = coast_count;
      host_drain();
    } else if (strncmp(input, "!TIME ", 6U) == 0) {
      unsigned delta = 0U;
      assert(sscanf(input, "!TIME %u", &delta) == 1 && delta <= 200U);
      host_time(host_tick + delta);
      host_drain();
    } else if (strncmp(input, "!ULTRASONIC ", 12U) == 0) {
      unsigned echo_us = 0U;
      assert(sscanf(input, "!ULTRASONIC %u", &echo_us) == 1 && echo_us > 0U);
      rr2_ultrasonic_pulse(echo_us);
      host_drain();
    } else {
      size_t length = strlen(input);
      if (length == 0U || input[length - 1U] != '\n') {
        fputs("host input exceeded line cap\n", stderr);
        return;
      }
      host_line(input);
    }
    if (host_output[0] != '\0') {
      fputs(host_output, stdout);
      fflush(stdout);
      host_output[0] = '\0';
    }
  }
}
int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1],"rr2_stdio")) {
    rr2_stdio();
    return 0;
  } else if (!strcmp(argv[1],"boot")) {
    host_reset(); all_off(); assert(Mission_GetState()==MISSION_IDLE);
    assert(host_receive_calls==1 && host_primask==0);
    assert(GPIOA->ODR==0 && GPIOB->ODR==0 && GPIOC->ODR==0);
  } else if (!strcmp(argv[1],"fatal_early")) {
    htim1.Instance=NULL; htim2.Instance=NULL;
    Robot_FatalStop(); assert(Mission_GetState()==MISSION_FAULT);
    htim1.Instance=TIM1; htim2.Instance=TIM2;
    TIM2->CCR1=TIM2->CCR2=TIM2->CCR3=TIM2->CCR4=100;
    Robot_FatalStop(); drive_off();
  } else if (!strcmp(argv[1],"fatal_partial")) {
    /* Accessory PWM initialized; drive initialization has not started. */
    htim2.Instance=NULL; Actuators_Init(); Front_Pump_Start(); assert(TIM1->CCR1==WATER_PUMP_PWM_DUTY);
    Robot_FatalStop(); assert(Actuators_GetOutputMask()==0 && TIM1->CCR1==0);
    assert(Mission_GetState()==MISSION_FAULT);
  } else if (!strcmp(argv[1],"fatal_initialized")) {
    host_reset(); Mission_SetState(ACTUATOR_TEST);
    Front_Cleaner_Run(1200); Front_Pump_Start(); Side_Cleaner_Run(2000); Side_Pump_Start();
    Motor_Run(1,800); Robot_FatalStop(); all_off();
    assert(Mission_GetState()==MISSION_FAULT && Drive_GetDirection()==0);
  } else if (!strncmp(argv[1],"control_",8)) scenario_control(argv[1]);
  else if (!strncmp(argv[1],"uart_v2_",8)) scenario_uart_v2(argv[1]);
  else if (!strncmp(argv[1],"fault_",6)) scenario_fault(argv[1]);
  else if (!strncmp(argv[1],"ramp_",5)) scenario_ramp(argv[1]);
  else if (!strncmp(argv[1],"debug_diag_",11)) scenario_debug_diag(argv[1]);
  else if (!strncmp(argv[1],"sensor_debug_",13)) scenario_sensor_debug(argv[1]);
  else if (!strncmp(argv[1],"debug_",6)) scenario_debug(argv[1]);
  else if (!strncmp(argv[1],"actuator_",9)) scenario_actuator(argv[1]);
  else if (!strncmp(argv[1],"mission_",8)) scenario_mission(argv[1]);
  else if (!strcmp(argv[1],"rgb_full_mission")) scenario_rgb_full_mission();
  else if (!strcmp(argv[1],"rgb_status")) scenario_rgb_status();
  else if (!strcmp(argv[1],"rgb_mission_gate")) scenario_rgb_mission_gate();
  else if (!strcmp(argv[1],"rgb_idle_session")) scenario_rgb_idle_session();
  else if (!strcmp(argv[1],"rgb_pwm")) scenario_rgb();
  else if (!strcmp(argv[1],"ultrasonic_states")) scenario_ultrasonic_states();
  else if (!strcmp(argv[1],"ultrasonic_edges")) scenario_ultrasonic_edges();
  else if (!strcmp(argv[1],"ultrasonic_prestart")) scenario_ultrasonic_prestart();
  else if (!strcmp(argv[1],"ultrasonic_mission_gate")) scenario_ultrasonic_mission_gate();
  else if (!strcmp(argv[1],"ultrasonic")) scenario_ultrasonic();
  else scenario_runtime(argv[1]);
  printf("PASS %s\n",argv[1]);
  return 0;
}
