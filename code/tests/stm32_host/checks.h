#ifndef HOST_CHECKS_H
#define HOST_CHECKS_H
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "fake_hal.h"
#include "robot_app.h"
#include "drive_control.h"
#include "drive_sync_pid.h"
#include "actuators.h"
#include "mission.h"
#include "uart_protocol.h"
#include "robot_config.h"
/* Test oracle uses floating arithmetic, independent of production integer rounding. */
static int right_base_pwm(int base) {
  return (int)(base * (double)RIGHT_MOTOR_FORWARD_TRIM_BP / 10000.0 + 0.5);
}
#define SID "0123456789abcdef0123456789abcdef"
static int right_reverse_base_pwm(int base) {
  return (int)(base * (double)RIGHT_MOTOR_REVERSE_TRIM_BP / 10000.0 + 0.5);
}
static void drive_off(void) {
  assert(TIM2->CCR1==0 && TIM2->CCR2==0 && TIM2->CCR3==0 && TIM2->CCR4==0);
}
static void all_off(void) {
  drive_off(); assert(Actuators_GetOutputMask()==0);
  assert(TIM1->CCR1==0 && TIM1->CCR2==0 && TIM1->CCR3==0 && TIM1->CCR4==0);
}
/* 실패하면 호출한 줄과 기대·실제 CCR을 stderr에 남긴다. PWM 값을 조정한 뒤 어느 시나리오가
 * 옛 값을 기대하는지 바로 찾을 수 있다. */
static void drive_check_fail(const char *what, int line, int a, int b) {
  fprintf(stderr, "%s(%d,%d) failed at scenario line %d; TIM2 CCR1..4 = %u %u %u %u\n",
          what, a, b, line, (unsigned)TIM2->CCR1, (unsigned)TIM2->CCR2,
          (unsigned)TIM2->CCR3, (unsigned)TIM2->CCR4);
}
static void forward_at(int a,int b,int line) {
  if (!(TIM2->CCR1==0 && TIM2->CCR4==0 && TIM2->CCR2==(uint32_t)a && TIM2->CCR3==(uint32_t)b))
    drive_check_fail("forward", line, a, b);
  assert(TIM2->CCR1==0 && TIM2->CCR4==0 && TIM2->CCR2==(uint32_t)a && TIM2->CCR3==(uint32_t)b);
}
static void reverse_at(int a,int b,int line) {
  if (!(TIM2->CCR2==0 && TIM2->CCR3==0 && TIM2->CCR1==(uint32_t)a && TIM2->CCR4==(uint32_t)b))
    drive_check_fail("reverse", line, a, b);
  assert(TIM2->CCR2==0 && TIM2->CCR3==0 && TIM2->CCR1==(uint32_t)a && TIM2->CCR4==(uint32_t)b);
}
#define forward(a,b) forward_at((a),(b),__LINE__)
#define reverse(a,b) reverse_at((a),(b),__LINE__)
void scenario_debug(const char *name);
void scenario_actuator(const char *name);
void scenario_mission(const char *name);
void scenario_runtime(const char *name);
#endif
