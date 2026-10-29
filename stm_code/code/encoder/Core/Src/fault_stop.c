#include "fault_stop.h"
#include "robot_board.h"
#include "rgb_pwm.h"

/* IWDG 키 값(RM0008): 시작, 쓰기 허용, 카운터 재장전. */
#define IWDG_KEY_START 0xCCCCU
#define IWDG_KEY_ACCESS 0x5555U
#define IWDG_KEY_RELOAD 0xAAAAU
/* PVU/RVU가 풀리길 기다리는 상한. LSI 동기화는 보통 1 ms 안에 끝난다. */
#define WATCHDOG_SYNC_LOOPS 200000U

void Watchdog_Start(void)
{
  uint32_t guard = 0U;
  /* 디버거로 코어를 멈춘 동안에는 워치독도 멈춰 중단점에서 리셋되지 않는다. */
  DBGMCU->CR |= DBGMCU_CR_DBG_IWDG_STOP;
  IWDG->KR = IWDG_KEY_START;
  IWDG->KR = IWDG_KEY_ACCESS;
  IWDG->PR = WATCHDOG_PRESCALER_DIV32;
  IWDG->RLR = WATCHDOG_RELOAD;
  while ((IWDG->SR & (IWDG_SR_PVU | IWDG_SR_RVU)) != 0U && ++guard < WATCHDOG_SYNC_LOOPS)
  {
  }
  IWDG->KR = IWDG_KEY_RELOAD;
}

void Watchdog_Feed(void)
{
  IWDG->KR = IWDG_KEY_RELOAD;
}

void Fault_OutputsOff(void)
{
  /* Motor_Stop·CleaningActuators_StopAll과 같은 결과를 레지스터 직접 쓰기로 낸다.
   * PWM 모드 1에서 CCR=0이면 출력이 항상 비활성(LOW)이다. 방향 핀은 BSRR 상위 절반으로 끈다. */
  TIM2->CCR1 = 0U;
  TIM2->CCR2 = 0U;
  TIM2->CCR3 = 0U;
  TIM2->CCR4 = 0U;
  TIM1->CCR1 = 0U;
  TIM1->CCR2 = 0U;
  TIM1->CCR3 = 0U;
  TIM1->CCR4 = 0U;
  FRONT_CLEANER_GPIO_PORT->BSRR = ((uint32_t)(FRONT_CLEANER_IN1_PIN | FRONT_CLEANER_IN2_PIN)) << 16;
  FRONT_WATER_PUMP_GPIO_PORT->BSRR = ((uint32_t)(FRONT_WATER_PUMP_IN1_PIN | FRONT_WATER_PUMP_IN2_PIN)) << 16;
  /* 옆 청소기와 옆 펌프는 같은 포트(GPIOC)라 한 번에 끈다. */
  SIDE_CLEANER_GPIO_PORT->BSRR = ((uint32_t)(SIDE_CLEANER_IN1_PIN | SIDE_CLEANER_IN2_PIN |
                                             SIDE_WATER_PUMP_IN1_PIN | SIDE_WATER_PUMP_IN2_PIN)) << 16;
}

FAULT_NORETURN void Fault_StopAndHalt(void)
{
  /* 우선순위가 설정 가능한 폴트(MemManage 등)에서 호출돼도 TIM3 PID ISR이 끼어들어
   * 방금 끈 PWM을 다시 쓰지 못하게 먼저 막는다. */
  __disable_irq();
  Fault_OutputsOff();
  RGB_ForceRed();
  for (;;)
  {
    Watchdog_Feed();
  }
}
