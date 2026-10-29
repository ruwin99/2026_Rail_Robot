#include "checks.h"
#include "fault_stop.h"
#include "robot_board.h"

/* 폴트 핸들러가 부르는 Fault_OutputsOff()와 독립 워치독 설정·먹이 주기를 검증한다.
 * Fault_StopAndHalt()는 무한 루프라 호스트에서 실행하지 않고, 핸들러 연결은 소스 검사와 ARM 링크 시험이 맡는다. */

static void fill_outputs(void) {
  TIM2->CCR1 = 1600U; TIM2->CCR2 = 1587U; TIM2->CCR3 = 800U; TIM2->CCR4 = 794U;
  TIM1->CCR1 = 3000U; TIM1->CCR2 = 1000U; TIM1->CCR3 = 3000U; TIM1->CCR4 = 1000U;
  GPIOA->BSRR = GPIOB->BSRR = GPIOC->BSRR = 0U;
}

void scenario_fault(const char *name) {
  if (!strcmp(name, "fault_outputs_off")) {
    host_reset();
    fill_outputs();
    Fault_OutputsOff();
    /* 주행 네 채널과 청소·펌프 네 채널의 PWM이 모두 0이다. */
    drive_off();
    assert(TIM1->CCR1 == 0U && TIM1->CCR2 == 0U && TIM1->CCR3 == 0U && TIM1->CCR4 == 0U);
    /* 방향 핀은 BSRR 상위 절반(리셋)으로 꺼진다. 같은 포트의 핀은 한 번에 끈다. */
    assert(GPIOB->BSRR == (((uint32_t)(GPIO_PIN_13 | GPIO_PIN_14)) << 16));
    assert(GPIOA->BSRR == (((uint32_t)(GPIO_PIN_5 | GPIO_PIN_6)) << 16));
    assert(GPIOC->BSRR == (((uint32_t)(GPIO_PIN_13 | GPIO_PIN_14 | GPIO_PIN_8 | GPIO_PIN_9)) << 16));
    /* 상위 절반만 쓰므로 어떤 핀도 켜지지 않는다(SET 비트가 없다). */
    assert((GPIOA->BSRR & 0xffffU) == 0U && (GPIOB->BSRR & 0xffffU) == 0U && (GPIOC->BSRR & 0xffffU) == 0U);
  } else if (!strcmp(name, "fault_outputs_off_while_running")) {
    /* 실제 구동 중 상태에서 호출해도 같은 결과다(HAL·상태 변수에 의존하지 않는다). */
    host_reset();
    Mission_SetState(ACTUATOR_TEST);
    Motor_Run(1, 800);
    Front_Cleaner_Run(1200); Front_Pump_Start(); Side_Cleaner_Run(2000); Side_Pump_Start();
    assert(TIM2->CCR2 != 0U || TIM2->CCR3 != 0U);
    Fault_OutputsOff();
    drive_off();
    assert(TIM1->CCR1 == 0U && TIM1->CCR2 == 0U && TIM1->CCR3 == 0U && TIM1->CCR4 == 0U);
  } else if (!strcmp(name, "fault_watchdog")) {
    host_reset();
    Watchdog_Start();
    assert(IWDG->PR == WATCHDOG_PRESCALER_DIV32 && IWDG->RLR == WATCHDOG_RELOAD);
    /* 시작 뒤 마지막 쓰기는 재장전 키다. 디버거 정지 시 워치독을 멈추는 비트도 켜진다. */
    assert(IWDG->KR == 0xAAAAU && (DBGMCU->CR & DBGMCU_CR_DBG_IWDG_STOP) != 0U);
    /* LSI 30~60 kHz 편차에서도 정상 지연(UART 20 ms)보다 길고 1 s보다 짧다. */
    assert((WATCHDOG_RELOAD * 32U) / 60U >= 250U && (WATCHDOG_RELOAD * 32U) / 30U <= 800U);
    /* main 루프의 Robot_Update가 먹이를 준다. */
    IWDG->KR = 0U;
    Robot_Update();
    assert(IWDG->KR == 0xAAAAU);
    /* Robot_Update가 FAULT 상태에서도 먹이를 주므로 FAULT 표시는 리셋으로 사라지지 않는다. */
    Mission_LatchFault();
    IWDG->KR = 0U;
    Robot_Update();
    assert(Mission_GetState() == MISSION_FAULT && IWDG->KR == 0xAAAAU);
  } else assert(!"unknown fault scenario");
}
