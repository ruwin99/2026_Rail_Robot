#ifndef FAULT_STOP_H
#define FAULT_STOP_H

#include "main.h"

#if defined(_MSC_VER)
#define FAULT_NORETURN __declspec(noreturn)
#elif defined(__GNUC__)
#define FAULT_NORETURN __attribute__((noreturn))
#else
#define FAULT_NORETURN
#endif

/* 독립 워치독(IWDG)을 약 0.5 s(LSI 30~60 kHz 편차로 0.33~0.67 s)로 시작한다.
 * 한 번 시작하면 리셋 외에는 멈출 수 없다. HAL IWDG 모듈 대신 레지스터를 직접 써서,
 * CubeMX로 코드를 다시 생성해도 stm32f1xx_hal_conf.h 설정에 의존하지 않는다. */
#define WATCHDOG_PRESCALER_DIV32 3U
#define WATCHDOG_RELOAD 625U
void Watchdog_Start(void);
/* Robot_Update 맨 앞에서 호출한다. main 루프가 멈추면 먹이가 끊겨 MCU가 리셋된다. */
void Watchdog_Feed(void);

/* 주행 TIM2와 청소·펌프 TIM1의 PWM을 0으로, 방향 핀을 LOW로 만든다.
 * HAL 호출·전역 상태·스택 변수 없이 상수 주소에만 쓰므로 폴트 핸들러에서도 안전하다. */
void Fault_OutputsOff(void);
/* NMI·HardFault·MemManage·BusFault·UsageFault 핸들러와 Error_Handler 전용이다.
 * 인터럽트를 막고 출력을 끈 뒤 빨간 LED를 켠 채 멈춘다. 워치독 먹이는 계속 주므로
 * 전원을 끌 때까지 래치된다(자동 재부팅으로 폴트 흔적이 사라지지 않게 한다). */
FAULT_NORETURN void Fault_StopAndHalt(void);

#endif
