#ifndef DEBUG_CONSOLE_H
#define DEBUG_CONSOLE_H

#include <stdint.h>

void Debug_Init(void);
void Debug_Update(void);
/* ISR에서 줄을 모으고, Update가 main에서 검증/실행/응답한다. */
int Debug_ParseByte(uint8_t byte);
void Debug_ResetRx(void);
void Debug_ObstacleBarrier(void);
void Debug_RequestAbort(void);
void Debug_Deactivate(void);
/* ISR에서 기록하고 main의 Debug_Update에서 DIAG/ERROR로 전송한다. */
void Debug_LatchDriveFault(int fault);
int Debug_IsSensorSession(void);
/* 요청된 고정 색상 또는 반복 효과의 현재 밝기를 반환한다. fault/장애물 빨강은 robot_app에서 우선한다. */
int Debug_GetLedOverride(uint8_t *red, uint8_t *green, uint8_t *blue);

#endif
