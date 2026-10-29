#ifndef DRIVE_CONTROL_H
#define DRIVE_CONTROL_H

#include <stdint.h>

/* 실제 제어 계산의 최신 한 표본. raw delta와 정규화 값은 표본 구간 기준이며 RPM이 아니다.
 * flags: 1=PI 보정 활성, 2=현재 목표에 유효한 표본, 4=실제 PWM 적용 완료.
 * STOP/새 목표는 valid/applied를 지운다. sequence는 계산 시도마다 증가한다. */
typedef struct {
  uint32_t sequence, tick, elapsed_ms, flags;
  int32_t direction, delta1, delta2, normalized_right, normalized_left;
  int32_t error, integral_q8, correction;
} DriveControlSnapshot;
void Drive_GetControlSnapshot(DriveControlSnapshot *snapshot);

void Drive_Init(void);
void Drive_ObstaclePause(void);
void Drive_ObstacleResume(void);
void Drive_MarkMainAlive(void);
int Drive_GetFault(void);
int Drive_GetDirection(void);
/* 기존 명령 방향 표시는 PWM 정지와 별도로 갱신한다. */
void Drive_SetDirection(int direction);
void Motor_Run(int direction, int pwm_duty);
void Motor_Stop(void);
void Motor_StartNextEncoderSection(int direction, int pwm_duty);
/* 정지 뒤 굴러간 거리(관성)를 구간 위치에 반영한다.
 * start_count에서 current_count까지 움직인 양을 진행 방향 기준 구간 위치(0..arr)로 바꿔 CNT를 돌려준다.
 * 경계를 넘었으면 경계 직전(다음 한 count에 구간 종료), 뒤로 밀렸다면 구간 시작으로 맞춘다. */
uint32_t Drive_CarrySectionCount(uint32_t start_count, uint32_t current_count, int direction, uint32_t arr);
/* TIM3 ISR이 구간 경계에서 멈출 때 표시한다. 다음 같은 방향 구간은 경계 뒤 굴러간 만큼을 넣고 시작한다. */
void Drive_MarkSectionBoundary(int direction);
/* 임무 시작·중단·시험 전환 때 이전 임무의 경계 표시와 보정 구간을 지운다. */
void Drive_ClearSectionCarry(void);
/* 방향 전환 첫 구간의 보정 이벤트(이전 방향 관성만큼 되돌아온 지점)이면 1을 돌려주고 지운다.
 * TIM3 ISR은 이 이벤트를 구간으로 세지 않는다. */
int Drive_ConsumeSectionPreroll(void);
/* 장애물 정지 순간에 주행 중이던 방향. 주행 중이 아니었다면 0. */
int Drive_GetObstacleResumeDirection(void);
/* SysTick ISR 전용. PID 비활성 빌드에서는 아무 동작도 하지 않는다. */
void DriveSync_Tick1ms(void);

#endif
