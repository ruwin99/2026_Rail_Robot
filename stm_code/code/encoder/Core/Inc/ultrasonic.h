#ifndef ULTRASONIC_H
#define ULTRASONIC_H
#include <stdint.h>
typedef enum { ULTRASONIC_WAITING, ULTRASONIC_ECHO, ULTRASONIC_NO_ECHO } UltrasonicRangeStatus;
typedef struct {
  uint32_t distance_mm;
  uint8_t valid, detected, held;
  UltrasonicRangeStatus status;
} UltrasonicSnapshot;
/* main 문맥에서 원자적으로 복사한다. NO_ECHO는 센서 정상 판정이 아니다. */
void Ultrasonic_GetSnapshot(UltrasonicSnapshot *snapshot);
void Ultrasonic_Init(void);
/* Drop measurements collected before a newly accepted MISSION session. */
void Ultrasonic_ResetWindow(void);
/* Arm the obstacle interlock before a mission or powered test can start. */
void Ultrasonic_Arm(void);
/* IDLE and sensor-only debugging still measure distance without an interlock. */
void Ultrasonic_Disarm(void);
void Ultrasonic_Tick50us(void);
void Ultrasonic_OnEcho(int high);
void Ultrasonic_Update(void);
int Ultrasonic_IsHeld(void); /* Includes range-unknown inhibit once armed. */
int Ultrasonic_HasObstacle(void);
uint32_t Ultrasonic_OperationalTick(void);
#endif
