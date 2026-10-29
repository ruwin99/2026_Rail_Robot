#ifndef DRIVE_SYNC_PID_H
#define DRIVE_SYNC_PID_H

#include <stdint.h>
#include "drive_sync_config.h"

enum {
  DRIVE_SYNC_OK = 0,
  DRIVE_SYNC_BAD_CONFIG = 1,
  DRIVE_SYNC_LATE = 2,
  DRIVE_SYNC_ENCODER_RANGE = 3,
  DRIVE_SYNC_STALL = 4,
  DRIVE_SYNC_WRONG_DIRECTION = 5
};

typedef struct {
  int32_t kp_q8, ki_q8, kd_q8;
  int32_t max_correction, max_slew;
  uint32_t period_ms, max_gap_ms;
  uint32_t encoder1_modulus, encoder2_modulus;
  int32_t encoder1_scale, encoder2_scale;
  int32_t encoder1_sign, encoder2_sign;
  uint32_t stall_timeout_ms;
  uint32_t forward_right_target_bp;
  uint32_t reverse_left_target_bp;
} DriveSyncPidConfig;

typedef struct {
  uint32_t previous1, previous2, last_tick;
  int32_t integral_q8, previous_error, correction;
  int32_t integral_remainder;
  int64_t previous_error_bp;
  uint32_t no_motion_ms1, no_motion_ms2;
  int fault;
} DriveSyncPidState;

typedef struct {
  int32_t pwm1, pwm2, error, correction;
  int64_t error_bp; /* 내부 고정소수점 오차(count*10000). 기존 UART error 단위는 바꾸지 않는다. */
  int32_t delta1, delta2, normalized_right, normalized_left;
  uint32_t elapsed_ms;
  int valid, fault;
} DriveSyncPidOutput;

/* 상태는 호출자가 소유한다. 구현은 Core/Src/drive_sync_pid.c에 있다. */
DriveSyncPidConfig DriveSyncPid_DefaultConfig(void);
/* 기존 DefaultConfig는 전진 설정. 주행 요청은 방향별 설정을 선택한 뒤 상태를 Reset한다. */
DriveSyncPidConfig DriveSyncPid_DefaultConfigForDirection(int direction);

void DriveSyncPid_Reset(DriveSyncPidState *state,
                        uint32_t count1, uint32_t count2,
                        uint32_t now);

DriveSyncPidOutput DriveSyncPid_Step(
    DriveSyncPidState *state, const DriveSyncPidConfig *config,
    uint32_t count1, uint32_t count2, uint32_t now, int direction,
    int32_t base_pwm1, int32_t base_pwm2);

#endif
