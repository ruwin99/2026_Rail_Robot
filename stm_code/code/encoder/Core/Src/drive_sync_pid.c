/* 하드웨어에 접근하지 않는 정수 PID 계산. 상태는 호출자가 소유하며 IRQ 제어는 하지 않는다. */
#include "drive_sync_pid.h"

DriveSyncPidConfig DriveSyncPid_DefaultConfig(void)
{
  return (DriveSyncPidConfig){
    .kp_q8 = DRIVE_SYNC_PID_KP_Q8,
    .ki_q8 = DRIVE_SYNC_PID_KI_Q8,
    .kd_q8 = DRIVE_SYNC_PID_KD_Q8,
    .max_correction = DRIVE_SYNC_PID_MAX_CORRECTION,
    .max_slew = DRIVE_SYNC_PID_MAX_SLEW,
    .period_ms = DRIVE_SYNC_PID_PERIOD_MS,
    .max_gap_ms = DRIVE_SYNC_PID_MAX_GAP_MS,
    .encoder1_modulus = DRIVE_SYNC_PID_ENCODER1_MODULUS,
    .encoder2_modulus = DRIVE_SYNC_PID_ENCODER2_MODULUS,
    .encoder1_scale = DRIVE_SYNC_PID_ENCODER1_SCALE,
    .encoder2_scale = DRIVE_SYNC_PID_ENCODER2_SCALE,
    .encoder1_sign = DRIVE_SYNC_PID_ENCODER1_SIGN,
    .encoder2_sign = DRIVE_SYNC_PID_ENCODER2_SIGN,
    .stall_timeout_ms = DRIVE_SYNC_PID_STALL_TIMEOUT_MS,
    .forward_right_target_bp = DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_BP,
    .reverse_left_target_bp = DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_BP
  };
}

DriveSyncPidConfig DriveSyncPid_DefaultConfigForDirection(int direction)
{
  DriveSyncPidConfig config = DriveSyncPid_DefaultConfig();
  if (direction < 0) {
    config.kp_q8 = DRIVE_SYNC_PID_REVERSE_KP_Q8;
    config.ki_q8 = DRIVE_SYNC_PID_REVERSE_KI_Q8;
    config.kd_q8 = DRIVE_SYNC_PID_REVERSE_KD_Q8;
    config.max_correction = DRIVE_SYNC_PID_REVERSE_MAX_CORRECTION;
    config.max_slew = DRIVE_SYNC_PID_REVERSE_MAX_SLEW;
  }
  return config;
}

static int32_t DriveSyncPid_Clamp(int32_t value, int32_t low, int32_t high)
{
  return value < low ? low : (value > high ? high : value);
}

void DriveSyncPid_Reset(DriveSyncPidState *state,
                        uint32_t count1, uint32_t count2,
                        uint32_t now)
{
  *state = (DriveSyncPidState){0};
  state->previous1 = count1;
  state->previous2 = count2;
  state->last_tick = now;
}

static int32_t DriveSyncPid_Delta(uint32_t current, uint32_t previous,
                                  uint32_t modulus)
{
  int32_t delta = (int32_t)current - (int32_t)previous;
  if (delta > (int32_t)(modulus / 2U)) delta -= (int32_t)modulus;
  if (delta < -(int32_t)(modulus / 2U)) delta += (int32_t)modulus;
  return delta;
}

DriveSyncPidOutput DriveSyncPid_Step(
    DriveSyncPidState *state, const DriveSyncPidConfig *config,
    uint32_t count1, uint32_t count2, uint32_t now, int direction,
    int32_t base_pwm1, int32_t base_pwm2)
{
  DriveSyncPidOutput output = {0};
  uint32_t elapsed = now - state->last_tick;
  int32_t delta1, delta2, error, limit, candidate_i, raw, correction;
  int32_t next_remainder = 0;
  int64_t error_bp;
  int fractional;
  int64_t wide;
  output.elapsed_ms = elapsed;
  if (state->fault != DRIVE_SYNC_OK) {
    output.fault = state->fault;
    return output;
  }
  /* 게인은 표본 단위 Q8 값이다. 설정 범위를 제한하고 중간 곱셈은 int64로 계산한다. */
  if (config->period_ms == 0U || config->period_ms > 1000U ||
      config->max_gap_ms < config->period_ms || config->max_gap_ms > 2000U ||
      config->stall_timeout_ms < config->max_gap_ms || config->stall_timeout_ms > 60000U ||
      config->encoder1_modulus < 3U || config->encoder1_modulus > 65536U ||
      config->encoder2_modulus < 3U || config->encoder2_modulus > 65536U ||
      config->encoder1_scale < 1 || config->encoder1_scale > 16 ||
      config->encoder2_scale < 1 || config->encoder2_scale > 16 ||
      (config->encoder1_sign != 1 && config->encoder1_sign != -1) ||
      (config->encoder2_sign != 1 && config->encoder2_sign != -1) ||
      config->kp_q8 < 0 || config->kp_q8 > 65536 ||
      config->ki_q8 < 0 || config->ki_q8 > 65536 ||
      config->kd_q8 < 0 || config->kd_q8 > 65536 ||
      config->forward_right_target_bp < DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_MIN_BP ||
      config->forward_right_target_bp > DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_MAX_BP ||
      config->reverse_left_target_bp < DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_MIN_BP ||
      config->reverse_left_target_bp > DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_MAX_BP ||
      config->max_correction < 0 || config->max_correction > 3599 ||
      config->max_slew < 1 || config->max_slew > 3599 ||
      base_pwm1 < 0 || base_pwm1 > 3599 || base_pwm2 < 0 || base_pwm2 > 3599 ||
      (direction != 1 && direction != -1)) {
    state->fault = DRIVE_SYNC_BAD_CONFIG;
    output.fault = state->fault;
    return output;
  }
  if (elapsed < config->period_ms) return output;
  if (elapsed > config->max_gap_ms) {
    state->fault = DRIVE_SYNC_LATE;
    output.fault = state->fault;
    return output;
  }
  if (count1 >= config->encoder1_modulus || count2 >= config->encoder2_modulus ||
      state->previous1 >= config->encoder1_modulus || state->previous2 >= config->encoder2_modulus) {
    state->fault = DRIVE_SYNC_ENCODER_RANGE;
    output.fault = state->fault;
    return output;
  }
  /* 한 표본 간 실제 이동이 반 주기보다 작아야 shortest-wrap이 유일하다.
   * CNT만으로 한 바퀴 이상 누락을 알아낼 수 없으므로 실측 최고속도에서 이 조건을 검증해야 한다. */
  delta1 = DriveSyncPid_Delta(count1, state->previous1, config->encoder1_modulus);
  delta2 = DriveSyncPid_Delta(count2, state->previous2, config->encoder2_modulus);
  if ((delta1 >= (int32_t)(config->encoder1_modulus / 2U)) ||
      (delta1 <= -(int32_t)(config->encoder1_modulus / 2U)) ||
      (delta2 >= (int32_t)(config->encoder2_modulus / 2U)) ||
      (delta2 <= -(int32_t)(config->encoder2_modulus / 2U))) {
    state->fault = DRIVE_SYNC_ENCODER_RANGE;
    output.fault = state->fault;
    return output;
  }
  /* 표본당 역방향 한 raw count는 양자화/경계 잡음으로 허용한다.
   * 명확한 반대 방향 피드백을 보정 증폭으로 덮지 않고 방향/PPR 배선 점검이 필요한 오류로 고정한다. */
  output.delta1 = delta1;
  output.delta2 = delta2;
  output.normalized_right = delta1 * config->encoder1_scale * config->encoder1_sign * direction;
  output.normalized_left = delta2 * config->encoder2_scale * config->encoder2_sign * direction;
  if (delta1 * config->encoder1_sign * direction < -1 ||
      delta2 * config->encoder2_sign * direction < -1) {
    state->fault = DRIVE_SYNC_WRONG_DIRECTION;
    output.fault = state->fault;
    return output;
  }
  state->previous1 = count1;
  state->previous2 = count2;
  state->last_tick = now;
  state->no_motion_ms1 = delta1 == 0 && base_pwm1 > 0 ? state->no_motion_ms1 + elapsed : 0U;
  state->no_motion_ms2 = delta2 == 0 && base_pwm2 > 0 ? state->no_motion_ms2 + elapsed : 0U;
  if (state->no_motion_ms1 >= config->stall_timeout_ms ||
      state->no_motion_ms2 >= config->stall_timeout_ms) {
    state->fault = DRIVE_SYNC_STALL;
    output.fault = state->fault;
    return output;
  }
  /* 같은 기계 해상도에서 TIM3 TI1(x2)와 TIM4 TI12(x4)를 같은 단위로 만든다.
   * 방향을 곱해 후진도 진행방향 속도로 비교한다. 배선 변경 시 설정 헤더의 부호를 재확인한다. */
  wide = ((int64_t)delta1 * config->encoder1_scale * config->encoder1_sign -
          (int64_t)delta2 * config->encoder2_scale * config->encoder2_sign) * direction;
  error = (int32_t)(wide * config->period_ms / elapsed);
  /* 물리 delta/정규화 telemetry와 안전 판정은 그대로 두고 방향별 제어 대응만 적용한다. */
  if (direction > 0) error *= DRIVE_SYNC_PID_FORWARD_FEEDBACK_SIGN;
  else error *= DRIVE_SYNC_PID_REVERSE_FEEDBACK_SIGN;
  /* 해당 방향 목표가 중립10000이면 기존 정수 경로를 bit-exact하게 유지한다.
   * 전진은 R-qL, 후진은 qR-L: 후진 sign=-1 적용후 L-qR이 되어 목표 L=qR이다.
   * 반전 feedback 연결에서 양의 correction은 PWM1을 줄이고 PWM2를 늘린다.
   * 이 보정 방향을 목표 속도와 구분하고 실제 장치 응답은 별도 확인한다. */
  fractional = direction > 0 ? config->forward_right_target_bp != 10000U :
                              config->reverse_left_target_bp != 10000U;
  error_bp = (int64_t)error * 10000;
  if (fractional) {
    if (direction > 0) {
      wide = ((int64_t)output.normalized_right * 10000 -
              (int64_t)output.normalized_left * config->forward_right_target_bp);
      error_bp = (wide * config->period_ms / elapsed) * DRIVE_SYNC_PID_FORWARD_FEEDBACK_SIGN;
    } else {
      wide = ((int64_t)output.normalized_right * config->reverse_left_target_bp -
              (int64_t)output.normalized_left * 10000);
      error_bp = (wide * config->period_ms / elapsed) * DRIVE_SYNC_PID_REVERSE_FEEDBACK_SIGN;
    }
    error = (int32_t)(error_bp / 10000);
  }
  output.error_bp = error_bp;
  limit = config->max_correction;
  if (limit > base_pwm1 / 4) limit = base_pwm1 / 4;
  if (limit > base_pwm2 / 4) limit = base_pwm2 / 4;
  if (limit > 3599 - base_pwm1) limit = 3599 - base_pwm1;
  if (limit > 3599 - base_pwm2) limit = 3599 - base_pwm2;
  /* 게인은 기준 period_ms 단위다. 실제 경과시간에 맞춰 I는 누적하고 D는 나눈다.
   * elapsed == period_ms이면 기존 정주기 계산과 동일하다. */
  if (fractional) {
    int64_t denominator = (int64_t)10000 * config->period_ms;
    wide = (int64_t)config->ki_q8 * error_bp * elapsed + state->integral_remainder;
    next_remainder = (int32_t)(wide % denominator);
    wide = (int64_t)state->integral_q8 + wide / denominator;
  } else {
    wide = (int64_t)state->integral_q8 +
           (int64_t)config->ki_q8 * error * elapsed / config->period_ms;
  }
  candidate_i = wide > limit * 256 ? limit * 256 :
                (wide < -limit * 256 ? -limit * 256 : (int32_t)wide);
  if (wide != candidate_i) next_remainder = 0;
  if (fractional) {
    wide = (int64_t)config->kp_q8 * error_bp / 10000 + candidate_i +
           (int64_t)config->kd_q8 * (error_bp - state->previous_error_bp) *
           config->period_ms / ((int64_t)elapsed * 10000);
  } else {
    wide = (int64_t)config->kp_q8 * error + candidate_i +
           (int64_t)config->kd_q8 * (error - state->previous_error) *
           config->period_ms / elapsed;
  }
  raw = (int32_t)(wide / 256);
  /* 출력 포화와 같은 방향으로 오차가 커질 때 적분을 멈춰 windup을 제한한다.
   * 보정은 기준 PWM의 최대25% 이내이며, 표본당 변화량도 제한한다. */
  if (!((raw > limit && error_bp > 0) || (raw < -limit && error_bp < 0))) {
    state->integral_q8 = candidate_i;
    state->integral_remainder = next_remainder;
  }
  correction = DriveSyncPid_Clamp(raw, -limit, limit);
  correction = DriveSyncPid_Clamp(correction,
      state->correction - config->max_slew, state->correction + config->max_slew);
  correction = DriveSyncPid_Clamp(correction, -limit, limit);
  state->correction = correction;
  state->previous_error = error;
  state->previous_error_bp = error_bp;
  output.pwm1 = base_pwm1 - correction;
  output.pwm2 = base_pwm2 + correction;
  /* 공개 error는 호환용 정수다. -0.04count는0으로 보이지만 내부I/잔여에는 보존된다. */
  output.error = error;
  output.correction = correction;
  output.valid = 1;
  return output;
}
