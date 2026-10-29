/* 주행 PWM, PID 실행 상태, 방향 표시를 이 파일에서 소유한다.
 * main은 목표/STOP을 요청하고 SysTick은 계산한다. generation과 PRIMASK로 늦은 출력을 차단한다. */
#include "drive_control.h"
#include "drive_sync_pid.h"
#include "robot_board.h"
#include "robot_config.h"
#include "actuators.h"
#include "mission.h"
#include "debug_console.h"
#include "ultrasonic.h"

#include <string.h>

static int motor_direction;
static DriveControlSnapshot control_snapshot;
static int obstacle_resume_duty, obstacle_resume_direction, obstacle_resume_section;
static volatile int section_carry_direction, section_preroll_pending;

uint32_t Drive_CarrySectionCount(uint32_t start_count, uint32_t current_count, int direction, uint32_t arr)
{
  /* 구간 위치: 전진은 CNT 그대로(0→ARR 증가), 후진은 ARR-CNT(ARR→0 감소). 경계는 위치 ARR 다음 한 count다.
   * 이동량은 ARR+1 주기의 최단 거리로 계산한다(정지 뒤 반 바퀴 이상 구르지 않는다는 전제). */
  int32_t modulus = (int32_t)arr + 1;
  int32_t delta = (int32_t)current_count - (int32_t)start_count;
  int32_t position;
  if (delta > modulus / 2) delta -= modulus;
  if (delta < -(modulus / 2)) delta += modulus;
  if (delta > ENCODER_COAST_CARRY_MAX_COUNTS || delta < -ENCODER_COAST_CARRY_MAX_COUNTS) delta = 0;
  if (direction > 0) position = (int32_t)start_count + delta;
  else position = ((int32_t)arr - (int32_t)start_count) - delta;
  if (position < 0) position = 0;
  if (position > (int32_t)arr) position = (int32_t)arr;
  return direction > 0 ? (uint32_t)position : arr - (uint32_t)position;
}

void Drive_MarkSectionBoundary(int direction) { section_carry_direction = direction; }
void Drive_ClearSectionCarry(void)
{
  section_carry_direction = MOTOR_DIRECTION_STOPPED;
  section_preroll_pending = 0;
}
int Drive_ConsumeSectionPreroll(void)
{
  if (!section_preroll_pending) return 0;
  section_preroll_pending = 0;
  return 1;
}
int Drive_GetObstacleResumeDirection(void)
{
  return obstacle_resume_duty > 0 ? obstacle_resume_direction : MOTOR_DIRECTION_STOPPED;
}
#if DRIVE_SYNC_PID_ENABLED
static struct {
  DriveSyncPidState state;
  DriveSyncPidConfig config;
  volatile uint32_t generation, main_cycle, peak_cycles, peak_gap_cycles;
  uint32_t sample_cycle, sample_ms, cycles_per_ms;
  volatile int active, ready, fault;
  int direction, base1, base2;
  /* 출발 가속 상태와, 구간 전환을 정지 후 출발과 구별하기 위한 마지막 주행 기록. */
  int ramp_active, last_direction;
  uint32_t ramp_start_ms, last_drive_ms;
} drive_sync;
#endif

void Motor_Stop(void)
{
  obstacle_resume_duty=0;
  obstacle_resume_section=0;
  /* 주행 TIM2의 양방향 입력 네 개를 모두 0으로 만든다. motor_direction 변수는 호출자가 따로 관리한다.
   * 정지 판정 시 변수만 보지 말고 CCR1~4도 확인해야 한다. 청소/펌프 OFF는 별도 함수다. */
#if DRIVE_SYNC_PID_ENABLED
  uint32_t primask = __get_PRIMASK();
  __disable_irq();
  drive_sync.active = 0;
  drive_sync.generation++;
  control_snapshot.flags &= 1U;
#endif
  if (htim2.Instance == TIM2)
  {
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, 0);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 0);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 0);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, 0);
  }
#if DRIVE_SYNC_PID_ENABLED
  if (primask == 0U) __enable_irq();
#endif
}

#if DRIVE_SYNC_PID_ENABLED
static void DriveSync_WritePwm(int direction, int pwm1, int pwm2)
{
  /* 같은 방향 제어 중 반대 채널은 계속 0이며 PID 갱신마다 PWM을 끊지 않는다. */
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, direction < 0 ? pwm1 : 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, direction > 0 ? pwm1 : 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, direction > 0 ? pwm2 : 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, direction < 0 ? pwm2 : 0);
  if (pwm1 > 0 || pwm2 > 0) {
    drive_sync.last_direction = direction;
    drive_sync.last_drive_ms = HAL_GetTick();
  }
}

#if DRIVE_SOFT_START_MS > 0U
/* 시작값(기준의 FLOOR_PERCENT)에서 기준 PWM까지 smoothstep 3t^2-2t^3을 Q16 정수로 계산한다.
 * t=0이면 시작값, t=끝이면 기준 PWM과 같다. */
static int DriveSync_RampPwm(int base, uint32_t elapsed_ms)
{
  uint32_t floor_pwm = ((uint32_t)base * DRIVE_SOFT_START_FLOOR_PERCENT + 50U) / 100U;
  uint64_t x = ((uint64_t)elapsed_ms << 16) / DRIVE_SOFT_START_MS;
  uint64_t s = (x * x * ((3ULL << 16) - 2ULL * x)) >> 32;
  return (int)(floor_pwm + (((uint64_t)((uint32_t)base - floor_pwm) * s + 32768ULL) >> 16));
}

/* 가속 중에는 보정을 쓰지 않으므로 적분·보정·정지 시간도 쌓지 않는다.
 * 가속이 끝난 뒤 PID는 보정 0에서 slew 제한을 지키며 시작한다. */
static void DriveSync_HoldPidDuringRamp(void)
{
  drive_sync.state.integral_q8 = 0;
  drive_sync.state.integral_remainder = 0;
  drive_sync.state.correction = 0;
  drive_sync.state.previous_error = 0;
  drive_sync.state.previous_error_bp = 0;
  drive_sync.state.no_motion_ms1 = 0U;
  drive_sync.state.no_motion_ms2 = 0U;
}
#endif

static void DriveSync_Fault(int fault)
{
  /* UART 송신 없이 고장 상태를 고정한다. main 정체/센서 누락 후 늦은 ON으로 재시작하지 않는다. */
  drive_sync.fault = fault;
  Motor_Stop();
  CleaningActuators_StopAll();
  __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);
  Drive_SetDirection(MOTOR_DIRECTION_STOPPED);
  Mission_LatchFault();
  Debug_LatchDriveFault(fault);
  Debug_Deactivate();
}

static void DriveSync_Request(int direction, int pwm1, int pwm2)
{
  uint32_t primask = __get_PRIMASK();
  __disable_irq();
  if (Ultrasonic_IsHeld() || drive_sync.ready == 0 || drive_sync.fault != 0 ||
      Mission_GetState() == MISSION_FAULT || Mission_IsTransitionPending() != 0 ||
      (direction != 1 && direction != -1) || pwm1 <= 0 || pwm2 <= 0) {
    Motor_Stop();
  } else if (pwm1 > 3599 || pwm2 > 3599) {
    DriveSync_Fault(DRIVE_SYNC_BAD_CONFIG);
  } else {
    if (drive_sync.active == 0 || drive_sync.direction != direction ||
        drive_sync.base1 != pwm1 || drive_sync.base2 != pwm2) {
      Motor_Stop();
      drive_sync.direction = direction;
      drive_sync.base1 = pwm1;
      drive_sync.base2 = pwm2;
      drive_sync.sample_cycle = DWT->CYCCNT;
      drive_sync.sample_ms = HAL_GetTick();
      /* 새 목표마다 해당 방향의 게인/제한을 선택하므로 전후진 설정이 서로 섞이지 않는다.
       * 후진 보정 OFF는 게인만 0으로 만든다. Step과 주기/엔코더/정지 보호는
       * 그대로 실행하며, 아래 Reset으로 이전 방향의 적분과 보정량도 지운다. */
      drive_sync.config = DriveSyncPid_DefaultConfigForDirection(direction);
#if !DRIVE_SYNC_PID_REVERSE_ENABLED
      if (direction < 0) {
        drive_sync.config.kp_q8 = 0;
        drive_sync.config.ki_q8 = 0;
        drive_sync.config.kd_q8 = 0;
      }
#endif
      DriveSyncPid_Reset(&drive_sync.state, __HAL_TIM_GET_COUNTER(&htim3),
                         __HAL_TIM_GET_COUNTER(&htim4), drive_sync.sample_ms);
      drive_sync.active = 1;
      control_snapshot = (DriveControlSnapshot){
        .sequence = control_snapshot.sequence, .direction = direction,
        .flags = (drive_sync.config.kp_q8 || drive_sync.config.ki_q8 || drive_sync.config.kd_q8) ? 1U : 0U
      };
#if DRIVE_SOFT_START_MS > 0U
      /* 같은 방향으로 방금까지 달렸다면(구간 전환, 같은 방향 속도 변경) 기존처럼 바로 기준 PWM을 쓴다.
       * 그 밖의 출발(정지·촬영 대기·장애물 정지 뒤)은 시작값에서 가속한다. 목표 PWM과 방향 검사는 위에서 끝났다. */
      /* 이어서 출발할 때 가속이 진행 중이었다면 시작 시각을 유지해 곡선을 이어간다. */
      if (!(drive_sync.last_direction == direction &&
            (uint32_t)(drive_sync.sample_ms - drive_sync.last_drive_ms) <= DRIVE_SOFT_START_RESUME_MS)) {
        drive_sync.ramp_active = 1;
        drive_sync.ramp_start_ms = drive_sync.sample_ms;
      }
      if (drive_sync.ramp_active &&
          (uint32_t)(drive_sync.sample_ms - drive_sync.ramp_start_ms) < DRIVE_SOFT_START_MS) {
        uint32_t ramp_elapsed = drive_sync.sample_ms - drive_sync.ramp_start_ms;
        DriveSync_WritePwm(direction, DriveSync_RampPwm(pwm1, ramp_elapsed),
                           DriveSync_RampPwm(pwm2, ramp_elapsed));
      } else {
        drive_sync.ramp_active = 0;
        DriveSync_WritePwm(direction, pwm1, pwm2);
      }
#else
      DriveSync_WritePwm(direction, pwm1, pwm2);
#endif
    }
    /* 같은 목표의 DBG heartbeat는 PID 적분과 CNT 기준을 초기화하지 않는다. */
  }
  if (primask == 0U) __enable_irq();
}

void DriveSync_Tick1ms(void)
{
  uint32_t start_cycle, commit_cycle, commit_elapsed, gap, generation, primask, elapsed_ms;
  DriveSyncPidOutput output;
  if (drive_sync.ready == 0 || drive_sync.active == 0) return;
  start_cycle = DWT->CYCCNT;
  generation = drive_sync.generation;
  gap = start_cycle - drive_sync.sample_cycle;
  /* 우선순위가 더 높은 TIM3 정지/UART IRQ가 선점할 수 있다. 계산 뒤 generation을 다시 검사한다.
   * main이 설정된 제한시간을 넘겨 돌아오지 않으면 이전 명령을 계속 유지하지 않는다. */
  if ((uint32_t)(start_cycle - drive_sync.main_cycle) >
      drive_sync.cycles_per_ms * DRIVE_SYNC_PID_MAIN_TIMEOUT_MS) {
    primask = __get_PRIMASK();
    __disable_irq();
    if (generation == drive_sync.generation && drive_sync.active != 0)
      DriveSync_Fault(DRIVE_SYNC_LATE);
    if (primask == 0U) __enable_irq();
    return;
  }
  if (gap < drive_sync.cycles_per_ms * drive_sync.config.period_ms) return;
  if (gap > drive_sync.peak_gap_cycles) drive_sync.peak_gap_cycles = gap;
  elapsed_ms = gap / drive_sync.cycles_per_ms;
  drive_sync.sample_cycle = start_cycle;
  drive_sync.sample_ms += elapsed_ms;
  output = DriveSyncPid_Step(&drive_sync.state, &drive_sync.config,
      __HAL_TIM_GET_COUNTER(&htim3), __HAL_TIM_GET_COUNTER(&htim4),
      drive_sync.sample_ms, drive_sync.direction, drive_sync.base1, drive_sync.base2);
  primask = __get_PRIMASK();
  __disable_irq();
  commit_cycle = DWT->CYCCNT;
  commit_elapsed = commit_cycle - start_cycle;
  /* format/UART 없이 정수 표본 하나만 저장한다. 정지 선점 시 계산값을 적용값으로 보고하지 않는다. */
  control_snapshot = (DriveControlSnapshot){
    .sequence = control_snapshot.sequence + 1U, .tick = drive_sync.sample_ms,
    .elapsed_ms = output.elapsed_ms, .direction = drive_sync.direction,
    .flags = (drive_sync.config.kp_q8 || drive_sync.config.ki_q8 || drive_sync.config.kd_q8) ? 1U : 0U,
    .delta1 = output.delta1, .delta2 = output.delta2,
    .normalized_right = output.normalized_right, .normalized_left = output.normalized_left,
    .error = output.error, .integral_q8 = drive_sync.state.integral_q8, .correction = output.correction
  };
  if (generation == drive_sync.generation && drive_sync.active != 0) {
    /* 높은 우선순위 ISR이 계산 중 오래 선점했을 수 있으므로 실제 CCR 기록 직전에도 시간을 확인한다.
     * 이전 표본부터 기록까지의 전체 지연과 main 생존 시각 모두 유효할 때만 새 PWM을 적용한다. */
    if ((uint64_t)gap + commit_elapsed >
            (uint64_t)drive_sync.cycles_per_ms * drive_sync.config.max_gap_ms ||
        (uint32_t)(commit_cycle - drive_sync.main_cycle) >
            drive_sync.cycles_per_ms * DRIVE_SYNC_PID_MAIN_TIMEOUT_MS ||
        output.fault != 0)
      DriveSync_Fault(output.fault != 0 ? output.fault : DRIVE_SYNC_LATE);
    else if (output.valid != 0) {
      int pwm1 = output.pwm1, pwm2 = output.pwm2;
#if DRIVE_SOFT_START_MS > 0U
      /* 방향·엔코더 범위 같은 안전 검사는 위 Step이 그대로 수행했다. 가속 중에는 보정 없이 곡선 값만 쓴다. */
      if (drive_sync.ramp_active) {
        uint32_t ramp_elapsed = drive_sync.sample_ms - drive_sync.ramp_start_ms;
        if (ramp_elapsed >= DRIVE_SOFT_START_MS) {
          drive_sync.ramp_active = 0;
        } else {
          pwm1 = DriveSync_RampPwm(drive_sync.base1, ramp_elapsed);
          pwm2 = DriveSync_RampPwm(drive_sync.base2, ramp_elapsed);
          DriveSync_HoldPidDuringRamp();
          control_snapshot.correction = 0;
          control_snapshot.integral_q8 = 0;
        }
      }
#endif
      DriveSync_WritePwm(drive_sync.direction, pwm1, pwm2);
      control_snapshot.flags |= 6U;
    }
  }
  if (primask == 0U) __enable_irq();
  gap = DWT->CYCCNT - start_cycle;
  if (gap > drive_sync.peak_cycles) drive_sync.peak_cycles = gap;
}
#else
void DriveSync_Tick1ms(void) {}
#endif

void Motor_Run(int direction, int pwm_duty)
{
  uint32_t guard=__get_PRIMASK();
  __disable_irq();
  if (Ultrasonic_IsHeld()) {
    if (direction==0 || pwm_duty<=0) Motor_Stop();
    if (!guard) __enable_irq();
    return;
  }
  /* 먼저 양방향 출력을 0으로 하고 방향에 맞는 두 채널만 켠다. 두 모터의 배선 극성이 반대다.
   * 오른쪽 ENC1에는 방향별 trim을 적용하고, 보정이 활성화된 방향만 PID로 보정한다. */
    uint32_t right_trim_bp = direction < 0 ? RIGHT_MOTOR_REVERSE_TRIM_BP : RIGHT_MOTOR_FORWARD_TRIM_BP;
    int encoder_1_pwm_duty =
      (int)(((uint32_t)pwm_duty * right_trim_bp + 5000U) / 10000U);

#if DRIVE_SYNC_PID_ENABLED
  DriveSync_Request(direction, encoder_1_pwm_duty, pwm_duty);
  if (!guard) __enable_irq();
  return;
#endif
  Motor_Stop();

  if (direction > 0)
  {
    /* Physical right ENC1 has opposite polarity and the configured base PWM trim. */
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, encoder_1_pwm_duty);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, pwm_duty);
  }
  else if (direction < 0)
  {
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, encoder_1_pwm_duty);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, pwm_duty);
  }
  if (!guard) __enable_irq();
}

void Motor_StartNextEncoderSection(int direction, int pwm_duty)
{
  uint32_t guard=__get_PRIMASK();
  __disable_irq();
  /* An echo may interrupt the FSM after state assignment but before section start.
   * Defer that already-selected section instead of losing its drive request. */
  if (Ultrasonic_IsHeld()) {
    if (direction!=MOTOR_DIRECTION_STOPPED && Mission_GetState()!=WAIT_CAPTURE_ACK &&
        Mission_GetState()!=MISSION_FAULT) {
      obstacle_resume_direction=direction; obstacle_resume_duty=pwm_duty;
      obstacle_resume_section=1;
    }
    if (!guard) __enable_irq();
    return;
  }
  /* 촬영 ACK 대기에서는 요청이 와도 주행을 시작하지 않는다. 정상 시작은 이전 TIM3 IRQ/플래그를 정리한다.
   * 전진은 CNT=0에서 증가, 후진은 CNT=ARR에서 감소해야 한 구간 경계가 동일하게 발생한다.
   * 너무 빨리 정지하면 encoder 배선 방향과 TIM3 CNT 진행 방향, ARR 값을 먼저 확인한다. */
  if ((direction == MOTOR_DIRECTION_STOPPED) ||
      (Mission_GetState() == WAIT_CAPTURE_ACK))
  {
    Motor_Stop();
    if (!guard) __enable_irq();
    return;
  }

  /* Forward must count up on TIM3; reverse then counts down. */
#if DRIVE_SYNC_PID_ENABLED
  /* CNT 구간 리셋을 속도 펄스로 오인하지 않도록 이전 표본을 중단하고 Motor_Run에서 다시 기준을 잡는다. */
  Motor_Stop();
#endif
  __HAL_TIM_DISABLE_IT(&htim3, TIM_IT_UPDATE);

  {
    /* 같은 방향의 이전 구간이 경계에서 끝났다면 정지 뒤 굴러간 거리를 이번 구간에 넣는다.
     * 방향이 바뀌거나 경계 표시가 없으면(임무 첫 구간, 방향 전환 뒤) 기존처럼 0/ARR에서 시작한다. */
    uint32_t arr = __HAL_TIM_GET_AUTORELOAD(&htim3);
    uint32_t start = direction > 0 ? 0U : arr;
    section_preroll_pending = 0;
    if (section_carry_direction == direction) {
      start = Drive_CarrySectionCount(start, __HAL_TIM_GET_COUNTER(&htim3), direction, arr);
    } else if (section_carry_direction == -direction) {
      /* 방향 전환 단계의 첫 구간: 이전 방향으로 경계 뒤 굴러간 coast만큼 되돌아와야 원래 경계다.
       * coast 뒤 첫 경계 이벤트가 나도록 CNT를 두고, 그 이벤트는 구간으로 세지 않는다(preroll).
       * 그 뒤 평소처럼 ARR+1 count 구간을 센다. coast=0이면 기존과 같다. */
      uint32_t old_start = section_carry_direction > 0 ? 0U : arr;
      uint32_t old_count = Drive_CarrySectionCount(old_start, __HAL_TIM_GET_COUNTER(&htim3),
                                                   section_carry_direction, arr);
      uint32_t coast = section_carry_direction > 0 ? old_count : arr - old_count;
      if (coast > 0U) {
        start = direction > 0 ? arr - (coast - 1U) : coast - 1U;
        section_preroll_pending = 1;
      }
    }
    section_carry_direction = MOTOR_DIRECTION_STOPPED;
    __HAL_TIM_SET_COUNTER(&htim3, start);
  }

  __HAL_TIM_CLEAR_FLAG(&htim3, TIM_FLAG_UPDATE);
  HAL_NVIC_ClearPendingIRQ(TIM3_IRQn);

  Motor_Run(direction, pwm_duty);

  __HAL_TIM_ENABLE_IT(&htim3, TIM_IT_UPDATE);
  if (!guard) __enable_irq();
}

void Drive_Init(void)
{
  control_snapshot = (DriveControlSnapshot){0};
  motor_direction = MOTOR_DIRECTION_STOPPED;
#if DRIVE_SYNC_PID_ENABLED
  memset(&drive_sync, 0, sizeof(drive_sync));
#endif
  /* TIM2 drives both MDD3A channels at 20 kHz: M1A/M1B, M2A/M2B. */
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, 0);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, 0);
  if ((HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4) != HAL_OK))
  {
    Error_Handler();
  }
  if (HAL_TIM_Encoder_Start(&htim3, TIM_CHANNEL_ALL) != HAL_OK)
  {
    Error_Handler();
  }
  /* TIM4는 PID 활성화 시 속도 비교에 쓰며 비활성화 시 진단용이다. 구간 IRQ 판정은 TIM3만 담당한다. */
  if (HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL) != HAL_OK)
  {
    Error_Handler();
  }

#if DRIVE_SYNC_PID_ENABLED
  /* DWT는 코어 clock을 센다. 현재 설정은 HSE8MHz x9=72MHz이며 SystemCoreClock에서 실제 배율을 받는다.
   * 구간/거리 판정은 여전히 TIM3만 사용한다. TIM4는 PID 활성화 시 속도 동기화 피드백에도 사용한다. */
  drive_sync.config = DriveSyncPid_DefaultConfig();
  drive_sync.cycles_per_ms = SystemCoreClock / 1000U;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  drive_sync.main_cycle = DWT->CYCCNT;
  drive_sync.ready = 1;
#endif

  Motor_Stop();
}

void Drive_MarkMainAlive(void)
{
#if DRIVE_SYNC_PID_ENABLED
  drive_sync.main_cycle = DWT->CYCCNT;
#endif
}

int Drive_GetFault(void)
{
#if DRIVE_SYNC_PID_ENABLED
  return drive_sync.fault;
#else
  return 0;
#endif
}

void Drive_GetControlSnapshot(DriveControlSnapshot *snapshot)
{
  uint32_t primask = __get_PRIMASK();
  __disable_irq();
  *snapshot = control_snapshot;
  if (primask == 0U) __enable_irq();
}

int Drive_GetDirection(void) { return motor_direction; }
void Drive_SetDirection(int direction) { motor_direction = direction; }

void Drive_ObstaclePause(void)
{
  int direction=Drive_GetDirection();
  int duty=(int)__HAL_TIM_GET_COMPARE(&htim2,direction>0 ? TIM_CHANNEL_3 : TIM_CHANNEL_4);
#if DRIVE_SYNC_PID_ENABLED
  if (drive_sync.active) { direction=drive_sync.direction; duty=drive_sync.base2; }
#endif
  Motor_Stop();
  obstacle_resume_direction=direction;
  obstacle_resume_duty=duty;
}
void Drive_ObstacleResume(void)
{
  int duty=obstacle_resume_duty, section=obstacle_resume_section;
  obstacle_resume_duty=obstacle_resume_section=0;
  if (duty>0 && Mission_GetState()!=MISSION_FAULT) {
    if (section) Motor_StartNextEncoderSection(obstacle_resume_direction,duty);
    else Motor_Run(obstacle_resume_direction,duty);
  }
}
