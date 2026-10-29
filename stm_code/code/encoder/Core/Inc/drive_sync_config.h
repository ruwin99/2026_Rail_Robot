#ifndef DRIVE_SYNC_CONFIG_H
#define DRIVE_SYNC_CONFIG_H

/* 로컬 벤치용 주행 동기화 PID 설정. 기본 활성화하며 빌드 시
 * -DDRIVE_SYNC_PID_ENABLED=0으로 기존 고정 PWM 동작을 선택할 수 있다.
 * 절대 RPM/위치 제어가 아니라 두 바퀴의 진행방향 속도 차이를 보정한다. */
#ifndef DRIVE_SYNC_PID_ENABLED
#define DRIVE_SYNC_PID_ENABLED 1
#endif

/* 후진은 아래 별도 게인/제한으로 제어한다. 0이면 기준 PWM만 적용한다.
 * 전역 PID가 켜져 있으면 이 설정이 0이어도 엔코더/정지/시간 초과 보호는 유지한다. */
#ifndef DRIVE_SYNC_PID_REVERSE_ENABLED
#define DRIVE_SYNC_PID_REVERSE_ENABLED 1
#endif
#if DRIVE_SYNC_PID_REVERSE_ENABLED != 0 && DRIVE_SYNC_PID_REVERSE_ENABLED != 1
#error "DRIVE_SYNC_PID_REVERSE_ENABLED must be 0 or 1"
#endif

/* 전진 게인: Q8의256=1.0, P=8, I=0.5, D=0. 기존 설정 이름을 유지한다. */
#define DRIVE_SYNC_PID_KP_Q8 2048
#define DRIVE_SYNC_PID_KI_Q8 128
#define DRIVE_SYNC_PID_KD_Q8 0
#define DRIVE_SYNC_PID_MAX_CORRECTION 400
#define DRIVE_SYNC_PID_MAX_SLEW 20 /* 표본당 PWM 보정 변화 상한 */
/* 후진 전용: P=8, I=0.5, D=0. 전진과 독립적으로 튜닝한다. */
#define DRIVE_SYNC_PID_REVERSE_KP_Q8 2048
#define DRIVE_SYNC_PID_REVERSE_KI_Q8 128
#define DRIVE_SYNC_PID_REVERSE_KD_Q8 0
#define DRIVE_SYNC_PID_REVERSE_MAX_CORRECTION 400
#define DRIVE_SYNC_PID_REVERSE_MAX_SLEW 20
#define DRIVE_SYNC_PID_PERIOD_MS 10U
#define DRIVE_SYNC_PID_MAX_GAP_MS 20U

/* 사용자 확인: 두 바퀴의 실제 전진 방향은 정상이며, 실측 전진 CNT는
 * TIM3 0->4, TIM4 0->594(601 기준 -7)이다. TIM4만 PID 입력 부호를 반전한다.
 * 타이머는 TIM3 TI1(x2), TIM4 TI12(x4)이므로 raw count만 2:1로 정규화한다.
 * 모터 극성/PWM은 변경하지 않는다. TIM3 ARR=600과 구간 거리 판정은 유지한다.
 * modulus는 각각 ARR+1이다. */
#define DRIVE_SYNC_PID_ENCODER1_MODULUS 601U
#define DRIVE_SYNC_PID_ENCODER2_MODULUS 601U
#define DRIVE_SYNC_PID_ENCODER1_SCALE 2
#define DRIVE_SYNC_PID_ENCODER2_SCALE 1
#define DRIVE_SYNC_PID_ENCODER1_SIGN 1
#define DRIVE_SYNC_PID_ENCODER2_SIGN -1

/* 전진 오른쪽 목표 속도 = 왼쪽 속도 * BP / 10000. raw 엔코더 해상도 보정과 별개다.
 * 목표 비율은97~101%만 허용한다. 후진 목표는 아래 별도 설정이다.
 * 범위를 바꿀 때는 아래 MIN/MAX만 수정한다. 빌드/실행 검증이 함께 참조한다. */
#define DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_MIN_BP 9300U
#define DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_MAX_BP 10500U
#ifndef DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_BP
#define DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_BP 9950U
#endif
#if DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_BP < DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_MIN_BP || \
    DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_BP > DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_MAX_BP
#error "DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_BP must be within configured MIN_BP..MAX_BP"
#endif

/* 후진 왼쪽 목표 속도 = 오른쪽 속도 * BP / 10000. 전진과 독립적으로 조정한다.
 * 10000은 동일 속도, 9900은 왼쪽 목표99%. 빌드/실행 검증은 같은 범위를 참조한다. */
#define DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_MIN_BP 9700U
#define DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_MAX_BP 10100U
#ifndef DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_BP
#define DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_BP 10000U
#endif
#if DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_BP < DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_MIN_BP || \
    DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_BP > DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_MAX_BP
#error "DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_BP must be within configured MIN_BP..MAX_BP"
#endif

/* 사용자 요청: 전진에서만 엔코더 속도 차이에 대한 출력 보정 연결을 반대로 적용한다.
 * raw/정규화 엔코더의 물리 좌우, 방향·range·stall 검사는 변경하지 않는다.
 * -1이면 전진 control error=-(정규화 오른쪽-왼쪽), +1이면 기존 연결이다.
 * 후진 error 및 후진 보정 활성 설정에는 영향을 주지 않는다. */
#ifndef DRIVE_SYNC_PID_FORWARD_FEEDBACK_SIGN
#define DRIVE_SYNC_PID_FORWARD_FEEDBACK_SIGN -1
#endif
#if DRIVE_SYNC_PID_FORWARD_FEEDBACK_SIGN != -1 && DRIVE_SYNC_PID_FORWARD_FEEDBACK_SIGN != 1
#error "DRIVE_SYNC_PID_FORWARD_FEEDBACK_SIGN must be -1 or 1"
#endif
/* 후진 sign=-1: 정규화 R<L이면 PWM1을 낮추고 PWM2를 높이도록 보정 대응을 반전한다.
 * 전진 설정과 독립적이며 PWM1=오른쪽/PWM2=왼쪽의 실제 대응은 실기로 확인한다.
 * 방향 정규화 뒤 제어 오차에만 적용하며 raw/normalized telemetry와 안전 검사는 유지한다. */
#ifndef DRIVE_SYNC_PID_REVERSE_FEEDBACK_SIGN
#define DRIVE_SYNC_PID_REVERSE_FEEDBACK_SIGN -1
#endif
#if DRIVE_SYNC_PID_REVERSE_FEEDBACK_SIGN != -1 && DRIVE_SYNC_PID_REVERSE_FEEDBACK_SIGN != 1
#error "DRIVE_SYNC_PID_REVERSE_FEEDBACK_SIGN must be -1 or 1"
#endif
#define DRIVE_SYNC_PID_STALL_TIMEOUT_MS 500U
#define DRIVE_SYNC_PID_MAIN_TIMEOUT_MS 250U

/* 출발 가속(soft start): 정지 상태에서 출발하면 두 바퀴 PWM을 시작값(아래 FLOOR_PERCENT)에서 기준 PWM까지 이 시간 동안
 * smoothstep(S자) 곡선으로 올린다. 0이면 기존처럼 첫 기록부터 기준 PWM이다.
 * - 정지·폴트·장애물 정지는 가속 없이 즉시 0이다(Motor_Stop 그대로).
 * - 가속 중에는 PID 보정을 0으로 두고 정지(stall) 판정도 가속이 끝난 뒤부터 센다.
 *   따라서 바퀴가 막혀 있을 때 폴트까지 최대 이 값 + DRIVE_SYNC_PID_STALL_TIMEOUT_MS가 걸린다.
 * - 같은 방향으로 달리다 RESUME_MS 안에 다시 출발하면(구간 전환 등) 가속 없이 바로 기준 PWM을 쓴다.
 * PID를 끈 빌드(DRIVE_SYNC_PID_ENABLED=0)에는 적용하지 않는다. */
#ifndef DRIVE_SOFT_START_MS
#define DRIVE_SOFT_START_MS 500U
#endif
#define DRIVE_SOFT_START_RESUME_MS 50U
#if DRIVE_SOFT_START_MS > 3000U
#error "DRIVE_SOFT_START_MS must be 0..3000"
#endif
/* 가속 시작값: 기준 PWM의 이 비율(%)에서 출발한다. 0에서 시작하면 모터가 실제로 돌기 전의 약한 구간이 길어
 * 경사에서 뒤로 밀려 역방향 폴트(10 ms에 2 count 이상 반대)가 날 수 있다. 바퀴가 막 움직이기 시작하는
 * 값보다 조금 높게 둔다. 30은 실측 전 임시값이며 레일 위에서 확인 후 조정한다. */
#ifndef DRIVE_SOFT_START_FLOOR_PERCENT
#define DRIVE_SOFT_START_FLOOR_PERCENT 30U
#endif
#if DRIVE_SOFT_START_FLOOR_PERCENT > 90U
#error "DRIVE_SOFT_START_FLOOR_PERCENT must be 0..90"
#endif

#endif
