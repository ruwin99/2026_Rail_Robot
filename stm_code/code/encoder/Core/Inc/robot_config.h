#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

/* 동작 상수: PWM, 구간 수, 임무/시험 시간 제한. 핀 배선은 robot_board.h를 본다. */
/* PWM 상수는 타이머 CCR에 쓰는 비교값이다. 이름의 12_5/25는 프로토콜 단계명으로 실제 duty 비율과 다르다.
 * DBG1의 100%는 PWM_DUTY(1600)를 기준으로 하며, 실제 오른쪽 encoder 1에는
 * 방향별 RIGHT_MOTOR_*_TRIM_BP로 보정한 기준 PWM을 적용한다. 왼쪽 encoder 2 기준은 100%다.
 * 엔코더 피드백은 구간 판정/속도 동기화/진단에 사용한다. 여기의 속도 명령은 PWM 기준값이며 RPM 목표가 아니다. */
#define PWM_DUTY 1600 /* drive PWM 44.4% of the TIM2 3600-count period. */
#define MISSION_FORWARD_PWM_DUTY 1200 /* DBG1 75% of PWM_DUTY; TIM2 duty 33.3%. */
#define MISSION_REVERSE_PWM_DUTY 1200 /* DBG1 75% of PWM_DUTY; TIM2 duty 33.3%. */
#define LIVE_PWM_DUTY MISSION_FORWARD_PWM_DUTY /* realtime uses the same forward PWM. */
#define FRONT_DC_MOTOR_PWM_DUTY_12_5 1000 /* pwm 27.8% TIM1 channel (3600기준) */
#define SIDE_DC_MOTOR_PWM_DUTY_12_5 1000 /* pwm 27.8% TIM1 channel */
#define DC_MOTOR_PWM_DUTY_25 1200 /* pwm 33.3% TIM1 channel */
#define WATER_PUMP_PWM_DUTY 3600 /* pwm 100% TIM1 channel (ARR=3599, CCR=ARR+1). */
#define CLEANER_COMMAND_TIMEOUT_MS 1000
#define PUMP_COMMAND_TIMEOUT_MS 1000
#define ENCODER_SECTION_LIMIT 6
/* 정지 뒤 관성으로 볼 수 있는 TIM3 이동량 상한(count, 구간 601의 약 1/4).
 * 이보다 큰 변화(손으로 밀기, 잡음)는 관성으로 보지 않고 구간 위치에 넣지 않는다. */
#define ENCODER_COAST_CARRY_MAX_COUNTS 150
#define MISSION_TRANSITION_PAUSE_MS 1000U
#define MOTOR_DIRECTION_FORWARD 1
#define MOTOR_DIRECTION_REVERSE -1
#define MOTOR_DIRECTION_STOPPED 0
/* 오른쪽 ENC1/TIM3의 전진/후진 보정값을 각각 변경한다. 왼쪽 ENC2는 100%다.
 * 10000=100%, 9950=99.50%, 9921=99.21%. 변경 후 재빌드/플래시가 필요하다.
 * #ifndef는 호스트 시험의 방향별 컴파일 설정 변형을 허용한다. */
#ifndef RIGHT_MOTOR_FORWARD_TRIM_BP
#define RIGHT_MOTOR_FORWARD_TRIM_BP 10000U
#endif
#ifndef RIGHT_MOTOR_REVERSE_TRIM_BP
#define RIGHT_MOTOR_REVERSE_TRIM_BP 10000U
#endif
/* 이전 소스의 이름은 전진값의 별칭이다. 사용자 수정 지점은 위 두 literal이다. */
#define RIGHT_MOTOR_TRIM_BP RIGHT_MOTOR_FORWARD_TRIM_BP
#if RIGHT_MOTOR_FORWARD_TRIM_BP < 1 || RIGHT_MOTOR_FORWARD_TRIM_BP > 20000
#error "RIGHT_MOTOR_FORWARD_TRIM_BP must be 1..20000 (10000=100 percent)"
#endif
#if RIGHT_MOTOR_REVERSE_TRIM_BP < 1 || RIGHT_MOTOR_REVERSE_TRIM_BP > 20000
#error "RIGHT_MOTOR_REVERSE_TRIM_BP must be 1..20000 (10000=100 percent)"
#endif
/* TIM2 ARR=3599. 위 범위와 base<=3599이면 uint32 곱셈도 넘치지 않는다. */
#if PWM_DUTY < 0 || PWM_DUTY > 3599 || LIVE_PWM_DUTY < 0 || LIVE_PWM_DUTY > 3599
#error "Drive base PWM must fit TIM2 ARR=3599"
#endif
#if ((PWM_DUTY * RIGHT_MOTOR_FORWARD_TRIM_BP + 5000UL) / 10000UL) > 3599 || \
    ((LIVE_PWM_DUTY * RIGHT_MOTOR_FORWARD_TRIM_BP + 5000UL) / 10000UL) > 3599
#error "RIGHT_MOTOR_FORWARD_TRIM_BP exceeds TIM2 PWM range"
#endif
#if ((PWM_DUTY * RIGHT_MOTOR_REVERSE_TRIM_BP + 5000UL) / 10000UL) > 3599 || \
    ((LIVE_PWM_DUTY * RIGHT_MOTOR_REVERSE_TRIM_BP + 5000UL) / 10000UL) > 3599
#error "RIGHT_MOTOR_REVERSE_TRIM_BP exceeds TIM2 PWM range"
#endif
#define DEBUG_COMMAND_TIMEOUT_MS 600U
#define DEBUG_STATE_PERIOD_MS 100U
#define DEBUG_REVERSE_STOP_MS 100U

#endif
