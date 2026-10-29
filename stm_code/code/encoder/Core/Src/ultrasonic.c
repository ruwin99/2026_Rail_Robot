#include <stddef.h>
#include "ultrasonic.h"
#include "robot_board.h"
#include "drive_control.h"
#include "actuators.h"
#include "mission.h"
#include "uart_protocol.h"

/* 정지 판정 거리는 명목 10 cm다(2026-10-05 사용자 지시로 25 cm에서 축소, 온도·센서 오차 여유 없음).
 * 343 m/s 기준 10 cm 왕복 583.1 us를 정수 us로 올림했다. */
#define ULTRASONIC_NEAR_ECHO_US 584U
#define ULTRASONIC_WINDOW_SIZE 20U
#define ULTRASONIC_REQUIRED_COUNT 16U
/* 근거리 결과가 연속 3회(명목 약 180 ms)면 해제 창을 기다리지 않고 바로 정지한다. */
#define ULTRASONIC_HOLD_STREAK 3U
#define ULTRASONIC_SAMPLE_UNKNOWN 0U
#define ULTRASONIC_SAMPLE_NEAR 1U
#define ULTRASONIC_SAMPLE_FAR 2U

static volatile uint8_t armed, held, obstacle, detected, phase, saw_rise;
static uint8_t reported;
/* 해제·시작 판정은 20개 창의 원거리 16개가 기준이다. 운용 정책상 NO_ECHO는
 * 장애물 근거가 아니므로 원거리와 같게 센다. 정지는 근거리 연속 횟수로 판단한다. */
static uint8_t samples[ULTRASONIC_WINDOW_SIZE], sample_index;
static volatile uint8_t sample_count, near_count, far_count, near_streak;
static volatile uint16_t interval, age;
static volatile uint32_t rise_cycle, paused_at, paused_total;
static uint32_t saved_cnt3, saved_cnt4, saved_update;
static volatile uint32_t distance_mm;
static volatile UltrasonicRangeStatus range_status;

static void inhibit(void)
{
  uint32_t mask=__get_PRIMASK();
  __disable_irq();
  if (!held) {
    held=1; paused_at=HAL_GetTick();
    saved_cnt3=__HAL_TIM_GET_COUNTER(&htim3);
    saved_cnt4=__HAL_TIM_GET_COUNTER(&htim4);
    saved_update=htim3.Instance->DIER & TIM_IT_UPDATE;
    __HAL_TIM_DISABLE_IT(&htim3,TIM_IT_UPDATE);
    Drive_ObstaclePause();
    CleaningActuators_StopAll();
    Actuators_ClearPending();
    UART_ObstacleBarrier();
  }
  if (!mask) __enable_irq();
}

static void range_result(uint8_t sample)
{
  uint8_t previous;
  if (sample_count==ULTRASONIC_WINDOW_SIZE) {
    previous=samples[sample_index];
    if (previous==ULTRASONIC_SAMPLE_NEAR) --near_count;
    else if (previous==ULTRASONIC_SAMPLE_FAR) --far_count;
  } else ++sample_count;
  samples[sample_index]=sample;
  if (++sample_index>=ULTRASONIC_WINDOW_SIZE) sample_index=0;
  if (sample==ULTRASONIC_SAMPLE_NEAR) ++near_count;
  else if (sample==ULTRASONIC_SAMPLE_FAR) ++far_count;
  near_streak = sample==ULTRASONIC_SAMPLE_NEAR ?
    (uint8_t)(near_streak<ULTRASONIC_HOLD_STREAK ? near_streak+1U : near_streak) : 0U;
  if (near_streak>=ULTRASONIC_HOLD_STREAK && !detected) {
    /* 창을 근거리로 채워, 해제에는 이후 원거리 16/20이라는 기존 기준을 그대로 요구한다. */
    for (uint8_t i=0; i<ULTRASONIC_WINDOW_SIZE; ++i) samples[i]=ULTRASONIC_SAMPLE_NEAR;
    sample_count=near_count=ULTRASONIC_WINDOW_SIZE; far_count=0;
    detected=1;
  } else if (sample_count==ULTRASONIC_WINDOW_SIZE &&
      near_count>=ULTRASONIC_REQUIRED_COUNT) detected=1;
  else if (sample_count==ULTRASONIC_WINDOW_SIZE &&
           far_count>=ULTRASONIC_REQUIRED_COUNT) detected=0;
  if (armed && detected && !obstacle) {
    inhibit();
    obstacle=1;
  }
}

void Ultrasonic_Init(void)
{
  uint8_t i;
  armed=held=obstacle=detected=phase=saw_rise=reported=0;
  sample_index=sample_count=near_count=far_count=near_streak=0;
  for (i=0; i<ULTRASONIC_WINDOW_SIZE; ++i) samples[i]=ULTRASONIC_SAMPLE_UNKNOWN;
  interval=age=0; paused_total=0;
  distance_mm=0; range_status=ULTRASONIC_WAITING;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; /* Never reset a live PID clock. */
  HAL_GPIO_WritePin(GPIOB,GPIO_PIN_8,GPIO_PIN_RESET);
}

void Ultrasonic_ResetWindow(void)
{
  uint32_t mask=__get_PRIMASK();
  uint8_t i;
  __disable_irq();
  /* A live interlock owns its range history. Session activation normally
   * occurs in IDLE; never erase evidence while the robot is moving/held. */
  if (!armed && !held) {
    for (i=0; i<ULTRASONIC_WINDOW_SIZE; ++i) samples[i]=ULTRASONIC_SAMPLE_UNKNOWN;
    sample_index=sample_count=near_count=far_count=near_streak=0;
    detected=obstacle=reported=0;
    phase=saw_rise=0; interval=age=0;
    distance_mm=0; range_status=ULTRASONIC_WAITING;
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_8,GPIO_PIN_RESET);
  }
  if (!mask) __enable_irq();
}

void Ultrasonic_Arm(void)
{
  uint32_t mask=__get_PRIMASK();
  __disable_irq();
  if (!armed) {
    armed=1;
    /* An unknown/near range must prevent the first motor command. A complete
     * recent far window may start immediately; pre-start readings never
     * caused a hold or obstacle event while the robot was idle. */
    if (sample_count!=ULTRASONIC_WINDOW_SIZE ||
        far_count<ULTRASONIC_REQUIRED_COUNT) {
      inhibit();
      if (detected) obstacle=1;
    }
  }
  if (!mask) __enable_irq();
}

void Ultrasonic_Disarm(void)
{
  uint32_t mask=__get_PRIMASK();
  __disable_irq();
  armed=held=obstacle=reported=0;
  paused_total=0;
  if (!mask) __enable_irq();
}

void Ultrasonic_Tick50us(void)
{
  uint32_t mask=__get_PRIMASK();
  __disable_irq(); /* Serialize timeout vs a higher-priority echo edge. */
  if (phase==1U) {
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_8,GPIO_PIN_RESET);
    phase=2U; age=0; saw_rise=0;
  } else if (phase==2U && ++age>=600U) {
    distance_mm=0; range_status=ULTRASONIC_NO_ECHO;
    /* 상태 표시는 NO_ECHO로 남기되, 운용 정책상 주행을 막는 장애물 근거로 쓰지 않는다. */
    phase=0; range_result(ULTRASONIC_SAMPLE_FAR);
  }
  if (++interval>=1200U) {
    interval=0; phase=1; saw_rise=0;
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_8,GPIO_PIN_SET);
  }
  if (!mask) __enable_irq();
}

void Ultrasonic_OnEcho(int high)
{
  uint32_t cycles, scale=SystemCoreClock/1000000U;
  if (phase!=2U || scale==0U) return;
  if (high) {
    if (!saw_rise) { rise_cycle=DWT->CYCCNT; saw_rise=1; }
  } else if (saw_rise) {
    cycles=DWT->CYCCNT-rise_cycle; phase=0; saw_rise=0;
    /* 왕복 시간(us)을 명목 음속 343 m/s 기준 mm로 변환한다. 안전 판정의
     * 원래 cycle 임계값은 변경하지 않고 디버깅 표시용 값만 보관한다. */
    distance_mm=((cycles/scale)*343U+1000U)/2000U;
    range_status=ULTRASONIC_ECHO;
    /* 명목 10 cm 이내 왕복이면 근거리 표본이다. */
    range_result(cycles<=ULTRASONIC_NEAR_ECHO_US*scale ? ULTRASONIC_SAMPLE_NEAR : ULTRASONIC_SAMPLE_FAR);
  }
}

void Ultrasonic_Update(void)
{
  uint32_t mask;
  uint8_t emit_clear=0;
  if (armed && obstacle && !reported) {
    reported=1;
    (void)UART_QueueEvent("OBSTACLE_HOLD", NULL);
  }
  mask=__get_PRIMASK(); __disable_irq();
  if (held && sample_count==ULTRASONIC_WINDOW_SIZE && far_count>=ULTRASONIC_REQUIRED_COUNT) {
    paused_total += HAL_GetTick()-paused_at;
    {
      /* 주행 중 멈췄다면 정지 뒤 굴러간 거리를 진행 방향 기준으로 구간 위치에 넣는다.
       * 정지 중 경계를 넘었다면 경계 직전으로 맞춰 재개 직후 구간이 끝나고, 뒤로 밀렸다면 그만큼 다시 간다.
       * 주행 중이 아니었다면(촬영 대기 등) 기존처럼 정지 순간 값으로 되돌린다. */
      int direction = Drive_GetObstacleResumeDirection();
      __HAL_TIM_SET_COUNTER(&htim3, direction != 0 ?
          Drive_CarrySectionCount(saved_cnt3, __HAL_TIM_GET_COUNTER(&htim3), direction,
                                  __HAL_TIM_GET_AUTORELOAD(&htim3)) : saved_cnt3);
    }
    __HAL_TIM_SET_COUNTER(&htim4,saved_cnt4);
    __HAL_TIM_CLEAR_FLAG(&htim3,TIM_FLAG_UPDATE);
    HAL_NVIC_ClearPendingIRQ(TIM3_IRQn);
    UART_ObstacleBarrier();
    Actuators_ClearPending();
    held=0; obstacle=0;
    if (reported) { reported=0; emit_clear=1; }
    if (Mission_GetState()!=MISSION_FAULT && Mission_GetState()!=MISSION_IDLE &&
        Mission_GetState()!=DEBUG_TEST && Mission_GetState()!=ACTUATOR_TEST) {
      if (saved_update) __HAL_TIM_ENABLE_IT(&htim3,TIM_IT_UPDATE);
      Drive_MarkMainAlive();
      Drive_ObstacleResume();
    }
  }
  if (!mask) __enable_irq();
  if (emit_clear) (void)UART_QueueEvent("OBSTACLE_CLEAR", NULL);
}

int Ultrasonic_IsHeld(void) { return held; }
void Ultrasonic_GetSnapshot(UltrasonicSnapshot *snapshot)
{
  uint32_t mask=__get_PRIMASK();
  __disable_irq();
  snapshot->distance_mm=distance_mm;
  snapshot->status=range_status;
  snapshot->valid=(range_status==ULTRASONIC_ECHO);
  snapshot->detected=detected;
  snapshot->held=held;
  if (!mask) __enable_irq();
}
int Ultrasonic_HasObstacle(void) { return obstacle; }
uint32_t Ultrasonic_OperationalTick(void)
{
  uint32_t mask=__get_PRIMASK(), now;
  __disable_irq();
  now=(held ? paused_at : HAL_GetTick())-paused_total;
  if (!mask) __enable_irq();
  return now;
}
