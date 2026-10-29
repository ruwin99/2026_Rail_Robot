#include "ultrasonic.h"
/* 앞/옆 브러시와 펌프의 상태·명령함·시간 제한을 이 파일에서 소유한다.
 * 명령은 ISR에서도 접수하지만 Update는 main 전용이다. 고장/구간 종료의 StopAll은 ISR에서도 즉시 실행한다. */
#include "actuators.h"
#include "robot_board.h"
#include "robot_config.h"
#include "drive_sync_config.h"
#include "drive_control.h"
#include "mission.h"

#include <string.h>

typedef struct
{
  volatile ActuatorCommand pending_command;
  volatile int active;
  volatile int hw_ready;
  volatile int last_on_tick;
} ActuatorContext;

static ActuatorContext actuators[ACTUATOR_COUNT];

static void Cleaner_StopHardware(ActuatorContext *actuator,
                                 uint32_t tim_channel,
                                 GPIO_TypeDef *gpio_port,
                                 uint16_t in1_pin,
                                 uint16_t in2_pin)
{
  if (actuator->hw_ready != 0)
  {
    __HAL_TIM_SET_COMPARE(&htim1, tim_channel, 0);
    HAL_GPIO_WritePin(gpio_port, in1_pin | in2_pin, GPIO_PIN_RESET);
  }
  actuator->active = 0;
}

static void Cleaner_RunHardware(ActuatorContext *actuator,
                                uint32_t tim_channel,
                                GPIO_TypeDef *gpio_port,
                                uint16_t in1_pin,
                                uint16_t in2_pin,
                                uint16_t pwm_duty)
{
  uint32_t primask = __get_PRIMASK();
  __disable_irq();
  /* main이 ON 허용을 확인한 직후 PID ISR fault가 나도, 늦은 레지스터 쓰기로 다시 켜지지 않게 한다. */
  if (Ultrasonic_IsHeld() || Drive_GetFault() != 0 || Mission_GetState() == MISSION_FAULT) {
    if (primask == 0U) __enable_irq();
    return;
  }
  if (actuator->hw_ready == 0)
  {
    if (primask == 0U) __enable_irq();
    return;
  }

  if (actuator->active == 0)
  {
    /* Select direction only while this role's PWM output is at zero. */
    __HAL_TIM_SET_COMPARE(&htim1, tim_channel, 0);
    HAL_GPIO_WritePin(gpio_port, in1_pin | in2_pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(gpio_port, in1_pin, GPIO_PIN_SET);
  }
  __HAL_TIM_SET_COMPARE(&htim1, tim_channel, pwm_duty);
  actuator->active = 1;
  if (primask == 0U) __enable_irq();
}

static void Pump_StopHardware(ActuatorContext *actuator,
                              uint32_t tim_channel,
                              GPIO_TypeDef *gpio_port,
                              uint16_t in1_pin,
                              uint16_t in2_pin)
{
  if (actuator->hw_ready != 0)
  {
    __HAL_TIM_SET_COMPARE(&htim1, tim_channel, 0);
    HAL_GPIO_WritePin(gpio_port, in1_pin | in2_pin, GPIO_PIN_RESET);
  }
  actuator->active = 0;
}

static void Pump_StartHardware(ActuatorContext *actuator,
                               uint32_t tim_channel,
                               GPIO_TypeDef *gpio_port,
                               uint16_t in1_pin,
                               uint16_t in2_pin)
{
  uint32_t primask = __get_PRIMASK();
  __disable_irq();
  if (Ultrasonic_IsHeld() || Drive_GetFault() != 0 || Mission_GetState() == MISSION_FAULT) {
    if (primask == 0U) __enable_irq();
    return;
  }
  if (actuator->hw_ready == 0)
  {
    if (primask == 0U) __enable_irq();
    return;
  }

  if (actuator->active == 0)
  {
    /* Select direction only while this role's PWM output is at zero. */
    __HAL_TIM_SET_COMPARE(&htim1, tim_channel, 0);
    HAL_GPIO_WritePin(gpio_port, in1_pin | in2_pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(gpio_port, in1_pin, GPIO_PIN_SET);
  }
  __HAL_TIM_SET_COMPARE(&htim1, tim_channel, WATER_PUMP_PWM_DUTY);
  actuator->active = 1;
  if (primask == 0U) __enable_irq();
}

void Front_Cleaner_Stop(void)
{
  Cleaner_StopHardware(&actuators[ACTUATOR_FRONT_CLEANER], FRONT_CLEANER_TIM_CHANNEL,
                       FRONT_CLEANER_GPIO_PORT, FRONT_CLEANER_IN1_PIN,
                       FRONT_CLEANER_IN2_PIN);
}

void Front_Cleaner_Run(uint16_t pwm_duty)
{
  Cleaner_RunHardware(&actuators[ACTUATOR_FRONT_CLEANER], FRONT_CLEANER_TIM_CHANNEL,
                      FRONT_CLEANER_GPIO_PORT, FRONT_CLEANER_IN1_PIN,
                      FRONT_CLEANER_IN2_PIN, pwm_duty);
}

void Front_Pump_Stop(void)
{
  Pump_StopHardware(&actuators[ACTUATOR_FRONT_PUMP], FRONT_WATER_PUMP_TIM_CHANNEL,
                    FRONT_WATER_PUMP_GPIO_PORT, FRONT_WATER_PUMP_IN1_PIN,
                    FRONT_WATER_PUMP_IN2_PIN);
}

void Front_Pump_Start(void)
{
  Pump_StartHardware(&actuators[ACTUATOR_FRONT_PUMP], FRONT_WATER_PUMP_TIM_CHANNEL,
                     FRONT_WATER_PUMP_GPIO_PORT, FRONT_WATER_PUMP_IN1_PIN,
                     FRONT_WATER_PUMP_IN2_PIN);
}

void Side_Cleaner_Stop(void)
{
  Cleaner_StopHardware(&actuators[ACTUATOR_SIDE_CLEANER], SIDE_CLEANER_TIM_CHANNEL,
                       SIDE_CLEANER_GPIO_PORT, SIDE_CLEANER_IN1_PIN,
                       SIDE_CLEANER_IN2_PIN);
}

void Side_Cleaner_Run(uint16_t pwm_duty)
{
  Cleaner_RunHardware(&actuators[ACTUATOR_SIDE_CLEANER], SIDE_CLEANER_TIM_CHANNEL,
                      SIDE_CLEANER_GPIO_PORT, SIDE_CLEANER_IN1_PIN,
                      SIDE_CLEANER_IN2_PIN, pwm_duty);
}

void Side_Pump_Stop(void)
{
  Pump_StopHardware(&actuators[ACTUATOR_SIDE_PUMP], SIDE_WATER_PUMP_TIM_CHANNEL,
                    SIDE_WATER_PUMP_GPIO_PORT, SIDE_WATER_PUMP_IN1_PIN,
                    SIDE_WATER_PUMP_IN2_PIN);
}

void Side_Pump_Start(void)
{
  Pump_StartHardware(&actuators[ACTUATOR_SIDE_PUMP], SIDE_WATER_PUMP_TIM_CHANNEL,
                     SIDE_WATER_PUMP_GPIO_PORT, SIDE_WATER_PUMP_IN1_PIN,
                     SIDE_WATER_PUMP_IN2_PIN);
}

void CleaningActuators_StopAll(void)
{
  Front_Cleaner_Stop();
  Front_Pump_Stop();
  Side_Cleaner_Stop();
  Side_Pump_Stop();
}

static int Actuator_ControlIsAllowed(void)
{
  /* 실시간 청소/시험 모드에서만 ON을 허용하며, 구간 끝 ISR의 event가 남아 있으면 추가 ON을 막는다.
   * UART로 ON이 들어와도 움직이지 않으면 state와 event를 먼저 본다. OFF는 별도로 항상 소비한다. */
  return (!Ultrasonic_IsHeld() && ((Mission_GetState() == REALTIME_CLEANING) ||
          (Mission_GetState() == ACTUATOR_TEST) ||
           (Mission_GetState() == DEBUG_TEST)) &&
          (Mission_IsTransitionPending() == 0) &&
          (Mission_HasEncoderEvent() == 0));
}

static void Actuator_QueueCommand(ActuatorContext *actuator,
                                  ActuatorCommand command)
{
  /* ISR과 main이 공유하는 장치별 한 칸 명령함이다. OFF가 들어오면 main이 꺼낼 때까지 뒤의 ON보다 우선한다.
   * 인터럽트를 잠깐 막아 검사+저장을 한 동작으로 처리하고 원래 IRQ 상태를 복원한다. */
  uint32_t primask = __get_PRIMASK();

  if (Ultrasonic_IsHeld() && command!=ACTUATOR_COMMAND_OFF) return;
  /* Keep the OFF latch atomic against UART ISR updates. */
  __disable_irq();
  /* OFF stays latched until main consumes it; later ON cannot overwrite it. */
  if ((command == ACTUATOR_COMMAND_OFF) ||
      (actuator->pending_command != ACTUATOR_COMMAND_OFF))
  {
    actuator->pending_command = command;
  }
  if (primask == 0U)
  {
    __enable_irq();
  }
}

static ActuatorCommand Actuator_TakePendingCommand(
    ActuatorContext *actuator)
{
  uint32_t primask;
  ActuatorCommand command;

  if (actuator->pending_command == ACTUATOR_COMMAND_NONE)
  {
    return ACTUATOR_COMMAND_NONE;
  }

  primask = __get_PRIMASK();
  /* Read-and-clear atomically against UART ISR command updates. */
  __disable_irq();
  command = actuator->pending_command;
  actuator->pending_command = ACTUATOR_COMMAND_NONE;
  if (primask == 0U)
  {
    __enable_irq();
  }

  return command;
}

void Actuators_Init(void)
{
  /* PWM의 시작값은 0이며 초기화 자체는 ON 명령이 아니다. */
  memset(actuators, 0, sizeof(actuators));
  /* TIM1 drives two cleaner motors and two water pumps at 1 kHz. */
  __HAL_TIM_SET_COMPARE(&htim1, FRONT_CLEANER_TIM_CHANNEL, 0);
  __HAL_TIM_SET_COMPARE(&htim1, FRONT_WATER_PUMP_TIM_CHANNEL, 0);
  __HAL_TIM_SET_COMPARE(&htim1, SIDE_CLEANER_TIM_CHANNEL, 0);
  __HAL_TIM_SET_COMPARE(&htim1, SIDE_WATER_PUMP_TIM_CHANNEL, 0);

  if ((HAL_TIM_PWM_Start(&htim1, FRONT_CLEANER_TIM_CHANNEL) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim1, FRONT_WATER_PUMP_TIM_CHANNEL) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim1, SIDE_CLEANER_TIM_CHANNEL) != HAL_OK) ||
      (HAL_TIM_PWM_Start(&htim1, SIDE_WATER_PUMP_TIM_CHANNEL) != HAL_OK))
  {
    Error_Handler();
  }
  actuators[ACTUATOR_FRONT_CLEANER].hw_ready = 1;
  actuators[ACTUATOR_FRONT_PUMP].hw_ready = 1;
  actuators[ACTUATOR_SIDE_CLEANER].hw_ready = 1;
  actuators[ACTUATOR_SIDE_PUMP].hw_ready = 1;
  CleaningActuators_StopAll();
}

void Actuators_QueueCommand(ActuatorId id, ActuatorCommand command)
{
  if ((unsigned)id < ACTUATOR_COUNT)
    Actuator_QueueCommand(&actuators[id], command);
}

void Actuators_ClearPending(void)
{
  for (unsigned i = 0; i < ACTUATOR_COUNT; ++i)
    actuators[i].pending_command = ACTUATOR_COMMAND_NONE;
}

void Actuators_Refresh(ActuatorId id, uint32_t tick)
{
  if ((unsigned)id < ACTUATOR_COUNT)
    actuators[id].last_on_tick = (int)tick;
}

uint32_t Actuators_GetOutputMask(void)
{
  return (actuators[ACTUATOR_FRONT_CLEANER].active ? 1U : 0U) |
         (actuators[ACTUATOR_SIDE_CLEANER].active ? 2U : 0U) |
         (actuators[ACTUATOR_FRONT_PUMP].active ? 4U : 0U) |
         (actuators[ACTUATOR_SIDE_PUMP].active ? 8U : 0U);
}

void Actuators_Update(void)
{
  ActuatorCommand front_cleaner_command;
  ActuatorCommand front_pump_command;
  ActuatorCommand side_cleaner_command;
  ActuatorCommand side_pump_command;

  front_cleaner_command = ACTUATOR_COMMAND_NONE;
  /* 네 장치마다 명령함을 한 번 꺼내 OFF를 먼저 적용하고 ON은 Actuator_ControlIsAllowed를 통과시킨다.
   * 유효 ON을 받을 때마다 last_on_tick을 갱신하므로 이미 켜진 장치도 반복 명령으로 수명을 연장한다.
   * 아래 앞/옆 브러시·펌프는 같은 규칙이며 브러시는 단계 PWM, 펌프는 고정 PWM을 사용한다. */
  if (actuators[ACTUATOR_FRONT_CLEANER].pending_command != ACTUATOR_COMMAND_NONE)
  {
    front_cleaner_command =
        Actuator_TakePendingCommand(&actuators[ACTUATOR_FRONT_CLEANER]);
  }
  if (front_cleaner_command == ACTUATOR_COMMAND_OFF)
  {
    Front_Cleaner_Stop();
  }
  else if ((front_cleaner_command ==
            ACTUATOR_COMMAND_CLEANER_PWM_12_5) ||
           (front_cleaner_command ==
            ACTUATOR_COMMAND_CLEANER_PWM_25))
  {
    if (Actuator_ControlIsAllowed() != 0)
    {
      actuators[ACTUATOR_FRONT_CLEANER].last_on_tick = (int)Ultrasonic_OperationalTick();
      if (front_cleaner_command == ACTUATOR_COMMAND_CLEANER_PWM_25)
      {
        Front_Cleaner_Run(DC_MOTOR_PWM_DUTY_25);
      }
      else
      {
        Front_Cleaner_Run(FRONT_DC_MOTOR_PWM_DUTY_12_5);
      }
    }
    else
    {
      Front_Cleaner_Stop();
    }
  }

  front_pump_command = ACTUATOR_COMMAND_NONE;
  if (actuators[ACTUATOR_FRONT_PUMP].pending_command != ACTUATOR_COMMAND_NONE)
  {
    front_pump_command = Actuator_TakePendingCommand(&actuators[ACTUATOR_FRONT_PUMP]);
  }
  if (front_pump_command == ACTUATOR_COMMAND_OFF)
  {
    Front_Pump_Stop();
  }
  else if (front_pump_command == ACTUATOR_COMMAND_ON)
  {
    if (Actuator_ControlIsAllowed() != 0)
    {
      actuators[ACTUATOR_FRONT_PUMP].last_on_tick = (int)Ultrasonic_OperationalTick();
      if (actuators[ACTUATOR_FRONT_PUMP].active == 0)
      {
        Front_Pump_Start();
      }
    }
    else
    {
      Front_Pump_Stop();
    }
  }

  side_cleaner_command = ACTUATOR_COMMAND_NONE;
  if (actuators[ACTUATOR_SIDE_CLEANER].pending_command != ACTUATOR_COMMAND_NONE)
  {
    side_cleaner_command =
        Actuator_TakePendingCommand(&actuators[ACTUATOR_SIDE_CLEANER]);
  }
  if (side_cleaner_command == ACTUATOR_COMMAND_OFF)
  {
    Side_Cleaner_Stop();
  }
  else if ((side_cleaner_command ==
            ACTUATOR_COMMAND_CLEANER_PWM_12_5) ||
           (side_cleaner_command ==
            ACTUATOR_COMMAND_CLEANER_PWM_25))
  {
    if (Actuator_ControlIsAllowed() != 0)
    {
      actuators[ACTUATOR_SIDE_CLEANER].last_on_tick = (int)Ultrasonic_OperationalTick();
      if (side_cleaner_command == ACTUATOR_COMMAND_CLEANER_PWM_25)
      {
        Side_Cleaner_Run(DC_MOTOR_PWM_DUTY_25);
      }
      else
      {
        Side_Cleaner_Run(SIDE_DC_MOTOR_PWM_DUTY_12_5);
      }
    }
    else
    {
      Side_Cleaner_Stop();
    }
  }

  side_pump_command = ACTUATOR_COMMAND_NONE;
  if (actuators[ACTUATOR_SIDE_PUMP].pending_command != ACTUATOR_COMMAND_NONE)
  {
    side_pump_command = Actuator_TakePendingCommand(&actuators[ACTUATOR_SIDE_PUMP]);
  }
  if (side_pump_command == ACTUATOR_COMMAND_OFF)
  {
    Side_Pump_Stop();
  }
  else if (side_pump_command == ACTUATOR_COMMAND_ON)
  {
    if (Actuator_ControlIsAllowed() != 0)
    {
      actuators[ACTUATOR_SIDE_PUMP].last_on_tick = (int)Ultrasonic_OperationalTick();
      if (actuators[ACTUATOR_SIDE_PUMP].active == 0)
      {
        Side_Pump_Start();
      }
    }
    else
    {
      Side_Pump_Stop();
    }
  }

  if ((actuators[ACTUATOR_FRONT_CLEANER].active != 0) &&
      /* 각 부속장치는 모드가 허용되지 않거나 마지막 ON 이후 1000ms가 지나면 독립적으로 끈다.
       * 주행 모터의 일반 임무 통신 watchdog은 이 블록에 포함되지 않는다. 실제 차단 시점은 main 실행에 의존한다. */
      ((Actuator_ControlIsAllowed() == 0) ||
       ((uint32_t)(Ultrasonic_OperationalTick() -
                   (uint32_t)actuators[ACTUATOR_FRONT_CLEANER].last_on_tick) >=
        CLEANER_COMMAND_TIMEOUT_MS)))
  {
    Front_Cleaner_Stop();
  }

  if ((actuators[ACTUATOR_FRONT_PUMP].active != 0) &&
      ((Actuator_ControlIsAllowed() == 0) ||
       ((uint32_t)(Ultrasonic_OperationalTick() -
                   (uint32_t)actuators[ACTUATOR_FRONT_PUMP].last_on_tick) >=
        PUMP_COMMAND_TIMEOUT_MS)))
  {
    Front_Pump_Stop();
  }

  if ((actuators[ACTUATOR_SIDE_CLEANER].active != 0) &&
      ((Actuator_ControlIsAllowed() == 0) ||
       ((uint32_t)(Ultrasonic_OperationalTick() -
                   (uint32_t)actuators[ACTUATOR_SIDE_CLEANER].last_on_tick) >=
        CLEANER_COMMAND_TIMEOUT_MS)))
  {
    Side_Cleaner_Stop();
  }

  if ((actuators[ACTUATOR_SIDE_PUMP].active != 0) &&
      ((Actuator_ControlIsAllowed() == 0) ||
       ((uint32_t)(Ultrasonic_OperationalTick() -
                   (uint32_t)actuators[ACTUATOR_SIDE_PUMP].last_on_tick) >=
        PUMP_COMMAND_TIMEOUT_MS)))
  {
    Side_Pump_Stop();
  }
}
