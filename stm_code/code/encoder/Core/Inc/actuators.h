#ifndef ACTUATORS_H
#define ACTUATORS_H

#include <stdint.h>

typedef enum
{
  ACTUATOR_COMMAND_NONE,
  ACTUATOR_COMMAND_ON,
  ACTUATOR_COMMAND_OFF,
  ACTUATOR_COMMAND_CLEANER_PWM_12_5,
  ACTUATOR_COMMAND_CLEANER_PWM_25
} ActuatorCommand;

typedef enum {
  ACTUATOR_FRONT_CLEANER,
  ACTUATOR_FRONT_PUMP,
  ACTUATOR_SIDE_CLEANER,
  ACTUATOR_SIDE_PUMP,
  ACTUATOR_COUNT
} ActuatorId;

void Actuators_Init(void);
void Actuators_Update(void);
void Actuators_QueueCommand(ActuatorId id, ActuatorCommand command);
/* 임무 전이 임계구역에서 대기 명령만 지운다. OFF 출력 동작은 StopAll이 맡는다. */
void Actuators_ClearPending(void);
void Actuators_Refresh(ActuatorId id, uint32_t tick);
uint32_t Actuators_GetOutputMask(void);
void Front_Cleaner_Stop(void);
void Front_Cleaner_Run(uint16_t pwm_duty);
void Front_Pump_Stop(void);
void Front_Pump_Start(void);
void Side_Cleaner_Stop(void);
void Side_Cleaner_Run(uint16_t pwm_duty);
void Side_Pump_Stop(void);
void Side_Pump_Start(void);
void CleaningActuators_StopAll(void);

#endif
