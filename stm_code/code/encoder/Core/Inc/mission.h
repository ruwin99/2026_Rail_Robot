#ifndef MISSION_H
#define MISSION_H

#include "main.h"

typedef enum
{
  MISSION_IDLE,
  INITIAL_SCAN,
  WAIT_CAPTURE_ACK,
  RETURNING,
  REALTIME_CLEANING,
  ACTUATOR_TEST,
  DEBUG_TEST,
  RESCAN_RETURN,
  RESCAN,
  MISSION_DONE,
  MISSION_FAULT,
  /* 재검사 끝에서 출발점으로 돌아오는 마지막 연속 후진. 기존 상태 번호(RR2 응답에 실림)를 바꾸지 않도록 끝에 둔다. */
  FINAL_RETURN
} MissionState;

void Mission_Init(void);
void Mission_Update(void);
void Mission_SetState(MissionState next_state);
void Mission_OnEncoderUpdate(TIM_HandleTypeDef *htim);
MissionState Mission_GetState(void);
int Mission_IsTransitionPending(void);
int Mission_GetSectionCount(void);
int Mission_HasEncoderEvent(void);
/* 아래 두 함수는 고장 경로의 짧은 IRQ 임계구역 안에서 호출한다. 송신하지 않는다. */
void Mission_LatchFault(void);
void Mission_OnUartFault(void);
/* RR2 acceptance runs in main, after CRC/session/sequence validation. */
int Mission_ApplyReliableCommand(int start_command);
/* sections: 다음 START 임무의 단계별 구간 수(1..ENCODER_SECTION_LIMIT). */
int Mission_ApplyReliablePrepare(int prepare, int sections);
int Mission_GetSectionLimit(void);
int Mission_IsPrepared(void);

#endif
