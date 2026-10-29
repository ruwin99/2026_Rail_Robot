#include "checks.h"
#include "debug_console.h"
#include "ultrasonic.h"

static void begin(void) {
  host_reset(); host_line("DBG1_HELLO " SID " 1\r\n");
  assert(Mission_GetState()==DEBUG_TEST);
  assert(strstr(host_output,"DBG1_DIAG " SID " 0 0 READY 0 0 0 "));
}
static void near(void) {
  while (!(GPIOB->ODR & GPIO_PIN_8)) Ultrasonic_Tick50us();
  Ultrasonic_Tick50us(); Ultrasonic_OnEcho(1);
  DWT->CYCCNT+=467U*72U; Ultrasonic_OnEcho(0); Robot_Update();
}
void scenario_debug_diag(const char *name) {
  char command[96];
  if (!strcmp(name,"debug_diag_stall")) {
    begin(); unsigned sequence=0;
    for(unsigned tick=10;tick<=510;tick+=10) {
      host_time(tick); Drive_MarkMainAlive();
      if(tick==10 || tick%200==0) {
        snprintf(command,sizeof(command),"DBG1_SET " SID " %u 1 10000 0\r\n",++sequence);
        host_line(command);
      }
      size_t before=strlen(host_output);
      DriveSync_Tick1ms();
      /* Fault ISR must remain free of blocking UART calls. */
      if(Drive_GetFault()) assert(strlen(host_output)==before);
      Robot_Update();
    }
    assert(Drive_GetFault()==DRIVE_SYNC_STALL && Mission_GetState()==MISSION_FAULT); all_off();
    const char *diag=strstr(host_output,"DBG1_DIAG " SID " 3 510 DRIVE_STALL 4 0 0 0 0 0 0\r\n");
    const char *error=strstr(host_output,"DBG1_ERROR " SID " DRIVE_STALL\r\n");
    assert(diag && error && diag<error);
    size_t before=strlen(host_output); Robot_Update(); assert(before==strlen(host_output));
    host_line("DBG1_CLOSE " SID "\r\n");
    host_line("DBG1_HELLO " SID " 1\r\n");
    assert(strstr(host_output,"BUSY state=10 fault=4\r\n"));
    assert(Mission_GetState()==MISSION_FAULT); all_off();
  } else if (!strcmp(name,"debug_diag_hold")) {
    begin(); host_time(100); host_line("DBG1_SET " SID " 1 1 10000 0\r\n");
    host_time(150); for(unsigned i=0;i<16;++i) near();
    unsigned writes=host_nonzero_writes;
    for(unsigned n=2;n<=10;++n) {
      host_time(n*100);
      snprintf(command,sizeof(command),"DBG1_SET " SID " %u 1 10000 0\r\n",n);
      host_line(command); all_off();
    }
    assert(Mission_GetState()==DEBUG_TEST && host_nonzero_writes==writes);
    assert(strstr(host_output,"DBG1_STATE " SID " 10 1000 0 0 0 0 0 0 0\r\n"));
    assert(strstr(host_output,"DBG1_DIAG " SID " 10 1000 ULTRASONIC_HOLD 0 1 1 0 0 0 0\r\n"));
    host_time(1599); Robot_Update(); assert(Mission_GetState()==DEBUG_TEST);
    host_time(1600); Robot_Update(); assert(Mission_GetState()==MISSION_IDLE); all_off();
    assert(strstr(host_output,"DBG1_TIMEOUT " SID));
    host_line("DBG1_HELLO " SID " 1\r\n");
    assert(Mission_GetState()==DEBUG_TEST && Ultrasonic_IsHeld());
    host_line("DBG1_SET " SID " 1 1 10000 0\r\n"); all_off();
  } else if (!strcmp(name,"debug_diag_pending")) {
    begin(); host_line("DBG1_SET " SID " 1 1 10000 0\r\n");
    host_feed("DBG1_CLOSE " SID "\r\n");
    host_time(251); size_t before=strlen(host_output); DriveSync_Tick1ms();
    assert(Drive_GetFault()==DRIVE_SYNC_LATE && before==strlen(host_output));
    /* A separately called deactivation must not erase the deferred fault report. */
    Debug_Deactivate(); Robot_Update();
    assert(strstr(host_output,"DBG1_DIAG " SID " 1 251 DRIVE_LATE 2 0 0 0 0 0 0\r\n"));
    assert(strstr(host_output,"DBG1_ERROR " SID " DRIVE_LATE\r\n"));
    assert(Mission_GetState()==MISSION_FAULT); all_off();
  } else if (!strcmp(name,"debug_diag_faults")) {
    const int faults[]={DRIVE_SYNC_BAD_CONFIG,DRIVE_SYNC_ENCODER_RANGE,DRIVE_SYNC_WRONG_DIRECTION};
    const char *reasons[]={"DRIVE_BAD_CONFIG","DRIVE_ENCODER_RANGE","DRIVE_WRONG_DIRECTION"};
    for(unsigned n=0;n<3;++n) {
      begin(); host_line("DBG1_SET " SID " 1 1 10000 0\r\n");
      size_t before=strlen(host_output);
      if(faults[n]==DRIVE_SYNC_BAD_CONFIG) Motor_Run(1,4000);
      else {
        host_time(10); TIM3->CNT=faults[n]==DRIVE_SYNC_ENCODER_RANGE ? 601U : 599U;
        DriveSync_Tick1ms();
      }
      assert(before==strlen(host_output) && Drive_GetFault()==faults[n]);
      Robot_Update(); all_off();
      snprintf(command,sizeof(command),"DBG1_ERROR " SID " %s\r\n",reasons[n]);
      assert(strstr(host_output,command));
      snprintf(command,sizeof(command),"%s %d 0 0 %lu 0 0 0\r\n",reasons[n],faults[n],(unsigned long)TIM3->CNT);
      assert(strstr(host_output,command));
      assert(Mission_GetState()==MISSION_FAULT);
    }
  } else assert(!"unknown debug diagnosis scenario");
}
