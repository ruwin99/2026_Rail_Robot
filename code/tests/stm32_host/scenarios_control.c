#include "checks.h"
static void begin(unsigned capability) {
  char line[96]; host_reset();
  snprintf(line,sizeof(line),"DBG1_HELLO " SID " %u\r\n",capability); host_line(line);
}
typedef struct {
  unsigned long command_seq, sample_seq, tick, dt, flags;
  long direction, d1, d2, nr, nl, error, integral, correction;
} Sample;
static Sample latest(void) {
  const char *found=NULL, *next=host_output; Sample sample={0};
  while((next=strstr(next,"DBG1_CTRL "))!=NULL) {
    found=next; const char *end=strstr(next,"\r\n"); assert(end && end-next+2<192); ++next;
  }
  assert(found);
  assert(sscanf(found,"%*s %*s %lu %lu %lu %lu %ld %lu %ld %ld %ld %ld %ld %ld %ld",
    &sample.command_seq,&sample.sample_seq,&sample.tick,&sample.dt,&sample.direction,&sample.flags,
    &sample.d1,&sample.d2,&sample.nr,&sample.nl,&sample.error,&sample.integral,&sample.correction)==13);
  return sample;
}
static void publish(unsigned tick) { host_time(tick); Drive_MarkMainAlive(); Robot_Update(); }
static void step(unsigned tick, unsigned c1, unsigned c2) {
  host_time(tick); Drive_MarkMainAlive(); TIM3->CNT=c1; TIM4->CNT=c2;
  size_t before=strlen(host_output); DriveSync_Tick1ms(); assert(strlen(host_output)==before);
}
void scenario_control(const char *name) {
  if (!strcmp(name,"control_negotiation")) {
    const unsigned accepted[]={17,19,21,23};
    const unsigned rejected[]={0,9,15,16,18,24,25,31,32};
    for(unsigned n=0;n<4;++n) {
      begin(accepted[n]); assert(Mission_GetState()==DEBUG_TEST);
      assert(strstr(host_output,"DBG1_CTRL " SID));
    }
    for(unsigned n=1;n<=8;++n) {
      begin(n); assert(Mission_GetState()==DEBUG_TEST);
      assert(!strstr(host_output,"DBG1_CTRL"));
    }
    for(unsigned n=0;n<sizeof(rejected)/sizeof(rejected[0]);++n) {
      begin(rejected[n]); assert(Mission_GetState()==MISSION_IDLE); all_off();
      assert(!strstr(host_output,"DBG1_READY"));
    }
    begin(17); host_line("DBG1_SET " SID " 1 0 0 1\r\n"); all_off();
    assert(strstr(host_output,"CAPABILITY_OR_RANGE"));
  } else if (!strcmp(name,"control_samples")) {
    begin(17); Sample s=latest(); assert(!(s.flags&6) && s.sample_seq==0);
    host_line("DBG1_SET " SID " 1 1 5000 0\r\n");
    for(unsigned n=1;n<=10;++n) step(10*n,10*n,(601-10*n)%601);
    /* Output is never sent from 10ms ISR; report only the last actual calculation. */
    assert(latest().sample_seq==0); publish(100); s=latest();
    assert(s.sample_seq==10 && s.tick==100 && s.dt==10 && s.direction==1 && s.flags==7);
    assert(s.d1==10 && s.d2==-10 && s.nr==20 && s.nl==10 && s.error==-10);
    assert(s.integral==-12825 && s.correction==-130);
    /* Heartbeat cannot multiply CTRL bandwidth, even if STATE responds immediately. */
    host_time(101); host_line("DBG1_SET " SID " 2 1 5000 0\r\n");
    assert(latest().command_seq==1);
    Motor_Stop(); publish(201); s=latest(); assert(!(s.flags&6) && s.sample_seq==10);
    Motor_Run(-1,800);
    for(unsigned n=1;n<=10;++n) step(201+10*n,100-5*n,(501+16*n)%601);
    publish(301); s=latest();
    assert(s.sample_seq==20 && s.tick==301 && s.flags==7 && s.direction==-1);
    assert(s.d1==-5 && s.d2==16 && s.nr==10 && s.nl==16 && s.error==6);
    assert(s.integral==7680 && s.correction==78);
    Motor_Run(1,800); publish(401); s=latest(); assert(!(s.flags&6));
  } else if (!strcmp(name,"control_preempt")) {
    begin(17); Motor_Run(1,800); host_stop_on_counter=1; step(10,10,591);
    publish(100); Sample s=latest(); assert(!(s.flags&4)); all_off();
  } else if (!strcmp(name,"control_elapsed_wrap_fault")) {
    DriveControlSnapshot sample;
    begin(17); Motor_Run(1,800); step(15,10,591);
    Drive_GetControlSnapshot(&sample);
    assert(sample.elapsed_ms==15 && sample.error==-6 && sample.integral_q8==-1282);
    assert(sample.normalized_right==20 && sample.normalized_left==10 && sample.flags==7);
    Motor_Stop(); TIM3->CNT=598; TIM4->CNT=3; Motor_Run(1,800);
    step(25,2,594); Drive_GetControlSnapshot(&sample);
    assert(sample.delta1==5 && sample.delta2==-10 && sample.error==0 && sample.flags==7);
    step(35,601,584); Drive_GetControlSnapshot(&sample);
    assert(Drive_GetFault()==DRIVE_SYNC_ENCODER_RANGE && !(sample.flags&6)); all_off();
  } else if (!strcmp(name,"control_disabled")) {
    begin(17); Motor_Run(1,800); step(10,10,591); publish(100);
    Sample s=latest(); assert(s.flags==0 && s.sample_seq==0);
  } else assert(!"control scenario not implemented");
}
