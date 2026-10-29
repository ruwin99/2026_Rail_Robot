#include "checks.h"
static void reset(void) {
  host_reset(); Mission_SetState(DEBUG_TEST); Drive_SetDirection(1); Motor_Run(1,800); forward(right_base_pwm(800),800);
}
/* 실제 전진: TIM3는 증가, TIM4는 감소한다. modulus는 각각 601이다. */
static void sample(void) { host_time(10); TIM3->CNT=10; TIM4->CNT=591; }
static void reverse_reset(void) {
  host_reset(); Mission_SetState(DEBUG_TEST); Drive_SetDirection(-1); Motor_Run(-1,800);
  reverse(right_reverse_base_pwm(800),800);
}
static void fault(int expected) {
  all_off(); assert(Drive_GetFault()==expected && Mission_GetState()==MISSION_FAULT);
  assert(Drive_GetDirection()==0 && !(TIM3->DIER&TIM_IT_UPDATE) && host_primask==0);
  uint32_t nonzero=host_nonzero_writes;
  Motor_Run(1,800); Front_Cleaner_Run(1200); Front_Pump_Start(); Side_Cleaner_Run(2000); Side_Pump_Start();
  all_off(); host_line("START\r\nFRONT_PUMP_ON\r\n"); all_off();
  assert(host_nonzero_writes==nonzero);
}
void scenario_runtime(const char *name) {
  if (!strcmp(name,"reverse_half_percent_target")) {
    reverse_reset(); DriveControlSnapshot s;
    for(unsigned n=1;n<=100;++n) {
      host_time(n*10); Drive_MarkMainAlive();
      TIM3->CNT=(601-(10*n)%601)%601; TIM4->CNT=(20*n)%601;
      DriveSync_Tick1ms();
    }
    Drive_GetControlSnapshot(&s);
    /* R=L=20: 0.5% 목표 차이는 0.1count이므로 정수 error=0이어도 적분에 보존된다. */
    assert(s.flags==7 && s.normalized_right==20 && s.normalized_left==20);
    assert(s.error==0 && s.integral_q8==1280 && s.correction==5 && Drive_GetFault()==0);
    reverse(right_reverse_base_pwm(800)-5,805);
    Motor_Run(1,800); Drive_GetControlSnapshot(&s);
    assert(s.integral_q8==0 && s.correction==0 && !(s.flags&6));
  } else if (!strcmp(name,"directional_gains") || !strcmp(name,"directional_disabled")) {
    int disabled=!strcmp(name,"directional_disabled");
    host_reset(); Mission_SetState(DEBUG_TEST); Motor_Run(1,1600);
    DriveControlSnapshot s;
    host_time(10); Drive_MarkMainAlive(); TIM3->CNT=10; TIM4->CNT=591;
    DriveSync_Tick1ms(); Drive_GetControlSnapshot(&s);
    assert(s.error==-10 && s.integral_q8==-1280 && s.correction==-20);
    forward(right_base_pwm(1600)+20,1580);
    for(unsigned n=2;n<=100;++n) {
      host_time(n*10); Drive_MarkMainAlive(); TIM3->CNT=(10*n)%601; TIM4->CNT=(601-(10*n)%601)%601;
      DriveSync_Tick1ms();
    }
    Drive_GetControlSnapshot(&s); assert(s.correction==-400);
    Motor_Run(-1,1600); Drive_GetControlSnapshot(&s);
    assert(s.integral_q8==0 && s.correction==0 && !(s.flags&6));
    uint32_t c1=TIM3->CNT,c2=TIM4->CNT;
    for(unsigned n=1;n<=100;++n) {
      host_time(1000+n*10); Drive_MarkMainAlive();
      TIM3->CNT=(c1+601-(5*n)%601)%601; TIM4->CNT=(c2+20*n)%601;
      DriveSync_Tick1ms(); Drive_GetControlSnapshot(&s);
      if(n==1) {
        assert(s.error==10 && s.integral_q8==(disabled?0:1280));
        assert(s.correction==(disabled?0:20));
      }
      assert(s.correction>=-400 && s.correction<=400 && Drive_GetFault()==0);
    }
    assert(s.correction==(disabled?0:400));
    reverse(right_reverse_base_pwm(1600)-(disabled?0:400),disabled?1600:2000);
    Motor_Run(1,1600); Drive_GetControlSnapshot(&s);
    assert(s.integral_q8==0 && s.correction==0);
    host_time(2010); Drive_MarkMainAlive(); TIM3->CNT=(TIM3->CNT+10)%601; TIM4->CNT=(TIM4->CNT+591)%601;
    DriveSync_Tick1ms(); Drive_GetControlSnapshot(&s);
    assert(s.integral_q8==-1280 && s.correction==-20 && s.flags==7);
  } else if (!strcmp(name,"reverse_target_bias")) {
    reverse_reset(); DriveControlSnapshot sample;
    for(unsigned n=1;n<=100;++n) {
      host_time(n*10); Drive_MarkMainAlive(); TIM3->CNT=(601-(10*n)%601)%601; TIM4->CNT=(20*n)%601;
      DriveSync_Tick1ms();
    }
    Drive_GetControlSnapshot(&sample);
    assert(Drive_GetFault()==0 && sample.flags==7 && sample.normalized_right==20 && sample.normalized_left==20);
    assert(sample.error==0 && sample.integral_q8==2560 && sample.correction==11);
    reverse(right_reverse_base_pwm(800)-11,811);
    Motor_Run(1,800); Drive_GetControlSnapshot(&sample);
    assert(sample.integral_q8==0 && sample.correction==0 && !(sample.flags&6));
    host_time(1010); Drive_MarkMainAlive(); TIM3->CNT=(TIM3->CNT+10)%601; TIM4->CNT=(TIM4->CNT+581)%601;
    DriveSync_Tick1ms(); Drive_GetControlSnapshot(&sample);
    /* 현재 전진9800의 첫 표본: -0.4count, I=-51/256, correction=-3. */
    assert(sample.integral_q8==-51 && sample.correction==-3 && sample.flags==7);
    forward(right_base_pwm(800)+3,797);
  } else if (!strcmp(name,"reverse_target_start")) {
    reverse_reset(); host_time(10); TIM3->CNT=599; TIM4->CNT=5; DriveSync_Tick1ms();
    DriveControlSnapshot sample; Drive_GetControlSnapshot(&sample);
    assert(Drive_GetFault()==0 && Mission_GetState()==DEBUG_TEST && sample.flags==7);
    assert(sample.normalized_right==4 && sample.normalized_left==5);
  } else if (!strcmp(name,"target_start")) {
    host_reset(); host_line("DBG1_HELLO " SID " 17\r\n");
    host_line("DBG1_SET " SID " 1 1 10000 0\r\n"); forward(right_base_pwm(1600),1600);
    host_time(10); TIM3->CNT=2; TIM4->CNT=596; DriveSync_Tick1ms();
    DriveControlSnapshot sample; Drive_GetControlSnapshot(&sample);
    assert(Drive_GetFault()==0 && Mission_GetState()==DEBUG_TEST && sample.flags==7);
    assert(sample.normalized_right==4 && sample.normalized_left==5);
    printf("TARGET %u ACTIVE PWM %lu %lu\n",(unsigned)DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_BP,
      (unsigned long)TIM2->CCR2,(unsigned long)TIM2->CCR3);
  } else if (!strcmp(name,"trim_probe")) {
    const int bases[]={1600,800,63};
    for (unsigned i=0;i<sizeof(bases)/sizeof(bases[0]);++i) {
      host_reset(); Motor_Run(1,bases[i]);
      printf("TRIM %d %lu %lu\n",bases[i],(unsigned long)TIM2->CCR2,(unsigned long)TIM2->CCR3);
      Motor_Stop(); Motor_Run(-1,bases[i]);
      printf("REVERSE %d %lu %lu\n",bases[i],(unsigned long)TIM2->CCR1,(unsigned long)TIM2->CCR4);
    }
    host_reset(); host_line("DBG1_HELLO " SID " 8\r\n");
    printf("%s",host_output); all_off();
  } else if (!strcmp(name,"trim_reverse_reset")) {
    host_reset(); Mission_SetState(DEBUG_TEST); Motor_Run(1,800);
    for (unsigned n=1;n<=10;++n) {
      host_time(10*n); Drive_MarkMainAlive(); TIM3->CNT=6*n; TIM4->CNT=(601-8*n)%601;
      DriveSync_Tick1ms();
    }
#if DRIVE_SYNC_PID_ENABLED
    assert(TIM2->CCR2>(uint32_t)right_base_pwm(800));
#endif
    Motor_Run(-1,800); reverse(right_reverse_base_pwm(800),800);
    host_time(110); Drive_MarkMainAlive(); TIM3->CNT-=5; TIM4->CNT+=10;
    DriveSync_Tick1ms(); reverse(right_reverse_base_pwm(800),800); assert(Drive_GetFault()==0);
  } else if (!strcmp(name,"direction")) {
    host_reset(); Motor_Run(1,1600); forward(right_base_pwm(1600),1600);
    Motor_Stop(); drive_off(); Motor_Run(-1,800); reverse(right_reverse_base_pwm(800),800); Motor_Stop(); drive_off();
  } else if (!strcmp(name,"normal")) {
    reset(); sample(); DriveSync_Tick1ms();
    assert(Drive_GetFault()==0 && TIM2->CCR2>right_base_pwm(800) && TIM2->CCR3<800 && TIM2->CCR1==0 && TIM2->CCR4==0);
  } else if (!strcmp(name,"equal")) {
    reset(); sample(); TIM3->CNT=5; DriveSync_Tick1ms(); forward(right_base_pwm(800),800); assert(Drive_GetFault()==0);
  } else if (!strcmp(name,"reverse")) {
    reset(); Motor_Stop(); Motor_Run(-1,800); host_time(10); TIM3->CNT=596; TIM4->CNT=10;
    DriveSync_Tick1ms(); reverse(right_reverse_base_pwm(800),800); assert(Drive_GetFault()==0);
  } else if (!strcmp(name,"reverse_fixed_restore")) {
    /* 먼저 전진 적분을 쌓아도 후진 진입 시 보정이 남지 않아야 한다. */
    reset(); sample(); DriveSync_Tick1ms();
    assert(TIM2->CCR2>(uint32_t)right_base_pwm(800));
    Motor_Run(-1,800);
    for (unsigned n=1;n<=60;++n) {
      host_time(10+10*n); Drive_MarkMainAlive();
      TIM3->CNT=(10+601-5*n)%601; TIM4->CNT=(591+16*n)%601;
      DriveSync_Tick1ms(); reverse(right_reverse_base_pwm(800),800);
      assert(Drive_GetFault()==0);
      Motor_Run(-1,800); reverse(right_reverse_base_pwm(800),800);
    }
    /* 후진 중 오차 누적 없이 전진 기본 게인을 복원한다. */
    Motor_Run(1,800); forward(right_base_pwm(800),800);
    host_time(620); Drive_MarkMainAlive();
    TIM3->CNT=(TIM3->CNT+5)%601; TIM4->CNT=(TIM4->CNT+601-10)%601;
    DriveSync_Tick1ms(); forward(right_base_pwm(800),800);
    host_time(630); TIM3->CNT=(TIM3->CNT+10)%601; TIM4->CNT=(TIM4->CNT+601-10)%601;
    DriveSync_Tick1ms(); assert(Drive_GetFault()==0);
    assert(TIM2->CCR2>(uint32_t)right_base_pwm(800) && TIM2->CCR3<800);
  } else if (!strcmp(name,"reverse_pi_enabled")) {
    reverse_reset(); host_time(10); TIM3->CNT=596; TIM4->CNT=16;
    DriveSync_Tick1ms(); assert(Drive_GetFault()==0);
    reverse(right_reverse_base_pwm(800)-20,820);
    /* 후진 sign -1: 오른쪽 norm이 크면 CCR1을 높이고 CCR4를 낮춘다. */
    reverse_reset(); host_time(10); TIM3->CNT=591; TIM4->CNT=10;
    DriveSync_Tick1ms(); reverse(right_reverse_base_pwm(800)+20,780);
    const int bases[]={800,1600};
    for(unsigned index=0;index<2;++index) {
      int base=bases[index], right=right_reverse_base_pwm(base), cap=400, previous=0;
      if(cap>right/4)cap=right/4;
      if(cap>base/4)cap=base/4;
      if(cap>3599-right)cap=3599-right;
      if(cap>3599-base)cap=3599-base;
      host_reset(); Mission_SetState(DEBUG_TEST); Motor_Run(-1,base);
      for(unsigned n=1;n<=200;++n) {
        host_time(n*10); Drive_MarkMainAlive(); TIM3->CNT=(601-(5*n)%601)%601; TIM4->CNT=(16*n)%601;
        DriveSync_Tick1ms(); DriveControlSnapshot sample; Drive_GetControlSnapshot(&sample);
        assert(Drive_GetFault()==0 && sample.flags==7 && sample.error==6);
        assert(sample.correction>=-cap && sample.correction<=cap);
        assert(sample.correction-previous>=-20 && sample.correction-previous<=20);
        reverse(right-sample.correction,base+sample.correction);
        if(n==10)assert(sample.integral_q8==7680 && sample.correction==78);
        previous=sample.correction;
      }
      assert(previous==cap);
      /* 후진 적분/포화 보정을 전진으로 가져가지 않는다. */
      Motor_Run(1,base); forward(right_base_pwm(base),base);
      DriveControlSnapshot sample; Drive_GetControlSnapshot(&sample);
      assert(sample.integral_q8==0 && sample.correction==0 && !(sample.flags&6));
      host_time(2010); Drive_MarkMainAlive(); TIM3->CNT=(TIM3->CNT+5)%601; TIM4->CNT=(TIM4->CNT+591)%601;
      DriveSync_Tick1ms(); forward(right_base_pwm(base),base);
      Drive_GetControlSnapshot(&sample);
      assert(sample.integral_q8==-2 && sample.correction==0 && sample.flags==7);
    }
  } else if (!strcmp(name,"forward_target_bias")) {
    reset(); DriveControlSnapshot sample;
    for(unsigned n=1;n<=100;++n) {
      host_time(n*10); Drive_MarkMainAlive(); TIM3->CNT=(10*n)%601; TIM4->CNT=(601-(20*n)%601)%601;
      DriveSync_Tick1ms();
    }
    Drive_GetControlSnapshot(&sample);
    assert(sample.normalized_right==20 && sample.normalized_left==20 && sample.error==0);
    assert(sample.integral_q8==-512 && sample.correction==-2 && sample.flags==7);
    forward(right_base_pwm(800)+2,798);
    Motor_Run(-1,800); Drive_GetControlSnapshot(&sample);
    assert(sample.integral_q8==0 && sample.correction==0);
    host_time(1010); Drive_MarkMainAlive(); TIM3->CNT=(TIM3->CNT+596)%601; TIM4->CNT=(TIM4->CNT+10)%601;
    DriveSync_Tick1ms(); Drive_GetControlSnapshot(&sample);
    assert(sample.integral_q8==0 && sample.correction==0 && sample.error==0);
    reverse(right_reverse_base_pwm(800),800);
  } else if (!strcmp(name,"reverse_stall")) {
    reverse_reset();
    for (unsigned n=1;n<=50;++n) {
      host_time(10*n); Drive_MarkMainAlive(); TIM3->CNT=(601-5*n)%601; DriveSync_Tick1ms();
    }
    fault(DRIVE_SYNC_STALL);
  } else if (!strcmp(name,"reverse_wrong_direction")) {
    reverse_reset(); host_time(10); TIM3->CNT=596; TIM4->CNT=599;
    DriveSync_Tick1ms(); fault(DRIVE_SYNC_WRONG_DIRECTION);
  } else if (!strcmp(name,"reverse_range")) {
    reverse_reset(); host_time(10); TIM3->CNT=601; TIM4->CNT=10;
    DriveSync_Tick1ms(); fault(DRIVE_SYNC_ENCODER_RANGE);
  } else if (!strcmp(name,"reverse_late")) {
    reverse_reset(); host_time(21); DriveSync_Tick1ms(); fault(DRIVE_SYNC_LATE);
  } else if (!strcmp(name,"reverse_main_timeout")) {
    reverse_reset(); host_time(251); DriveSync_Tick1ms(); fault(DRIVE_SYNC_LATE);
  } else if (!strcmp(name,"reverse_stop")) {
    reverse_reset(); host_time(10); TIM3->CNT=596; TIM4->CNT=16;
    uint32_t nonzero=host_nonzero_writes; host_stop_on_counter=1; DriveSync_Tick1ms();
    drive_off(); assert(Drive_GetFault()==0 && host_primask==0);
    host_time(20); DriveSync_Tick1ms(); drive_off(); assert(host_nonzero_writes==nonzero);
  } else if (!strcmp(name,"missing")) {
    reset(); Front_Pump_Start();
    for (unsigned n=1;n<=50;++n) { host_time(n*10); Drive_MarkMainAlive(); TIM3->CNT=5*n; DriveSync_Tick1ms(); }
    fault(DRIVE_SYNC_STALL);
  } else if (!strcmp(name,"restart")) {
    reset(); sample(); DriveSync_Tick1ms(); Motor_Stop(); TIM3->CNT=0; TIM4->CNT=200;
    host_time(1000); Drive_MarkMainAlive(); Motor_Run(1,800); TIM3->CNT=5; TIM4->CNT=190;
    host_time(1010); DriveSync_Tick1ms(); forward(right_base_pwm(800),800); assert(Drive_GetFault()==0);
  } else if (!strcmp(name,"stop")) {
    reset(); sample(); uint32_t nonzero=host_nonzero_writes;
    host_stop_on_counter=1; DriveSync_Tick1ms(); drive_off();
    assert(Drive_GetFault()==0 && host_primask==0); host_time(20); DriveSync_Tick1ms(); drive_off();
    assert(host_nonzero_writes==nonzero);
  } else if (!strcmp(name,"delay")) {
    reset(); sample(); host_commit_delay=11U*72000U; DriveSync_Tick1ms(); fault(DRIVE_SYNC_LATE);
  } else if (!strcmp(name,"boundary")) {
    reset(); sample(); host_commit_delay=10U*72000U; DriveSync_Tick1ms(); assert(Drive_GetFault()==0 && TIM2->CCR2>0);
    reset(); sample(); host_commit_delay=10U*72000U+1; DriveSync_Tick1ms(); fault(DRIVE_SYNC_LATE);
  } else if (!strcmp(name,"freshness")) {
    reset(); host_time(0); Drive_MarkMainAlive();
    host_time(239); Motor_Stop(); Motor_Run(1,800); host_time(249); TIM3->CNT=5; TIM4->CNT=591;
    host_commit_delay=2U*72000U; DriveSync_Tick1ms(); fault(DRIVE_SYNC_LATE);
    reset(); host_time(251); DriveSync_Tick1ms(); fault(DRIVE_SYNC_LATE);
  } else if (!strcmp(name,"heartbeat")) {
    reset(); sample(); DriveSync_Tick1ms();
    uint32_t first=TIM2->CCR2, second=TIM2->CCR3, writes=host_compare_writes;
    host_time(11); Motor_Run(1,800); assert(host_compare_writes==writes); forward((int)first,(int)second);
    host_time(20); TIM3->CNT=15; TIM4->CNT=581; DriveSync_Tick1ms();
    assert(Drive_GetFault()==0 && host_compare_writes==writes+4);
  } else if (!strcmp(name,"measured_encoder_sign")) {
    reset(); host_time(10); TIM3->CNT=4; TIM4->CNT=594; DriveSync_Tick1ms();
    assert(Drive_GetFault()==0 && Mission_GetState()==DEBUG_TEST);
    assert(TIM2->CCR1==0 && TIM2->CCR4==0 && TIM2->CCR2>0 && TIM2->CCR3>0);
    /* 후진 시 두 raw count의 wrap 방향도 정확하게 정규화해야 한다. */
    Motor_Stop(); TIM3->CNT=2; TIM4->CNT=598;
    host_time(20); Drive_MarkMainAlive(); Motor_Run(-1,800);
    host_time(30); TIM3->CNT=598; TIM4->CNT=7; DriveSync_Tick1ms();
    reverse(right_reverse_base_pwm(800),800); assert(Drive_GetFault()==0);
    /* TIM4가 실제 전진 중 증가한다면 오류 검출은 여전히 유지된다. */
    reset(); host_time(10); TIM3->CNT=4; TIM4->CNT=2; DriveSync_Tick1ms();
    fault(DRIVE_SYNC_WRONG_DIRECTION);
  } else if (!strcmp(name,"disabled")) {
    reset(); sample(); uint32_t writes=host_compare_writes; DriveSync_Tick1ms();
    forward(right_base_pwm(800),800); assert(host_compare_writes==writes && Drive_GetFault()==0);
  } else assert(!"unknown runtime scenario");
}
