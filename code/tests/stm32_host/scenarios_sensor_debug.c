#include "checks.h"
#include "ultrasonic.h"
#include "debug_console.h"
#include "rgb_pwm.h"

static void begin(void) {
  host_reset(); Ultrasonic_Init(); RGB_Init();
  assert(!Ultrasonic_IsHeld());
  host_line("DBG1_HELLO " SID " 8\r\n");
  assert(Mission_GetState()==DEBUG_TEST);
  assert(strstr(host_output,"DBG1_READY " SID " 8 "));
  assert(strstr(host_output," 0 0 0 0 1 0 0 0 WAITING\r\n"));
  all_off();
}
static void pulse(unsigned us) {
  while (!(GPIOB->ODR & GPIO_PIN_8)) Ultrasonic_Tick50us();
  Ultrasonic_Tick50us(); Ultrasonic_OnEcho(1);
  DWT->CYCCNT+=us*72U; Ultrasonic_OnEcho(0); Robot_Update();
}
static void no_echo(void) {
  while (!(GPIOB->ODR & GPIO_PIN_8)) Ultrasonic_Tick50us();
  Ultrasonic_Tick50us();
  for(unsigned i=0;i<600;++i) Ultrasonic_Tick50us();
  Robot_Update();
}
static void expected_colour(unsigned red, unsigned green, unsigned blue) {
  unsigned seen_red=0, seen_green=0, seen_blue=0;
  for(unsigned i=0;i<255;++i) {
    RGB_Tick50us();
    unsigned on=RGB_PWM_ACTIVE_LOW ? (GPIOB->BSRR >> 16) : (GPIOB->BSRR & 0xffffU);
    seen_red+=(on & 32U)!=0;
    seen_green+=(on & 16U)!=0;
    seen_blue+=(on & 8U)!=0;
  }
  assert(seen_red==red && seen_green==green && seen_blue==blue);
}
void scenario_sensor_debug(const char *name) {
  uint8_t r,g,b;
  if (!strcmp(name,"sensor_debug_range")) {
    begin(); no_echo();
    host_line("DBG1_SET " SID " 1 0 0 0\r\n");
    assert(strstr(host_output," 0 0 0 0 1 0 0 0 NO_ECHO\r\n"));
    for(unsigned i=0;i<19;++i) no_echo();
    assert(!Ultrasonic_IsHeld());
    for(unsigned i=0;i<20;++i) pulse(2000);
    assert(!Ultrasonic_IsHeld());
    for(unsigned i=0;i<16;++i) pulse(467);
    all_off();
    host_line("DBG1_SET " SID " 2 0 0 0\r\n");
    assert(strstr(host_output," 80 1 1 0 1 0 0 0 ECHO\r\n"));
    for(unsigned i=0;i<16;++i) pulse(2000);
    assert(!Ultrasonic_IsHeld());
    host_line("DBG1_SET " SID " 3 0 0 0\r\n");
    assert(strstr(host_output," 343 1 0 0 1 0 0 0 ECHO\r\n"));
    assert(host_nonzero_writes==0); all_off();
  } else if (!strcmp(name,"sensor_debug_prestart")) {
    UltrasonicSnapshot range;
    begin();
    host_line("DBG1_LED " SID " 1 manual 0 255 0\r\n");
    for(unsigned i=0;i<20;++i) pulse(467);
    assert(!Ultrasonic_IsHeld() && !Ultrasonic_HasObstacle());
    Ultrasonic_GetSnapshot(&range);
    assert(range.valid && range.detected && !range.held && range.distance_mm==80U);
    host_line("DBG1_SET " SID " 2 0 0 0\r\n");
    assert(strstr(host_output," 80 1 1 0 0 0 255 0 ECHO\r\n"));
    assert(strstr(host_output,"OBSTACLE_HOLD\r\n")==NULL);
    assert(Debug_GetLedOverride(&r,&g,&b) && r==0 && g==255 && b==0);
    expected_colour(0,255,0);
    host_line("DBG1_LED " SID " 3 auto 0 0 0\r\n");
    host_time(250); Robot_Update();
    expected_colour(0,0,127);
    assert(!strstr(host_output,"OBSTACLE_HOLD\r\n"));
    Mission_LatchFault(); Robot_Update();
    expected_colour(255,0,0);
  } else if (!strcmp(name,"sensor_debug_led")) {
    begin();
    host_line("DBG1_LED " SID " 1 manual 0 255 0\r\n");
    assert(Debug_GetLedOverride(&r,&g,&b) && r==0 && g==255 && b==0);
    assert(strstr(host_output," 0 0 0 0 0 0 255 0 WAITING\r\n"));
    for(unsigned i=0;i<255;++i) {
      RGB_Tick50us();
      assert(GPIOB->BSRR==(RGB_PWM_ACTIVE_LOW ? (40U | 16U<<16) : (16U | 40U<<16)));
    }
    for(unsigned i=0;i<20;++i) pulse(467);
    expected_colour(0,255,0);
    host_line("DBG1_LED " SID " 2 manual 0 0 255\r\n");
    assert(Debug_GetLedOverride(&r,&g,&b) && b==255); expected_colour(0,0,255);
    for(unsigned i=0;i<16;++i) pulse(2000);
    assert(!Ultrasonic_IsHeld());
    for(unsigned i=0;i<255;++i) {
      RGB_Tick50us();
      assert(GPIOB->BSRR==(RGB_PWM_ACTIVE_LOW ? (48U | 8U<<16) : (8U | 48U<<16)));
    }
    host_line("DBG1_LED " SID " 3 manual 0 0 0\r\n");
    for(unsigned i=0;i<255;++i) {
      RGB_Tick50us();
      assert(GPIOB->BSRR==(RGB_PWM_ACTIVE_LOW ? 56U : 56U<<16));
    }
    host_line("DBG1_LED " SID " 4 auto 0 0 0\r\n");
    assert(!Debug_GetLedOverride(&r,&g,&b));
    host_line("DBG1_CLOSE " SID "\r\n");
    assert(Mission_GetState()==MISSION_IDLE); all_off();
  } else if (!strcmp(name,"sensor_debug_effects")) {
    begin();
    host_line("DBG1_LED " SID " 1 breathe 0 0 255\r\n");
    expected_colour(0,0,0);
    host_time(250); host_line("DBG1_SET " SID " 2 0 0 0\r\n");
    expected_colour(0,0,127);
    host_time(500); host_line("DBG1_SET " SID " 3 0 0 0\r\n");
    expected_colour(0,0,255);
    host_time(750); host_line("DBG1_SET " SID " 4 0 0 0\r\n");
    expected_colour(0,0,127);
    host_time(1000); host_line("DBG1_SET " SID " 5 0 0 0\r\n");
    expected_colour(0,0,0);
    host_time(1250); host_line("DBG1_SET " SID " 6 0 0 0\r\n");
    expected_colour(0,0,127);
    host_line("DBG1_LED " SID " 7 breathe 0 255 0\r\n");
    expected_colour(0,0,0);
    host_time(1500); host_line("DBG1_SET " SID " 8 0 0 0\r\n");
    expected_colour(0,127,0);
    host_time(1750); host_line("DBG1_SET " SID " 9 0 0 0\r\n");
    expected_colour(0,255,0);
    for(unsigned i=0;i<20;++i) pulse(467);
    assert(!Ultrasonic_HasObstacle());
    expected_colour(0,255,0);
    for(unsigned i=0;i<16;++i) pulse(2000);
    assert(!Ultrasonic_HasObstacle());
    expected_colour(0,255,0);
    host_line("DBG1_LED " SID " 10 blink 255 0 0\r\n");
    expected_colour(255,0,0);
    host_time(2250); host_line("DBG1_SET " SID " 11 0 0 0\r\n");
    expected_colour(0,0,0);
    host_time(2750); host_line("DBG1_SET " SID " 12 0 0 0\r\n");
    expected_colour(255,0,0);
    host_line("DBG1_LED " SID " 13 manual 0 0 0\r\n");
    expected_colour(0,0,0);
    host_line("DBG1_LED " SID " 14 auto 0 0 0\r\n");
    assert(!Debug_GetLedOverride(&r,&g,&b));
    assert(host_nonzero_writes==0); all_off();
  } else if (!strcmp(name,"sensor_debug_timeout")) {
    begin(); pulse(467);
    host_line("DBG1_LED " SID " 1 manual 1 2 3\r\n");
    host_time(200); Robot_Update();
    assert(strstr(host_output,"DBG1_SENSOR " SID " 1 200 "));
    host_time(600); Robot_Update();
    assert(Mission_GetState()==MISSION_IDLE && strstr(host_output,"DBG1_TIMEOUT"));
    assert(!Debug_GetLedOverride(&r,&g,&b)); all_off();
    host_line("DBG1_HELLO " SID " 8\r\n");
    assert(Mission_GetState()==DEBUG_TEST); all_off();
    host_time(700); host_line("DBG1_SET " SID " 1 0 0 0\r\n");
    host_time(1299); host_line("DBG1_SET " SID " 1 0 0 0\r\n");
    assert(Mission_GetState()==DEBUG_TEST);
    host_time(1300); Robot_Update();
    assert(Mission_GetState()==MISSION_IDLE); all_off();
  } else if (!strcmp(name,"sensor_debug_reject")) {
    static const char *bad[]={
      "DBG1_SET " SID " 1 1 10000 0\r\n",
      "DBG1_SET " SID " 1 0 0 1\r\n",
      "DBG1_LED " SID " 1 manual 256 0 0\r\n",
      "DBG1_LED " SID " 1 manual 1 2 3 extra extra\r\n"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(bad[0]);++i) {
      begin(); host_line(bad[i]);
      assert(Mission_GetState()==MISSION_FAULT); all_off();
      assert(host_nonzero_writes==0 && !Debug_GetLedOverride(&r,&g,&b));
    }
    begin(); host_line("START\r\n");
    assert(Mission_GetState()==DEBUG_TEST); all_off();
    host_line("DBG1_LED " SID " 1 manual 1 2 3\r\n");
    Robot_OnUartError(&huart2); Robot_Update();
    assert(Mission_GetState()==MISSION_FAULT && !Debug_GetLedOverride(&r,&g,&b)); all_off();
  } else assert(!"unknown sensor debug scenario");
}
