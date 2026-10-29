# STM32 주행·클리닝 펌웨어

STM32가 주행 모터, 엔코더, 전면·측면 브러시, 워터펌프, 초음파 센서와 RGB LED를 제어한다. Jetson은 카메라 영상과 AI 분석을 처리하고, STM32에는 임무 명령과 장치별 제어 명령을 보낸다.

프로젝트는 `code/encoder`에 있다. 폴더 이름은 초기 엔코더 시험 때 사용하던 이름이며, 현재는 전체 로봇 제어 펌웨어가 들어 있다. STM32CubeIDE 빌드 타깃은 **STM32F103RB**다.

## 빌드와 설정 변경

1. STM32CubeIDE에서 **File → Import → General → Existing Projects into Workspace**를 선택한다.
2. 이 저장소의 `stm_code/code/encoder`를 선택하고 `encoder` 프로젝트를 가져온다.
3. 프로젝트를 Refresh한 뒤 **Project → Clean → Build Project**를 실행한다.
4. 생성된 펌웨어를 사용하는 보드에 맞는 디버거 설정으로 다운로드한다.

`Core`, `Drivers`, `.project`, `.cproject`, `encoder.ioc`, 링커 스크립트를 함께 사용한다. `Debug/`와 `Release/`는 빌드할 때 다시 생성되므로 저장소에 포함하지 않는다. 기존 `objects.list`나 `subdir.mk`를 직접 수정하지 말고 IDE에서 빌드 파일을 다시 생성한다.

PWM이나 구간 수를 바꿔도 이미 보드에 올라간 펌웨어는 바뀌지 않는다. 헤더를 수정한 뒤 다시 빌드하고 다운로드해야 한다. CubeMX 설정을 바꿔 코드를 재생성할 때는 기존 초기화와 콜백 연결이 유지되는지 변경 내용을 확인한다.

## 먼저 볼 파일

경로는 `code/encoder/Core` 기준이다.

| 파일 | 담당 내용 |
|---|---|
| `Inc/robot_config.h` | 주행·브러시·펌프 PWM, 구간 수, 명령 만료 시간 |
| `Inc/drive_sync_config.h` | 전진·후진 PI 게인, 엔코더 보정, 출력 제한, 출발 가속 |
| `Inc/robot_board.h` | 브러시·펌프의 타이머 채널과 방향 핀 |
| `Src/main.c` | 클록·GPIO·타이머·UART 초기화와 HAL 콜백 |
| `Src/robot_app.c` | 모듈 실행 순서, LED 상태 표시, 정지 처리 |
| `Src/mission.c` | 임무 FSM, 구간 이동, 촬영 대기, 복귀 |
| `Src/drive_control.c` | 모터 방향·PWM 적용과 정지 |
| `Src/drive_sync_pid.c` | 두 엔코더의 속도 차이 계산과 주행 보정 |
| `Src/actuators.c` | 브러시·펌프 명령 처리, OFF 우선 처리, 출력 만료 |
| `Src/uart_protocol.c` | RR2 세션, 명령·이벤트·상태 송수신 |
| `Src/debug_console.c` | 장치별 시험 명령과 디버그 세션 |
| `Src/ultrasonic.c` | 초음파 측정, 장애물 정지·해제 판단 |
| `Src/rgb_pwm.c` | BAM 방식 RGB LED 밝기 제어 |
| `Src/fault_stop.c` | 독립 워치독과 예외 발생 시 출력 차단 |

`drive_sync_pid`라는 파일명을 사용하지만 현재 D 게인은 0이므로 기본 설정은 PI 제어다.

## 현재 주행·출력 설정

다음 값은 저장소의 `robot_config.h`와 `drive_sync_config.h`에 들어 있는 설정이다.

| 설정 | 값 | 의미 |
|---|---:|---|
| `PWM_DUTY` | 1600 | 디버그 화면의 100% 기준 |
| `MISSION_FORWARD_PWM_DUTY` | 1200 | 임무 전진, 디버그 기준 75% |
| `MISSION_REVERSE_PWM_DUTY` | 1200 | 임무 후진, 디버그 기준 75% |
| `LIVE_PWM_DUTY` | 전진 설정과 동일 | 실시간 클리닝 중 주행 |
| 전면·측면 브러시 낮은 단계 | 1000 | 타이머 듀티 약 27.8% |
| 브러시 높은 단계 | 1200 | 타이머 듀티 약 33.3% |
| `WATER_PUMP_PWM_DUTY` | 3600 | 펌프 듀티 100% |
| `ENCODER_SECTION_LIMIT` | 6 | 임무 기본·최대 구간 수 |
| `MISSION_TRANSITION_PAUSE_MS` | 1000ms | 임무 단계 사이 대기 |

TIM1·TIM2의 ARR은 3599다. 따라서 주행 PWM 1200은 실제 타이머 듀티로 약 33.3%이며, 디버그 화면의 75%와 기준이 다르다. 브러시 명령 이름에 남아 있는 `12_5`, `25`도 현재 실제 듀티를 뜻하지 않는다. 속도를 수정할 때는 명령 이름보다 위 상수값을 확인한다.

엔코더 보정은 10ms마다 실행한다. 전진·후진 게인은 각각 P=8, I=0.5, D=0이며, 보정량은 최대 400, 표본 사이 보정 변화량은 최대 20으로 제한한다. 출력 포화 시 적분이 과도하게 쌓이지 않도록 처리한다. 엔코더 부호·해상도 보정과 제어 오차의 피드백 부호는 별도 설정이므로 배선을 바꿨다면 함께 확인해야 한다.

전진 오른쪽 목표 비율은 99.5%, 후진 왼쪽 목표 비율은 100%다. 이 값은 시제품에 맞춘 보정값이며 다른 기구에도 그대로 맞는 값은 아니다. 정지 상태에서 출발할 때는 기준 PWM의 30%부터 500ms 동안 출력을 올린다. 이 시작 비율도 모터와 하중에 맞춰 조정하는 값이다.

## 임무 순서와 Jetson 연동

```text
대기 → 준비 → 초기 검사 → 출발점 복귀 → 실시간 클리닝
     → 재검사 시작점 복귀 → 재검사 → 최종 복귀 → 완료
```

초기 검사와 재검사에서는 구간 경계에서 주행을 멈추고 Jetson에 촬영을 요청한다. Jetson이 영상을 저장하고 분석 작업을 접수한 뒤 `CAPTURE_OK`를 보내면 다음 구간으로 이동한다. 이 응답은 촬영·접수 확인이며, AI 분석이 모두 끝났다는 뜻은 아니다.

RR2의 `PREPARE`에서 다음 임무에 사용할 구간 수를 1~6으로 지정할 수 있다. STM32의 기본·최대값은 6이며 Jetson 쪽 구간 설정도 이 범위에 맞춰 사용한다. TIM3가 구간 경계를 판단하고 TIM4는 좌우 주행 보정에 사용한다. 구간 번호는 엔코더 기준의 논리 구간이며, 고정된 미터 단위 거리로 해석하지 않는다.

재검사가 끝나면 `RESCAN_DONE`을 보내고 최종 복귀를 수행한다. 출발점까지 돌아온 뒤에 `MISSION_DONE`을 보낸다. Jetson의 결과 정리·완료 음성 처리는 이 주행 완료 이후의 상위 작업이다.

## 주요 핀과 타이머

실제 매핑은 `main.h`, `robot_board.h`, `stm32f1xx_hal_msp.c`를 기준으로 한다.

| 기능 | 핀 / 주변장치 |
|---|---|
| Jetson 통신 | USART2, PA2 TX / PA3 RX, 115200bps, 8N1 |
| 주행 출력 | TIM2 CH1~4: PA0 / PA1 / PB10 / PB11 |
| 구간·오른쪽 엔코더 | TIM3 CH1/2: PC6 / PC7 |
| 왼쪽 엔코더 | TIM4 CH1/2: PB6 / PB7 |
| 전면 펌프 | TIM1 CH1 PA8, 방향 PA5 / PA6 |
| 측면 브러시 | TIM1 CH2 PA9, 방향 PC13 / PC14 |
| 측면 펌프 | TIM1 CH3 PA10, 방향 PC8 / PC9 |
| 전면 브러시 | TIM1 CH4 PA11, 방향 PB13 / PB14 |
| 초음파 센서 | PB8 TRIG / PB9 ECHO |
| RGB LED | PB3 Blue / PB4 Green / PB5 Red |

Jetson과 STM32의 TX/RX는 서로 교차 연결하고 공통 GND와 사용 중인 변환기의 전압 규격을 확인한다. 설정에 USART3도 있지만 임무용 RR2 통신은 USART2를 사용한다.

## 통신과 정지 처리

RR2는 `HELLO/READY`로 세션을 확인한 뒤 CRC, 명령 순번, ACK/NACK와 제한 재전송을 사용한다. 같은 명령을 다시 받아도 중복으로 실행하지 않도록 처리한다. Jetson의 SerialWorker는 명령 ACK 대기 중에도 수신과 PING을 계속 처리한다. 기본 PING 주기는 100ms다.

| 감시 대상 | 현재 처리 |
|---|---|
| RR2 연결 | 600ms 동안 유효한 연결 확인이 없으면 통신 이상 처리 및 출력 정지 |
| 브러시·펌프 명령 | 장치별 명령이 1000ms 동안 갱신되지 않으면 해당 출력 정지 |
| 엔코더 무펄스 | 기본 500ms 감시. 출발 가속 중에는 가속 종료 후부터 판정 |
| 주행 제어 갱신 | 제어 간격과 main 루프 갱신 지연 감시 |
| 소프트웨어 E_STOP | 주행·청소 출력 정지, 대기 명령 제거, FAULT 유지 |
| main 루프 정지 | IWDG로 명목 약 0.5초 후 MCU 리셋. 내부 LSI 편차에 따라 달라짐 |
| CPU 예외·Error_Handler | PWM·방향 출력을 직접 끄고 빨간 LED를 켠 상태로 정지 |

초음파 정지 기준은 **명목 10cm 이내 연속 3회**다. 시작 허용과 정지 해제는 최근 20개 측정 중 원거리 판정 16개를 기준으로 한다. 현재 운용 코드에서는 `NO_ECHO`도 이 해제 판정의 원거리 쪽에 포함한다. 이는 센서가 정상이라는 판정이 아니며, 측정 불가 상태와 유효 거리값은 별도로 구분한다. 과거 문서의 25cm 설정과는 다르다.

이 동작은 축소 시제품의 소프트웨어 제어다. 실제 정지거리는 주행 속도·관성·노면에 따라 달라지며, 원격 정지 명령이 물리 비상정지 장치를 대신하지는 않는다.

## RGB LED 표시

TIM2의 50µs 주기 인터럽트를 공유해 8비트 BAM으로 밝기를 조절한다. 주기 전체는 12.75ms이며 별도의 RGB 전용 PWM 타이머 없이 상태에 따라 색상과 패턴을 바꾼다.

| 상태 | 표시 |
|---|---|
| 임무 세션 연결 전 대기 | 소등 |
| 임무 세션 연결 후 대기 | 느린 파란색 밝기 변화 |
| 시작 준비 PREPARE | 빠른 파란색 점멸 |
| 단계 전환 대기·센서 시험 | 파란색 밝기 변화 |
| 임무 진행 | 초록색 밝기 변화 |
| 장애물·래치된 FAULT | 빨간색 점멸 |
| 치명적 CPU 예외 | 빨간색 점등 유지 |

## 로컬 시험

저장소 루트에서 `code`로 이동해 실행한다. Python과 호스트 C 컴파일러가 필요하다. Windows에서는 MSVC, Linux에서는 GCC/Clang을 사용할 수 있다.

```bash
cd code
python -m unittest discover -s tests -p 'test_stm_*.py' -v
python -m unittest discover -s tests -p test_actuator_debug_firmware.py -v
python -m unittest discover -s tests -p test_drive_sync_pid.py -v
```

호스트 시험은 실제 `Core/Src`의 제어 코드를 가짜 HAL과 함께 컴파일해 임무·UART·출력·센서 상태 전이를 확인한다. `test_stm_arm_build.py`는 ARM GCC가 설치된 환경에서 Cube 초기화 코드와 드라이버까지 빌드하고 링크한다. 시험 중 보드를 연결하거나 펌웨어를 플래시하지 않는다. 모터 부하, 실제 배선, 센서 오차와 정지 반응은 보드와 축소 테스트베드에서 따로 확인해야 한다.
