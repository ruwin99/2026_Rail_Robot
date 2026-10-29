# WAV 알림 구현 검증 기록

로컬 Windows에서 검사했으며 Jetson 배포, 실제 ROS 2 DDS 통신, USB 스피커 출력 및
초음파 실물 동작을 수행하지 않았다. WAV 파일은 사용자가 준비한다.

## 구현 범위

- `audio_alerts.py`: 단일 worker/단일 aplay, PCM WAV 검증, 유효기간/중복/대기열 제한.
- `ros/audio_alert_node.py`: tts 요청·결과·상태·초음파 상태와 ready 서비스.
- `ros/mission_node.py`: 시작 음성 완료 성공 이후에만 START, 최종 DONE/drain 후 완료 음성.
- `ros/stm_bridge_node.py`: 기존 HOLD/CLEAR 처리에 retained Bool 상태를 추가. 이전 미커밋 초음파 변경 보존.
- launch/CMake/entrypoints/package/requirement: 기본 mission 실행 연결과 ALSA 의존성.
- `ros/monitor_node.py`: 음성 상태/결과 및 retained 초음파 상태 관측.
- `ros2/README.md`, `audio.example.yaml`: 파일 배치·실행·설정·음성 단독 시험.
- 기존 ROS 시험의 crack mask 128행 fixture를 실제 settings의 ROI 높이(180행)에 맞췄다.
  모델·영상 처리 구현과 다른 기존 STM 시험 변경은 수정하지 않았다.

## 최종 관련 시험

Bundled Python, OpenCV 4.12.0.88 및 pyserial 3.5를 임시 테스트 경로에서 사용했다.
임시 패키지는 `.gitignore`로 제외했다. 기존 프로젝트 가상환경이나 Jetson 환경은 변경하지 않았다.

| 검사 | 결과 |
|---|---|
| `unittest discover -s tests -p 'test*audio*.py'` | 17개 통과 |
| `unittest discover -s tests -p 'test_ros_*.py'` | 104개 통과. 위 음성 시험 중 ROS adapter/mission 항목과 중복 |
| Python 문법 검사 | `rail_robot/code` 아래 581개 파일 통과(임시 의존성 포함) |
| 전체 `unittest discover -s tests` | 642개 실행, 실패 3개, 건너뜀 7개 |

전체 시험에서 남은 실패는 이번에 수정하지 않은 영상/제어 코어의 기존 ROI 시험이다.

1. `test_dual_engine_main.DualEngineMainTests.test_crack_warmup_failure_closes_all_detectors_before_uart`
   - 실제 ROI `(180,1280,3)`와 시험 기대값 `(128,1280,3)` 불일치.
2. `test_node_robotics.ControlAndUartNodeCoreTests.test_new_front_crack_stops_before_four_result_window_is_ready`
   - 시험의 균열 입력에 대해 immediate-off 기대값 불일치.
3. `test_node_vision.NodeVisionBoundaryTests.test_delayed_inference_retains_capture_time_and_cannot_be_ready`
   - 실제 ROI `(180,1280,3)`와 시험 기대값 `(128,1280,3)` 불일치.

전체 출력은 로컬 `code/tmp/audio-full-suite-verified.log`에 있다. 전부 통과했다고 보고하지 않는다.

## 실제로 검사한 조건

- 실제 mission 요청 JSON → AudioAlertNode → AudioAlerts worker → 결과 JSON → mission UART START 순서.
  사운드 출력 프로세스와 ROS 전송만 대체했다. 재생 중 START 없음, 성공 종료 후 START 1회.
- 실패·만료·fault·종료·다른 ID/cue·문자열 success 및 뒤늦은 응답으로 START를 승인하지 않음.
- 5초 이상 시작 음성 대기를 일반 준비 제한과 구분.
- HOLD가 시작 음성을 중단하면 실패. 반복 HOLD가 같은 경고를 중첩 재생하지 않음.
- CLEAR는 재생 중인 경고까지 취소. HOLD 유지 시 경고 클립 완료 후 다음 재생.
- 완료는 현재 경고 클립 경계에서 한 번 재생하고 HOLD 유지 시 경고 반복을 재개.
- 누락/잘림 WAV, aplay 누락/실패, 만료, bounded queue/cache, terminate/kill/wait 및 cleanup 실패.
- 늦은 음성/monitor 구독은 bridge가 관측한 마지막 초음파 상태 QoS를 요청.

## 현장 확인이 필요한 경계

WAV 세 파일, USB 오디오 장치 선택/볼륨, 실물에서 재생 끝과 START 순서, 반복/해제 지연,
ros_shutdown 후 재생 프로세스 종료를 확인해야 한다.
현재 STM은 HOLD/CLEAR 변화만 발행하므로 bridge 자체를 HOLD 중 재시작하면 이전 상태를 복원하지 못한다.
안내음의 재생 성공은 ALSA 프로세스 성공을 뜻하며 음소거 상태나 실제 청취 여부까지 증명하지 않는다.
