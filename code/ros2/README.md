# ROS 2 노드 구성

설치, 모델 설정, 전체 임무 실행 방법은 [Jetson 실행 안내](../README.md)에 정리했습니다. 이 문서는 노드별 책임과 문제를 확인할 위치를 설명합니다.

## 빌드

```bash
cd ~/Desktop/rail_robot/code/ros2
source /opt/ros/humble/setup.bash
rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install --packages-select rail_robot_interfaces rail_robot_nodes
source install/setup.bash
```

`rail_robot_interfaces`는 메시지·서비스를, `rail_robot_nodes`는 실행 노드를 설치합니다. Python 구현은 `../jetson_code`와 공유하므로 이 폴더만 따로 복사하지 않습니다. `build/`, `install/`, `log/`는 빌드 결과이며 저장소에 포함하지 않습니다.

## 노드별 책임

| 노드 역할 | 담당 내용 |
| --- | --- |
| camera | 카메라 입력과 촬영 서비스 |
| vision | 최신 프레임의 실시간 추론 |
| control | 분석 결과 유효성 확인과 액추에이터 판단 |
| stm_bridge | UART의 단일 소유자, RR2 명령·이벤트 교환 |
| mission | 준비 확인, 시작 안내, STM32 임무 진행 상태 연동 |
| inspection | 초기 검사·재검사 이미지 저장과 분석, 최종 보고서 확정 |
| display | 로컬 화면과 웹 전송용 JPEG 합성 |
| audio_alert | WAV 재생과 재생 결과 알림 |
| upload / dashboard_telemetry | 최종 보고서 업로드 / 실행 중 상태 전송 |
| remote_control | 인증된 웹 START·소프트웨어 E_STOP 처리 |
| webrtc_sender | 합성 JPEG를 받아 WebRTC 전송 |

`inspection`과 `report_upload` 등 일부 노드는 실행 파일 이름과 ROS 노드 이름이 다릅니다. 실제 이름은 `ros2 node list`로 확인합니다.

## 실행 모드

```bash
cd ~/Desktop/rail_robot/code
./scripts/run.sh --show-args
./scripts/run.sh mode:=camera
./scripts/run.sh mode:=vision display:=false
./scripts/run.sh mode:=mission display:=false serial_port:=/dev/ttyACM0
```

| mode | 용도 |
| --- | --- |
| camera | 카메라 화면 |
| front_camera / side_camera / foreign_camera | 카메라별 수동 촬영·분석 |
| vision / dual_camera | 다중 카메라 실시간 분석 |
| training | 학습용 원본 영상 수집 |
| mission | 전체 검사·클리닝 임무 |
| motor_encoder / dc_motor / water_pump / actuators | 구동부별 디버깅 |
| led / ultrasonic / sensors | RGB LED·초음파 디버깅 |
| monitor | ROS 상태 조회 |

UART를 쓰는 모드끼리는 동시에 실행하지 않습니다. 센서 디버깅은 임무·카메라·모델을 함께 실행하지 않으며, 실제 임무의 장애물 정지 동작과 구별해서 사용합니다.

## 시작과 상태 확인

다른 터미널에서도 같은 ROS 환경과 domain을 설정합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/rail_robot/code/ros2/install/setup.bash
export ROS_DOMAIN_ID=77
ros2 service call /mission/start std_srvs/srv/Trigger '{}'
ros2 topic echo /mission/state
```

터미널 A도 같은 `ROS_DOMAIN_ID`를 사용해야 합니다. launch의 `namespace`를 설정했다면 조회할 이름 앞에 해당 namespace를 붙입니다.

- `/mission/start`: 준비 상태와 시작 음성을 확인한 뒤 START 요청
- `/mission/state`: 임무 진행 상태
- `/stm/events`: STM32가 보낸 이벤트
- `/runtime/fault`: 실행 중 오류
- `/rail_robot/gui/image/compressed`: 웹 송신용 합성 JPEG
- `/rail_robot/gui/status`: 영상·추론 등 화면 관련 상태

시작 성공 응답과 실제 주행 시작은 별개입니다. STM32의 STARTED 이벤트를 확인합니다. 정상 완료 후에는 분석·보고서와 완료 음성을 정리한 뒤 스택이 자동 종료됩니다. 기본 완료 표시 대기 시간은 20초이며 업로드를 켠 경우 결과를 최대 180초 기다립니다.

## 설정 변경

`launch/robot.launch.py`가 실행 조합을, `../jetson_code/node_runtime/ros/deployment.py`가 기본 모델·장치를 정합니다. `params_file`로 YAML을 지정할 수 있으며 설정 예시는 `config/example.yaml`, `config/audio.example.yaml`에 있습니다.

```bash
./scripts/run.sh mode:=mission params_file:=/절대경로/robot.yaml
```

YAML보다 명시한 launch 인자가 우선하고, 실행 모드가 강제하는 파라미터는 마지막에 적용됩니다. 모델·카메라 등 시작 시 설정은 변경 후 노드를 다시 실행합니다.

## 문제가 생겼을 때

| 증상 | 확인할 내용 |
| --- | --- |
| `rail_robot_nodes`를 찾지 못함 | `colcon build`와 `source install/setup.bash` 여부 |
| CUDA Python 또는 TensorRT import 실패 | JetPack 버전, `requirement.sh`, ROS Python의 `PYTHONPATH` |
| 모델 파일·해시 오류 | `deployment.py`와 YAML의 경로·SHA-256 |
| 카메라를 열지 못함 | USB 연결, `/dev/v4l/by-path/`, 카메라 중복 점유 |
| UART READY가 오지 않음 | 포트, 대응 STM32 펌웨어, 다른 프로세스의 포트 점유 |
| `/mission/start`를 찾지 못함 | mission 모드, ROS domain, namespace, launch 최초 오류 |
| 시작 요청이 대기하다 실패함 | 노드 준비 상태, 음성 파일·장치, `/runtime/fault` |
| 완료 뒤 웹 결과가 없음 | `upload` 설정, 인증정보, 업로드 로그 |

여러 노드가 함께 종료되었다면 마지막 종료 메시지보다 먼저 발생한 예외와 launch의 종료 사유를 확인합니다. 상태 기록은 [실시간 대시보드 문서](../docs/live-dashboard.md), 음성 확인은 [WAV 검증 문서](../docs/audio-verification.md)를 참고합니다.
