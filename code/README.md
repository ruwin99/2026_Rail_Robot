# Jetson 영상 분석과 ROS 2 실행

카메라 영상 수집, AI 분석, STM32와의 임무 연동, 검사 결과 저장 및 웹 연결 코드입니다. 기본 실행은 ROS 2입니다. `scripts/run.sh`는 노드를 준비하며, 주행은 별도의 시작 요청을 받은 뒤 진행합니다.

## 코드 구성

| 경로 | 역할 |
| --- | --- |
| `ros2/rail_robot_nodes/launch/robot.launch.py` | 모드별 노드 조합과 실행 인자 |
| `ros2/rail_robot_interfaces/` | 영상·분석·임무·제어 메시지와 서비스 |
| `jetson_code/node_runtime/ros/deployment.py` | 모델 경로·해시, 카메라 장치, 분석 임계값 |
| `jetson_code/node_runtime/ros/mission_node.py` | 시작 준비와 STM32 임무 상태 연동 |
| `jetson_code/node_runtime/ros/inspection_node.py` | 구간 촬영 접수·분석 대기열·보고서 확정 |
| `jetson_code/node_runtime/control.py` | 전면·측면 클리너와 펌프 판단 |
| `jetson_code/node_runtime/serial_worker.py` | RR2 UART 송수신·응답·재전송 |
| `jetson_code/rust_detector.py`, `hrsegnet_crack_detector.py`, `obstacle_detector.py` | 녹·균열 후보·이물질 추론 |
| `jetson_code/inspection_report.py`, `dashboard_export.py` | XLSX 보고서와 웹 조회용 결과 |
| `tests/`, `tools/` | 테스트, 모델 변환·검증 및 측정 도구 |

`run.py`의 기본 호출도 ROS launch로 연결됩니다. `--legacy`는 예전 전문 시험을 재현하는 경로입니다. 일반 운용 설정은 `run.sh`의 legacy 모델 변수가 아니라 `deployment.py`와 ROS 파라미터를 확인합니다.

## 실행 환경 준비

기준 장치는 Jetson Orin Nano이며 Ubuntu 22.04, ROS 2 Humble, Python 3.10, TensorRT 10.3.x를 사용합니다. JetPack과 ROS는 먼저 설치되어 있어야 합니다. 아래는 저장소를 `~/Desktop/rail_robot`에 내려받은 경우입니다.

```bash
cd ~/Desktop/rail_robot/code
export RAIL_ROBOT_HOME="$PWD"
bash scripts/requirement.sh
source /opt/ros/humble/setup.bash
cd ros2
rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install --packages-select rail_robot_interfaces rail_robot_nodes
source install/setup.bash
cd ..
export PYTHONPATH="$RAIL_ROBOT_HOME/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=77
```

`requirement.sh`는 OpenCV, NumPy, pyserial, TensorRT, ALSA를 확인하고 CUDA Python 12.5.0과 openpyxl 3.1.5를 준비합니다. ROS 기본 실행이 이 설치 스크립트나 venv를 자동으로 실행하지는 않습니다.

### 모델·카메라 설정

학습 데이터, 사전학습 가중치, ONNX 및 TensorRT plan은 포함하지 않습니다. 모델을 별도로 준비해야 하며 다른 JetPack·TensorRT 환경에서는 해당 환경에서 plan을 다시 빌드해야 합니다. 외부 라이브러리는 설치하고, 모델·데이터의 이용 및 배포 조건은 각 원출처를 따릅니다.

| 용도 | 모델 |
| --- | --- |
| 실시간 녹 | MobileNetV2 기반 세그멘테이션, FP16 |
| 실시간 균열 후보 | 추가 학습한 HrSegNet-B32, FP16 |
| 정지 촬영 녹 | ResNet101 기반 세그멘테이션, FP32 |
| 정지 촬영 균열 후보 | HrSegNet-B32, FP32 |
| 이물질 | YOLO26n 기반 객체 탐지, FP16 |

`deployment.py`에는 기준 모델 경로와 SHA-256이 있습니다. 파일과 해시가 맞지 않으면 실행하지 않습니다. 실제 장치의 모델 경로·해시·카메라 경로는 별도 YAML을 작성해 `params_file:=/절대경로/robot.yaml`로 넘길 수 있습니다.

```bash
ls -l /dev/v4l/by-path/
sha256sum /실제/모델/경로/model.plan
```

다음은 YAML 형식 예시입니다. 주석의 모델 값은 보유한 모델에 맞게 채우고, 사용하는 모델 각각에 경로와 해시를 함께 지정합니다.

```yaml
/**:
  ros__parameters:
    top_camera_device: "/dev/video0"
    side_camera_device: "/dev/video2"
    # realtime_rust_engine: "/실제/경로/realtime-rust.plan"
    # realtime_rust_sha256: "실제 파일의 64자리 SHA-256"
    # realtime_crack_engine: "/실제/경로/realtime-crack.plan"
    # realtime_crack_sha256: "실제 파일의 64자리 SHA-256"
    # capture_rust_engine: "/실제/경로/capture-rust.plan"
    # capture_rust_sha256: "실제 파일의 64자리 SHA-256"
    # capture_crack_engine: "/실제/경로/capture-crack.plan"
    # capture_crack_sha256: "실제 파일의 64자리 SHA-256"
    # obstacle_engine: "/실제/경로/obstacle.plan"
    # obstacle_sha256: "실제 파일의 64자리 SHA-256"
```

`top`은 윗면을 촬영하는 전면 카메라이고 `side`는 옆면 카메라입니다. `/dev/videoN`은 연결 순서에 따라 달라질 수 있으므로 운용 시에는 확인한 `/dev/v4l/by-path/...` 경로를 사용합니다.

### 음성 파일

`~/.local/share/rail_robot/audio/`에 `mission_started.wav`, `mission_completed.wav`, `obstacle_detected.wav`를 준비합니다. PCM WAV를 사용하며 파일은 저장소에 포함하지 않습니다. `aplay -l`로 오디오 장치를 확인합니다. 기본 임무는 시작 안내 재생이 정상 종료되어야 START를 보냅니다. 의도적으로 음성을 생략하는 시험에서만 `audio_enabled:=false`를 지정합니다.

## 카메라 확인과 임무 시작

```bash
cd "$RAIL_ROBOT_HOME"
# 카메라 화면만
./scripts/run.sh mode:=camera
# 전면 카메라와 수동 정지 촬영 분석
./scripts/run.sh mode:=front_camera
# 카메라·실시간 추론만, UART 없음
./scripts/run.sh mode:=vision display:=false
```

전체 임무는 대응하는 STM32 펌웨어와 UART 포트를 확인한 뒤 실행합니다. 동일 포트를 다른 디버그 노드가 동시에 열면 안 됩니다. 모델 설정 YAML이 있다면 다음 실행 명령에 `params_file:=/절대경로/robot.yaml`을 추가합니다.

터미널 A:

```bash
cd "$RAIL_ROBOT_HOME"
export ROS_DOMAIN_ID=77
./scripts/run.sh mode:=mission display:=false serial_port:=/dev/ttyACM0
```

터미널 B:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/rail_robot/code/ros2/install/setup.bash
export ROS_DOMAIN_ID=77
ros2 service call /mission/start std_srvs/srv/Trigger '{}'
```

두 터미널의 ROS domain과 namespace를 같게 맞춥니다. 별도 namespace가 있으면 서비스 이름에도 붙입니다. 시작 서비스는 준비 확인과 시작 음성 완료를 기다립니다. 성공 응답은 START 전송을 뜻하며 실제 시작은 STM32의 STARTED 응답과 임무 상태로 확인합니다.

현재 기본값은 초기 검사 **6구간**, 재검사 **6구간**입니다. PREPARE에 구간 수를 함께 보내 Jetson과 STM32의 설정 불일치를 확인합니다. 짧은 시험은 `RAIL_ROBOT_SECTION_TARGET=2`를 설정해 실행할 수 있으며, STM32도 구간 수 인자를 지원해야 합니다.

임무는 초기 검사 → 복귀 → 실시간 클리닝 → 재검사 시작 위치로 복귀 → 재검사 → 최종 복귀 순서로 진행합니다. 마지막 분석·보고서 처리를 정리하고 완료 안내 후 종료합니다.

## 실시간 분석과 정지 촬영 분석

| 구분 | 실시간 분석 | 정지 촬영 분석 |
| --- | --- | --- |
| 목적 | 이동 중 상태 표시와 청소 제어 | 구간 기록과 청소 전후 비교 |
| 입력 | 최신 프레임과 제어 ROI | 지정 구간에서 저장한 원본 영상 |
| 모델 | 처리 지연을 줄인 실시간 모델 | 정지 촬영용 모델 |
| 실행 | 카메라별 분석을 교대로 수행 | 촬영 접수 후 대기열에서 분석 |
| 출력 | 분석 상태·클리너·펌프 판단 | 원본·분석 이미지, XLSX, JSON |

녹 argmax 후처리와 균열 전처리·선형 복원을 GPU에서 수행합니다. 카메라 FPS, 모델 처리 시간, GUI 발행률은 서로 다른 지표로 기록합니다.

STM32의 촬영 요청에 대해 Jetson은 영상을 저장하고 분석 작업을 접수한 뒤 CAPTURE_OK를 보냅니다. 이 응답은 AI 분석 완료가 아닙니다. 분석은 별도로 계속 진행되며 보고서 확정 전에 완료 여부를 확인합니다.

## 결과 저장과 웹 연결

기본 검사 저장 위치는 `~/rail_robot_data/captures`입니다. 검사 단계·구간·카메라 위치와 분석 결과를 기록하고 원본·분석 이미지, XLSX 및 JSON을 생성합니다. 녹·균열 후보는 픽셀 영역을, 이물질은 검출 박스를 기준으로 다룹니다.

- `dashboard_live:=true`: 실행 중 상태와 검사 사진 전송
- `upload:=true`: 확정된 임무 보고서 업로드
- `dashboard_control:=true`: 인증된 원격 START·소프트웨어 E_STOP 요청 처리
- 별도 WebRTC 송신 노드: 합성 영상 전송. 카메라나 UART를 직접 소유하지 않음

원격 영상은 FRONT를 왼쪽, SIDE를 오른쪽에 놓은 1280×360 JPEG이며 임무 상태는 별도 데이터로 보냅니다. 기본 GUI 발행 설정은 12 Hz입니다. Firestore는 연결 협상·명령·상태 데이터, Storage는 검사 이미지를 다룹니다. 영상은 WebRTC로 전송합니다.

Firebase 인증정보와 실제 장치 설정은 저장소 밖에서 관리합니다. `config/firebase.env.example`, `config/webrtc.example.json`은 작성 형식만 제공합니다. 실제 값을 넣은 파일은 커밋하지 않습니다. 프로그램 실행 서비스와 임무 START 요청은 별도 단계입니다.

- [ROS 2 노드 구성과 문제 확인](ros2/README.md)
- [웹 원격 실행과 제어](REMOTE_DASHBOARD_CONTROL.md)
- [실시간 상태·검사 사진](docs/live-dashboard.md)
- [촬영 결과 저장과 전송](docs/live-capture-output.md)
- [WebRTC 송신기](docs/webrtc-jetson.md)
- [WebRTC 연결 규약](docs/webrtc-signaling-contract.md)
- [음성 안내 확인](docs/audio-verification.md)

세부 문서의 장치 경로·실측 수치는 해당 시험 시점의 기록입니다. 현재 기본 모델과 파라미터는 실행 코드의 설정을 기준으로 확인합니다.

## 로컬 확인

장치가 없는 PC에서는 Python 문법과 장치 대체 테스트를 실행합니다.

```bash
cd ~/Desktop/rail_robot/code
python -c "from pathlib import Path; files=list(Path('jetson_code').rglob('*.py'))+list(Path('ros2').rglob('*.py')); [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in files]; print(len(files))"
python -m unittest discover -s tests -p 'test_inspection_report.py'
python -m unittest discover -s tests -p 'test_ros_deployment_vision.py'
python -m unittest discover -s tests -p 'test_ros_inspection.py'
python -m unittest discover -s tests -p 'test_section_target_mode.py'
```

테스트에는 NumPy, OpenCV, openpyxl 등 해당 모듈의 의존성이 필요합니다. 셸·펌웨어 테스트는 Bash 및 C/ARM 도구도 사용합니다. 장치 대체 테스트가 실제 카메라·GPU·모터 동작 확인을 대신하지는 않습니다.
