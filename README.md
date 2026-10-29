# AI 기반 항만 크레인 레일 점검·클리닝 로봇

축소 레일 위를 이동하면서 표면을 촬영하고, 녹·균열 후보·이물질을 분석하는 로봇입니다. 검사한 구간을 청소한 뒤 다시 촬영해 전후 상태를 비교합니다. Jetson은 영상 분석과 기록을, STM32는 주행과 브러시·워터펌프 제어를 담당합니다. 웹에서는 실시간 영상, 임무 상태, 검사 이력을 확인하고 실행·정지를 요청할 수 있습니다.

2026년 스마트해운물류 프로젝트에서 사용한 Jetson, STM32, 웹 대시보드 코드를 한 저장소에 정리했습니다. 기본 실행 경로는 **ROS 2**이며, 이전 단일 프로세스 실행 코드는 호환용으로 남겨 두었습니다.

## 시스템 구성

| 구성 | 담당 작업 |
|---|---|
| Jetson Orin Nano | 다중 카메라 입력, TensorRT 추론, 촬영 및 분석 결과 저장, ROS 2 노드 실행 |
| STM32 | 엔코더 기반 구간 이동, 좌우 주행 보정, 브러시·워터펌프 제어, 초음파 감지, RGB LED |
| UART RR2 | 임무 명령, 촬영 요청·접수 확인, 출력 제어, 상태 피드백 |
| 웹 대시보드 | WebRTC 영상, 임무 진행 상태, 검사 이력 및 청소 전후 비교, 원격 요청 |
| Firebase | 사용자·기기 인증, 상태와 명령 문서, 연결 협상 정보, 검사 이미지 및 결과 저장 |

윗면 카메라는 이물질과 균열 후보를, 옆면 카메라는 녹과 균열 후보를 분석합니다. 결과는 표면 상태를 확인하기 위한 보조자료입니다. 와이어로프 검사, 자동 윤활, 구조 강도나 잔존 수명 판정은 이 코드의 구현 범위에 포함하지 않습니다.

## 임무가 진행되는 순서

1. **준비**: UART 연결과 모델·촬영 준비 상태를 확인합니다. 시작 안내 음성이 정상적으로 끝난 뒤 주행 명령을 보냅니다.
2. **초기 검사**: 지정 구간마다 정지해 윗면·옆면 영상을 저장하고 분석을 접수합니다.
3. **복귀 및 클리닝**: 출발 쪽으로 돌아온 뒤 전진하면서 실시간 분석 결과에 따라 브러시와 펌프를 제어합니다.
4. **재검사**: 검사 시작 위치로 돌아와 같은 구간을 다시 촬영합니다.
5. **최종 복귀 및 완료**: 출발점 복귀와 분석·저장 완료를 확인하고 종료 안내를 재생합니다.

현재 작업본의 기본 검사 목표는 **초기 6구간 + 재검사 6구간**입니다. 구간 번호는 엔코더 이벤트에 따른 논리 구간이며, 절대 위치 좌표가 아닙니다. Jetson과 STM32 양쪽 설정이 맞아야 하며, 이전 펌웨어와 구간 수가 다르면 준비 단계에서 거절될 수 있습니다.

STM32가 보내는 촬영 요청에 Jetson이 `CAPTURE_OK`로 응답하면 다음 이동을 진행합니다. 이 응답은 영상 저장과 분석 접수의 확인입니다. 해당 영상의 AI 분석이 끝났다는 의미는 아니며, 분석 결과는 별도로 저장됩니다.

## 실시간 분석과 정지 촬영 분석

| 구분 | 실시간 분석 | 정지 촬영 분석 |
|---|---|---|
| 목적 | 주행 중 표면 상태 확인, 클리닝 출력 판단 | 구간별 기록, 청소 전후 비교 |
| 입력 | 카메라별 최신 프레임과 검사 영역 | 정지 후 저장한 원본 영상 |
| 구성 | 경량 녹 모델, 실시간 균열 모델, YOLO 이물질 모델 | 촬영용 녹·균열 모델 및 검사 기록 처리 |
| 최적화 | TensorRT, FP16, 모델 교대 실행, GPU 전처리·후처리 | 저장 영상의 분석 품질과 결과 추적에 중점 |
| 출력 | 영상 표시, STM32 제어 판단 | 이미지, XLSX, JSON, 웹 검사 이력 |

녹 모델의 픽셀 분류 후처리를 GPU에서 수행하고, 균열 모델에는 GPU 리사이즈·정규화와 선형 복원을 적용했습니다. 이물질 비율은 검출 박스의 합집합 면적 비율입니다. 녹의 픽셀 점유율이나 실제 이물질 윤곽 면적과 같은 지표로 취급하지 않습니다.

모델 엔진 파일은 저장소에 포함하지 않습니다. 실행 시 모델 경로와 SHA-256을 확인하며, Jetson·TensorRT 환경이 달라지면 해당 환경에 맞는 엔진 준비가 필요합니다. 모델별 설정 위치와 실행 인자는 [Jetson 실행 안내](code/README.md)에 정리했습니다.

## 주행과 출력 제어

양쪽 엔코더의 회전량 차이를 이용해 전진·후진 PI 보정을 적용합니다. 보정 상한, 주기별 변화량 제한, 적분 누적 제한과 출발 가속을 함께 사용합니다.

| 현재 설정 | 값 / 의미 |
|---|---|
| 전진·후진 기준 PWM | 각각 1200. 디버그 화면의 100% 기준값 1600에 대한 75% |
| 워터펌프 PWM | 3600. 타이머 ARR=3599 기준 최대 출력 |
| RR2 연결 감시 | Jetson PING 100ms, STM32 heartbeat 제한 600ms |
| 브러시·펌프 명령 감시 | 1초 동안 명령이 갱신되지 않으면 출력 정지 |
| 초음파 정지 조건 | 현재 펌웨어 기준 명목 10cm 이내 연속 3회 |

디버그 출력 비율과 타이머의 실제 듀티는 다릅니다. 예를 들어 주행 PWM 1200은 타이머 3600 카운트 주기에서 약 33.3%입니다. 설정값은 [robot_config.h](stm_code/code/encoder/Core/Inc/robot_config.h), PI 관련 값은 [drive_sync_config.h](stm_code/code/encoder/Core/Inc/drive_sync_config.h)에 있습니다.

통신에는 CRC 검사, 명령 순번, ACK/NACK, 제한 재전송을 사용합니다. 같은 명령이 재전송돼도 중복 실행하지 않도록 처리하며, 통신 중단·엔코더 정체·제어 지연·소프트웨어 비상정지에 대응하는 출력 차단 경로를 둡니다. RGB LED와 로컬 WAV 음성으로 준비·진행·장애물·완료 상태를 안내합니다.

## 저장소 구조

```text
.
├─ code/
│  ├─ jetson_code/          # 추론, 검사 기록, UART, ROS 노드 구현
│  ├─ ros2/                 # 메시지·서비스 정의, launch, ROS 패키지
│  ├─ scripts/              # 실행·진단 명령, systemd 서비스 예시
│  ├─ config/               # Firebase·WebRTC 설정 예시
│  ├─ tests/                # Python 회귀시험, STM32 호스트·ARM 빌드 시험
│  ├─ tools/                # 모델 변환·벤치마크·통신 점검 도구
│  └─ docs/                 # 영상·검사 데이터 연동 설명
├─ stm_code/code/encoder/   # STM32CubeIDE 프로젝트, HAL·CMSIS, 펌웨어
├─ rail_dashboard/          # Firebase 온라인 화면 및 로컬 조회 화면
└─ docs/                    # 통합 저장소 검증 기록
```

## 시작하기

### 1. 코드 받기

Jetson용 예제와 기존 서비스 파일은 `~/Desktop/rail_robot/code` 경로를 기준으로 합니다. 새 저장소도 아래처럼 `rail_robot`이라는 폴더 이름으로 받으면 경로를 그대로 사용할 수 있습니다. 해당 폴더에 기존 작업이 있다면 덮어쓰지 말고 다른 위치에 받은 뒤 서비스 경로를 조정합니다.

```bash
cd ~/Desktop
git clone https://github.com/ruwin99/2026_Rail_Robot.git rail_robot
cd rail_robot
```

### 2. Jetson 준비 및 ROS 2 빌드

사용 환경은 Ubuntu 22.04, ROS 2 Humble, Python 3.10, JetPack의 CUDA·TensorRT입니다. 현재 의존성 스크립트는 TensorRT 10.3.x를 검사합니다. 먼저 [Jetson 안내](code/README.md)에 따라 모델 엔진, 카메라 경로, UART 장치, 음성 파일과 Python 의존성을 준비합니다.

```bash
cd ~/Desktop/rail_robot/code/ros2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rail_robot_interfaces rail_robot_nodes
source install/setup.bash
```

### 3. 노드 실행과 임무 시작

첫 번째 터미널에서 필요한 장치·모델 설정을 적용한 뒤 실행합니다. `run.sh` 실행만으로 로봇이 출발하지는 않습니다.

```bash
cd ~/Desktop/rail_robot/code
source /opt/ros/humble/setup.bash
source ros2/install/setup.bash
export ROS_DOMAIN_ID=77
bash scripts/run.sh mode:=mission display:=false
```

두 번째 터미널에서도 같은 ROS 환경과 도메인을 사용합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/rail_robot/code/ros2/install/setup.bash
export ROS_DOMAIN_ID=77
ros2 service call /mission/start std_srvs/srv/Trigger '{}'
```

시작이 거절되면 응답 메시지와 노드 로그를 확인합니다. UART 준비, 모델 로딩, 음성 재생, 장애물 상태 등 준비 조건을 확인한 뒤 다시 요청해야 합니다. 실행 옵션·센서별 진단·원격 실행 서비스 설정은 [code/README.md](code/README.md)를 참고합니다.

### 4. STM32와 웹 대시보드

- **STM32**: STM32CubeIDE에서 `stm_code/code/encoder`를 가져와 빌드합니다. [펌웨어 안내](stm_code/README.md)에 핀 구성과 설정 위치를 정리했습니다.
- **웹**: `rail_dashboard`에서 의존성을 설치하고 Firebase 연결 정보를 설정합니다. 온라인 화면과 로컬 조회 화면은 실행 방법이 다릅니다. [대시보드 안내](rail_dashboard/README.md)를 참고합니다.

웹의 프로세스 실행 요청과 임무 시작 요청은 별개입니다. 프로세스 실행은 Jetson의 ROS 스택을 준비하고, START 요청은 준비된 임무를 시작합니다. 원격 정지 요청도 연결과 권한 확인을 거치는 소프트웨어 명령입니다.

## 검사 결과와 설정 파일

촬영 원본·분석 이미지, 구간별 XLSX, 웹 연동 JSON을 저장합니다. 모델 식별 정보와 분석 상태를 함께 남기므로 분석 실패나 미측정을 검출값 0으로 바꾸지 않습니다. 웹에서는 검사 단계, 카메라, 구간별 기록을 구분해 청소 전후를 비교합니다.

Firebase 계정 정보, 기기 토큰, TURN 비밀값과 실제 검사 기록은 Git에 올리지 않습니다. 예제 설정을 복사해 로컬에서 작성하고 접근 권한을 제한합니다. WebRTC 영상과 Firestore 연결 협상, 임무 명령, 검사 결과 업로드는 서로 다른 경로입니다. 필요한 Firebase 인증·접근 규칙을 함께 설정해야 합니다.

## 테스트

로컬에서 카메라나 모터를 연결하지 않고 실행하는 회귀시험입니다.

```bash
cd code
python -m unittest discover -s tests -q
```

NumPy·OpenCV·openpyxl 등 Python 의존성이 필요하며, Bash와 C 컴파일러가 필요한 시험도 있습니다. ARM 빌드 시험에는 `arm-none-eabi-gcc`가 필요합니다. 실제 ROS 실행, TensorRT 추론, 카메라·모터 동작은 장치가 있는 환경에서 별도로 확인합니다. 이번 통합본의 실행 결과와 범위는 [검증 기록](docs/VALIDATION.md)에 남겼습니다.

## 모델·외부 코드 이용

학습 데이터, 모델 가중치, TensorRT 엔진과 음성 파일은 별도로 준비합니다. 사용한 모델·데이터·음성의 이용 조건은 배포 용도에 맞게 확인해야 합니다. STM32 HAL·CMSIS 등 포함된 외부 코드의 저작권 표기와 라이선스 파일은 유지했습니다. 이 저장소에 별도 프로젝트 라이선스가 명시되지 않은 부분을 임의의 오픈소스 라이선스로 해석하지 않습니다.

자세한 구분은 [외부 코드 및 자산 안내](THIRD_PARTY_NOTICES.md)를 참고합니다.
