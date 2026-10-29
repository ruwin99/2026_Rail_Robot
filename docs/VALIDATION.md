# 통합본 검증 기록

검사일: 2026-10-29. GitHub에 게시할 복사본을 대상으로 실행했습니다. 원본 작업 폴더와 운영 장비에는 변경을 적용하지 않았습니다.

## 결과

| 검사 | 결과 |
|---|---|
| Python·ROS 대체 객체·UART·STM32 회귀시험 | 1,006개 실행, 998개 통과, 8개 건너뜀, 실패·오류 0 |
| STM32 ARM 컴파일·링크 | 기본 설정, PI 활성, PI 비활성 3종 통과. 위 1,006개에 포함 |
| 게시 Python 파일 문법 검사 | 174개 통과 |
| 웹 TypeScript 검사 | 통과 |
| Firebase Vite production build | 통과 |
| Vinext 로컬 조회 화면 build | 통과 |
| 웹 데이터·상태·WebRTC·TURN·SSR 시험 | 33개 통과 |
| 공개 파일 검사 | API 키·개인 키·개인 이메일, 검사 기록·모델 엔진·의존성 폴더 제외 확인 |

Python 시험은 Windows의 Python 3.11에서 UTF-8 모드로 실행했습니다. 8개 건너뜀 중 7개는 ONNX 미설치에 따른 모델 변환 시험이고, 1개는 Python 3.10 전용 예외 타입 차이를 확인하는 시험입니다. 실제 Jetson의 Python 3.10에서 실행한 결과로 해석하면 안 됩니다.

초기 실행은 Windows 기본 문자 인코딩 때문에 자식 프로세스의 로그 읽기 오류가 발생해 중단했습니다. 이후 `PYTHONUTF8=1`을 지정해 전체 시험을 다시 실행했으며, 위 표는 재실행 결과입니다. ARM 빌드는 임시 ELF를 만들고 링크·심벌·크기를 검사했습니다. 보드 플래시는 하지 않았습니다.

## 실행 명령

```bash
cd code
PYTHONUTF8=1 python -m unittest discover -s tests -v
```

PowerShell에서는 다음과 같이 설정합니다.

```powershell
$env:PYTHONUTF8 = '1'
python -m unittest discover -s tests -v
```

웹 검사는 `rail_dashboard/`에서 실행했습니다.

```bash
npm run typecheck
npm run firebase:build
npm run build
node --test tests/data-sync.test.mjs tests/obstacle-comparison.test.mjs tests/mission-stages.test.mjs tests/node-health.test.mjs tests/remote-command-selection.test.mjs tests/webrtc-peer.test.mjs functions/turn.test.js tests/rendered-html.test.mjs
```

이번 작업에서는 실제 카메라·CUDA/TensorRT 장치 추론, ROS DDS 실통신, 모터·펌프·음성 출력, Firebase 운영 서버 및 Emulator 규칙 시험을 실행하지 않았습니다. 로컬 자동 시험과 장비 시험의 범위를 구분합니다.
