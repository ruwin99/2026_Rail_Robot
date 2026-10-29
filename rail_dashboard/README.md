# 레일 로봇 관리자 대시보드

Jetson이 전송한 검사 기록을 확인하고, 로봇의 영상과 임무 상태를 보면서 시작·정지를 요청하는 웹 화면이다. 실제 온라인 운용 화면은 `firebase-client/`에 있다. `app/`의 로컬 조회 화면과 실행 명령이 다르므로 아래 두 경로를 구분해서 사용한다.

## 구성과 기능

| 경로 | 역할 |
| --- | --- |
| `firebase-client/` | Firebase 로그인, 검사 이력, 실시간 영상·텔레메트리, 원격 명령 |
| `functions/` | 권한이 있는 시청자에게 TURN 임시 계정 발급 |
| `firestore.rules`, `storage.rules` | 관리자·기기 계정의 문서 및 이미지 접근 제어 |
| `turn-server/` | 다른 네트워크 사이의 영상 연결을 위한 coturn 설정 |
| `app/`, `scripts/sync-dashboard-data.mjs` | JSON/이미지를 읽는 로컬 조회 화면과 동기화 도구 |
| `db/`, `drizzle/` | 별도로 남겨 둔 SQL/D1 데이터 구조 |

온라인 화면에서는 날짜와 점검 실행을 선택해 초기 검사·재검사 사진, 녹·균열 후보·이물질 분석 결과를 확인한다. 구간 수는 각 실행의 `capture_target`을 읽는다. 과거 기록과 새 기록의 목표 구간 수가 달라도 고정된 숫자로 덮어쓰지 않는다. 목표가 4였던 기존 기록도 그대로 각 실행 기준으로 표시한다.

녹은 분할된 픽셀의 면적 비율, 이물질은 검출 박스 합집합의 면적 비율로 표시한다. 이물질 비율은 실제 윤곽 면적이 아니며 두 지표를 합산하지 않는다. 분석이 실패했거나 없는 기록을 0%로 표시하지 않고 미측정·실패 상태로 구분한다. 청소 전후 집계가 끝났다는 사실과 로봇 임무가 정상 완료됐다는 사실도 따로 관리한다.

영상은 WebRTC로 전송한다. Firestore에는 영상 자체가 아니라 연결 협상 정보, 임무 상태, 명령 요청과 처리 결과가 저장된다. JPEG와 검사 이미지는 Cloud Storage에 저장한다. 관리자가 볼 수 있는 로봇 목록을 불러와 선택할 수 있으며, 영상 연결은 로봇별 단일 시청 세션 잠금을 사용한다.

원격 제어에는 세 가지 동작이 있다.

- **실행 요청:** Jetson의 `rail-mission-launcher`가 임무 ROS 프로세스를 실행한다.
- **임무 시작 요청:** 실행된 프로세스의 준비 상태를 확인한 뒤 START를 요청한다.
- **원격 정지 요청:** E_STOP 요청을 보내고 처리 결과를 확인한다.

프로세스 실행과 실제 주행 시작은 별개다. 해당 로봇의 시청·운용 권한과 원격 기능 설정이 있어야 버튼을 사용할 수 있다. 원격 정지는 소프트웨어 명령이므로 물리적인 비상정지 장치와 같지 않다.

## Firebase 화면 실행

Node.js 22.13 이상과 npm을 사용한다. 저장소 루트에서 다음과 같이 실행한다.

```bash
cd rail_dashboard
npm ci
cp .env.example .env.local
```

Windows PowerShell에서는 복사 명령 대신 `Copy-Item .env.example .env.local`을 사용한다. Firebase 콘솔의 웹 앱 설정으로 `.env.local`의 `VITE_FIREBASE_*` 항목을 채운다. 이 설정은 브라우저에 포함되는 웹 연결 정보다. 서비스 계정 개인 키, 기기 비밀번호, TURN 공유 비밀키를 `VITE_*` 변수에 넣으면 안 된다.

```bash
npm run firebase:dev
```

터미널에 표시되는 개발 서버 주소로 접속한다. 배포할 정적 파일을 만들거나 생성된 파일을 확인하려면 다음 명령을 사용한다.

```bash
npm run firebase:build
npm run firebase:preview
```

빌드 결과는 `firebase-dist/`에 생성된다. `firebase:start`는 `firebase:preview`와 같은 미리보기 명령이다. `npm run dev`와 `npm run build`는 아래의 별도 로컬 조회 화면을 대상으로 한다.

## Firebase 권한과 로봇 등록

공개 저장소에는 운영 계정과 프로젝트 연결값을 포함하지 않았다. 실행하려는 Firebase 프로젝트에서 다음 설정이 필요하다.

1. Authentication에서 관리자용 Google 로그인과 기기용 Email/Password 로그인을 설정한다. 접속할 도메인도 승인한다.
2. `firestore.rules`, `storage.rules`의 `isAdmin()`에 있는 `example.invalid` 주소를 실제 승인 관리자 주소로 바꾼다. 예시 주소는 로그인 가능한 운영 계정이 아니다. 로컬 규칙 시험은 예시 계정을 사용하므로 운영 규칙과 시험 계정을 혼동하지 않는다.
3. 기기 계정을 만든 뒤 `device_uploaders/<기기 UID>`에 `enabled: true`를 설정한다. Jetson에는 기기 계정을 사용하고 관리자 서비스 계정 키를 넣지 않는다.
4. `robots/<robotId>` 문서를 신뢰할 수 있는 관리 경로에서 등록한다. `enabled`, `senderId`, `viewerUids`를 지정하고, 원격 운용이 필요하면 `operatorUids`, `remoteControlEnabled`, `remoteLaunchEnabled`도 설정한다. `senderId`는 Jetson 기기 UID와 일치해야 한다.
5. Firestore·Storage 규칙과 인덱스를 해당 프로젝트에 배포한다. 로컬 구성만으로 운영 서버의 권한이 바뀌지는 않는다.

Firebase CLI를 설치하고 로그인한 상태에서, 자신이 관리하는 프로젝트 ID를 명시해 배포한다.

```bash
firebase deploy --project YOUR_PROJECT_ID --only hosting,firestore:rules,firestore:indexes,storage
```

위 명령 전에 `npm run firebase:build`를 수행한다. TURN을 사용할 때는 `functions/`의 의존성을 설치하고 `TURN_HOST`, Secret Manager의 `TURN_SHARED_SECRET`을 별도로 설정한다. 설치 방법은 [TURN 안내](turn-server/README.md)에 있다. 프로젝트별 실제 키와 계정 설정은 커밋하지 않는다.

시그널링 문서 구조는 [WebRTC 계약](WEBRTC_SIGNALING_CONTRACT.md), Jetson 실행 설정은 [코드 README](../code/README.md)를 참고한다.

## 로컬 검사 기록 조회

Firebase와 연결하지 않고 저장된 JSON/이미지를 조회하는 경로다.

```bash
npm run data:sync
npm run dev
```

기본 원본은 `../code/outputs/dashboard`다. 다른 폴더의 기록을 읽으려면 다음과 같이 지정한다.

```bash
npm run data:sync -- --source /path/to/outputs/dashboard
```

동기화 결과는 `public/data/inspection-export.json`과 `public/data/assets/`에 생성된다. 저장된 원본·분석 JPEG를 전부 공개 폴더로 복사하는 방식은 아니다. 실제 기록이 없으면 출처가 표시된 예시 데이터를 사용한다. 이 예시 사진과 마스크는 실제 로봇의 AI 성능 결과가 아니다. [사진 출처](public/demo/IMAGE_SOURCES.md)를 함께 보존한다.

로컬 플래너 일정은 현재 브라우저 저장소에 보존되며 다른 PC로 자동 동기화되지 않는다. `db/`와 `drizzle/`의 SQL 구조를 포함했다고 해서 온라인 SQL 서비스가 구성된 것은 아니다. 현재 Firebase 운용 경로와는 별개다.

## 검사 데이터

Jetson이 내보내는 `schema_version: 1` JSON은 `runs`, `model_provenance`, `captures`, `analyses`, `artifacts`, `run_summaries` 배열을 포함한다. 비율은 0~1 실수로 전달하며, 카메라 위치와 검사 단계·구간 번호로 기록을 연결한다. XLSX 수식의 캐시 값을 읽는 대신 JSON의 픽셀 수로 요약값을 계산한다.

균열 후보나 표면 오염률은 정비 판단에 참고할 자료다. 이 화면에서 구조 강도, 균열 깊이, 잔존 수명을 판정하지 않는다.

## 개발 확인

```bash
npm run typecheck
npm run firebase:build
node --test tests/data-sync.test.mjs tests/obstacle-comparison.test.mjs tests/mission-stages.test.mjs tests/node-health.test.mjs tests/remote-command-selection.test.mjs tests/webrtc-peer.test.mjs
node --test functions/turn.test.js
```

규칙 시험과 가상 원격 제어 시험은 Firebase Emulator가 필요하다. 실제 프로젝트 대신 `demo-` 프로젝트로 실행하는 절차는 [가상 제어 시험](REMOTE_CONTROL_VIRTUAL_TEST.md)에 정리했다. 이 시험은 실제 UART나 모터 출력을 켜지 않는다.
