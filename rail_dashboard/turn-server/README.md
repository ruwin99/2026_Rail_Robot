# TURN 서버 (Compute Engine + coturn)

다른 네트워크에서 대시보드 영상을 볼 때 WebRTC를 중계하는 서버다.

- 브라우저는 영상 연결 직전에 Cloud Function `turnCredentials`에서 65분짜리 임시 계정을 받는다.
- 임시 계정은 표준 TURN REST 방식으로 만든다. 함수와 coturn이 같은 공유 비밀키(`TURN_SHARED_SECRET`)로 HMAC을 계산한다.
- 공유 비밀키는 Secret Manager에만 있다. 저장소, VM 메타데이터, 브라우저에는 두지 않는다.

## 1. 공유 비밀키 만들기 (PowerShell, 한 번)

`rail_dashboard` 폴더에서 실행한다. 비밀키는 이 PC에서 무작위로 만들어 바로 등록하고, 임시 파일은 지운다.

```powershell
$b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
$f = Join-Path $env:TEMP "turn_secret.txt"; [IO.File]::WriteAllText($f, (($b | ForEach-Object { $_.ToString("x2") }) -join ""))
firebase functions:secrets:set TURN_SHARED_SECRET --project YOUR_PROJECT_ID --data-file $f
Remove-Item $f
```

## 2. Google Cloud 콘솔 설정 (본인 프로젝트 `YOUR_PROJECT_ID`)

1. **Compute Engine API 사용 설정**: Compute Engine 메뉴에 처음 들어가면 나오는 "사용 설정"을 누른다.
2. **고정 외부 IP 예약**: VPC 네트워크 → IP 주소 → 외부 고정 IP 주소 예약.
   - 이름 `rail-turn-ip`, 네트워크 서비스 등급 프리미엄, IPv4, 리전 `asia-northeast3`
3. **방화벽 규칙**: VPC 네트워크 → 방화벽 → 방화벽 규칙 만들기.
   - 이름 `allow-rail-turn`, 방향 수신, 대상 태그 `rail-turn`
   - 소스 IPv4 범위 `0.0.0.0/0`
   - 프로토콜·포트: TCP `3478`, UDP `3478,49160-49200`
4. **비밀키 읽기 권한**: 보안 → Secret Manager → `TURN_SHARED_SECRET` → 권한 → 액세스 권한 부여.
   - 주 구성원 `<PROJECT_NUMBER>-compute@developer.gserviceaccount.com`(기본 Compute 서비스 계정)
   - 역할 `Secret Manager 보안 비밀 접근자`
5. **VM 만들기**: Compute Engine → VM 인스턴스 → 인스턴스 만들기.
   - 이름 `rail-turn`, 리전 `asia-northeast3`, 영역 `asia-northeast3-a`
   - 머신 유형 `e2-micro`
   - 부팅 디스크: Debian 12, 표준 영구 디스크 10GB
   - 보안 → 액세스 범위: "모든 Cloud API에 대한 전체 액세스 허용"
   - 네트워킹: 네트워크 태그 `rail-turn`, 외부 IPv4 주소 `rail-turn-ip`
   - 고급 → 자동화 → 시작 스크립트: `coturn-startup.sh` 내용을 그대로 붙여 넣는다.
6. **확인**: VM 세부정보 → 로그(직렬 포트 1)에서 `coturn ready on <IP>:3478`이 보이면 된다.

## 3. 함수와 웹에 연결

- 예약한 고정 IP를 `functions/.env`에 `TURN_HOST=<IP>`로 넣는다. 비밀값이 아니며, `.env*`는 git에 올라가지 않는다.
- 함수를 배포한다: `firebase deploy --only functions --project YOUR_PROJECT_ID`
- 그다음 Hosting을 배포한다.

## 운영

- 비밀키 교체: 1번을 다시 실행한 뒤 VM을 재시작한다. 시작 스크립트가 새 값을 읽는다. 함수는 다음 호출부터 새 값을 쓴다.
- 쓰지 않는 기간에는 VM을 중지하면 VM 요금이 멈춘다. 예약 IP·디스크 요금은 남는다.
- coturn 로그: VM에서 `sudo journalctl -u coturn -n 100`.
