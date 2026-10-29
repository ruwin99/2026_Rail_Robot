#!/bin/bash
# [coturn 시작 스크립트] Compute Engine VM의 "시작 스크립트"로 넣는다. 부팅할 때마다 실행된다.
# - 공유 비밀키는 Secret Manager(TURN_SHARED_SECRET)에서 읽는다. 이 파일·VM 메타데이터에는 없다.
# - 임시 계정은 Cloud Function(turnCredentials)이 같은 비밀키로 계산한다(TURN REST 방식).
# - 사설망·메타데이터 주소로의 중계는 막는다(TURN 서버를 내부망 우회 통로로 쓰지 못하게).
set -euo pipefail

SECRET_NAME="TURN_SHARED_SECRET"
MIN_PORT=49160
MAX_PORT=49200
MD="http://metadata.google.internal/computeMetadata/v1"
md() { curl -fsS -H "Metadata-Flavor: Google" "$MD/$1"; }

if ! command -v turnserver >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y coturn
fi

EXTERNAL_IP="$(md instance/network-interfaces/0/access-configs/0/external-ip)"
INTERNAL_IP="$(md instance/network-interfaces/0/ip)"
SECRET="$(gcloud secrets versions access latest --secret="$SECRET_NAME")"
if [ "${#SECRET}" -lt 32 ]; then
  echo "TURN secret is missing or too short; coturn not started." >&2
  exit 1
fi

umask 077
cat > /etc/turnserver.conf <<EOF
listening-port=3478
listening-ip=${INTERNAL_IP}
relay-ip=${INTERNAL_IP}
external-ip=${EXTERNAL_IP}/${INTERNAL_IP}
min-port=${MIN_PORT}
max-port=${MAX_PORT}
realm=your-firebase-project
use-auth-secret
static-auth-secret=${SECRET}
fingerprint
stale-nonce=600
no-cli
no-tls
no-dtls
no-multicast-peers
no-tcp-relay
user-quota=4
total-quota=20
# 세션당 최대 1 MB/s(약 8 Mbps). 영상 2 Mbps에 여유를 둔 값이다.
max-bps=1000000
denied-peer-ip=0.0.0.0-0.255.255.255
denied-peer-ip=10.0.0.0-10.255.255.255
denied-peer-ip=100.64.0.0-100.127.255.255
denied-peer-ip=127.0.0.0-127.255.255.255
denied-peer-ip=169.254.0.0-169.254.255.255
denied-peer-ip=172.16.0.0-172.31.255.255
denied-peer-ip=192.0.0.0-192.0.0.255
denied-peer-ip=192.168.0.0-192.168.255.255
denied-peer-ip=198.18.0.0-198.19.255.255
denied-peer-ip=224.0.0.0-255.255.255.255
denied-peer-ip=::1
denied-peer-ip=fc00::-fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff
denied-peer-ip=fe80::-febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff
syslog
simple-log
EOF
unset SECRET
chown root:turnserver /etc/turnserver.conf
chmod 640 /etc/turnserver.conf

if [ -f /etc/default/coturn ]; then
  sed -i 's/^#\?TURNSERVER_ENABLED=.*/TURNSERVER_ENABLED=1/' /etc/default/coturn
fi
systemctl enable coturn
systemctl restart coturn
echo "coturn ready on ${EXTERNAL_IP}:3478 (relay ${MIN_PORT}-${MAX_PORT}/udp)"
