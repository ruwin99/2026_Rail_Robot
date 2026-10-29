# 외부 코드 및 자산

이 저장소는 Jetson 애플리케이션, STM32 펌웨어와 웹 화면을 함께 보관합니다. 프로젝트 전체에 새 라이선스를 일괄 지정하지 않았습니다. 아래 외부 코드와 자산의 조건은 각각 적용됩니다.

| 항목 | 위치 / 확인 방법 |
|---|---|
| STM32 HAL·CMSIS | `stm_code/code/encoder/Drivers/`의 라이선스 파일과 각 소스 헤더 |
| ROS 2·Python 의존성 | ROS 패키지의 `package.xml`, `code/scripts/requirement.sh`, 각 패키지 배포 조건 |
| 웹·Firebase 의존성 | `rail_dashboard/package.json`, `package-lock.json`, `functions/package.json` |
| 웹 예시 사진 | `rail_dashboard/public/demo/IMAGE_SOURCES.md`의 출처·이용 조건 |
| AI 모델과 학습 데이터 | 가중치·데이터·TensorRT 엔진은 미포함. 실제 사용하는 모델·가중치·데이터별 조건 확인 필요 |
| 음성 안내 | WAV 파일 미포함. 사용 가능한 음성 파일을 별도로 준비 |

검사 결과와 현장 촬영 데이터는 포함하지 않았습니다. 웹 예시 사진은 화면 확인을 위한 자료이며 로봇의 실제 AI 성능을 나타내는 결과가 아닙니다.
