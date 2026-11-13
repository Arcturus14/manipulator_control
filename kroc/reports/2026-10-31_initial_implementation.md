# KRoC 학부생 논문: DOB 기반 동적 상태 margin 1차 구현과 CUDA pilot

작성일: 2026-10-31 (Asia/Seoul)

## 완료 범위

사용자 확정안에 따라 Model-only MPPI의 동적 상태 margin을 구현했다.
`bar_tau = U_ff + R`이며 headroom에 d_hat을 재가산하지 않는다.
Headroom은 unclamped horizon center에서 시점별 reserve의 최소값을 사용한다.
상한 margin은 negative headroom, 하한 margin은 positive headroom으로 확대한다.
M(q) 변환은 full coupled mass matrix를 사용하지만 headroom-제동 부호 대응은
joint-wise approximation이다. 실행 가능한 braking input을 보장하지 않는다.

Clip, 최소 내부 구간 보호, margin용 d_hat clip, 빠른 증가/느린 감소 필터,
running/terminal 비대칭 penalty, 진단 로그를 추가했다. Tube MPC는 제외했다.
기본 GUI/Main의 제어 구조는 그대로 두고 KrocTracker subclass로 실행한다.
공통 코드 변경은 준비 hook, 실험 tracker 주입, 전체 cycle 시간 기록이다.

새 자료는 모두 kroc/ 아래에 저장했다. 기존 실험 자료는 옮기거나 복사하지 않았다.

## Clamp 확인

현재 dynamics/ur5_dynamics.py의 rollout_effective와 plant_step_effective는
상태 clamp를 사용하지 않는다. Actuator torque만 clamp한다.
검증 테스트에서 effective torque=400, dt=.02의 단순 일정 가속도 모델은
rollout과 plant 모두 속도 8 rad/s를 그대로 반환했다.
이번 27개 NPZ의 qd_raw와 qd는 완전히 동일하며, 실제 위치/속도로 다시 계산한
위반 횟수도 저장된 metrics와 모두 일치했다. 속도 위반이 clamp로 숨겨지지 않는다.
샘플 간 peak는 이 검사로 보장하지 않는다.

## 추가로 정한 초기 수치와 비교 해석

- Tr=.10 s, c_q=c_dq=1.
- adaptive 최소 마진: 전체 물리 범위의 2.5%/side.
- fixed 마진: 전체 물리 범위의 5%/side. 기존 90% soft boundary와 동일하다.
- 이번 pilot 최대 마진: 전체 물리 범위의 10%/side.
- 최대 마진 5/10/15% sweep은 구현했으며 이번 pilot에서는 10%만 실행했다.
- 최소 내부 구간: 전체 물리 범위의 50%.
- r_safe=각 관절 양의 torque 한계의 20%, k_tau=1.
- margin용 d_hat clip=각 관절 torque 한계. Rollout의 원래 추정치에는 미적용.
- beta_up=1, beta_down=.1.
- 비용항의 정규화 분모는 기존 고정 90% 경계의 길이로 유지했다.
  마진을 넓히면서 penalty 기울기가 동시에 약해지는 혼입을 피하기 위한 결정이다.

이 값들은 최적화된 최종 설계값이 아니다. 특히 adaptive 최소 2.5%와 fixed 5%는
다르므로 fixed 대비 차이를 동적 조절만의 효과로 해석할 수 없다.
기본 마진을 모두 5%로 맞춘 통제 비교 명령을 README에 제공했다.
directional과 headroom 비교는 기본 마진, filter, cap을 동일하게 사용한다.

## 실행과 재현 자료

GTX 1650 CUDA에서 3 시나리오 × 3 방법 × seeds 7/17/27 = 27회 실행했다.
각 실행은 원 궤적 250 step/5초, dt=.02, H=10, K=136,
temperature=1, temporal_noise_rho=.8, 상태 제한 가중치 각각 50000이다.
실험 순서는 seed별로 세 방법을 순환했다.

- nominal: 외란 없음, 원래 torque 한계.
- stress: 4배 step 외란, 원래 torque 한계.
- limited: 2배 step 외란, torque 한계를 40%로 낮춘 인위적 derating 조건.
  외란은 step 62/1.24초부터 시작한다. 실제 UR5 사양을 바꾼 실험으로 해석하지 않는다.

원본 및 자동 보고서: [결과 폴더](../results/20261031_220157_697646/REPORT.md).
같은 폴더에 run별 NPZ 27개, manifest.json, metrics.json,
aggregate.json(mean/median/min/max/std), audit.json, PNG 3개가 있다.
manifest에는 실행 당시 source hash, git 상태, CUDA 정보 및 인자를 기록했다.
공통 prefix kroc_의 배열이 solve마다 사용된 margin, clipped d_hat,
가속도, 양·음 headroom, 방향별 gain과 unclamped horizon center다.
d_hat_used는 rollout에 사용한 clipped되지 않은 추정치다.

## 관찰 결과

다음 수치는 seed별 지표의 평균이다. 위반 step은 하나 이상의 관절이 경계를
초과한 제어 샘플 수다. 최대 속도 열은 run별 최대값의 평균이다.

| 조건 | 방법 | RMSE mm | 위치 위반 step | 속도 위반 step | 최대 속도 rad/s | 포화 step |
|---|---|---:|---:|---:|---:|---:|
| nominal | fixed | 3.748 | 0 | 0 | 1.685 | 0 |
| nominal | directional | 3.748 | 0 | 0 | 1.685 | 0 |
| nominal | headroom | 3.748 | 0 | 0 | 1.685 | 0 |
| stress | fixed | 708.141 | 88.000 | 148.000 | 10.352 | 0 |
| stress | directional | 701.033 | 88.000 | 147.667 | 11.954 | 0 |
| stress | headroom | 701.033 | 88.000 | 147.667 | 11.954 | 0 |
| limited | fixed | 20.040 | 0 | 0 | 2.127 | 74.000 |
| limited | directional | 20.351 | 0 | 0 | 2.285 | 70.000 |
| limited | headroom | 20.372 | 0 | 0 | 2.298 | 71.667 |

nominal은 seed별 궤적 성능이 세 방법에서 동일했다.
stress는 큰 위반이 지속되며, dynamic 방식의 최대 속도는 오히려 커졌다.
이 조건에서 headroom gain은 전 구간 1로, directional과 headroom의 궤적이 같다.
즉 headroom 확대의 효력을 검증하는 조건으로 사용할 수 없다.

limited에서는 실제 출력 포화가 발생했고 headroom gain도 활성화됐다.
headroom 방법의 upper/lower gain 활성 비율은 전체 time×joint 기준 평균 약 13.8%다.
다만 모든 방법이 실제 상태 위반 0회여서 위반 감소 효과를 판단할 수 없다.
Tracking RMSE 평균도 fixed보다 개선되지 않았다. 이 pilot으로 우월성이나
일반적인 제약 만족을 주장하지 않는다.

## 계산시간과 검증

margin 계산과 진단 GPU tensor 복사를 포함한 solve p95는 평균 약 17.5~17.9ms다.
여기서 fixed도 동일 진단을 수집하므로 기존 비계측 controller 대비 overhead로
직접 해석하면 안 된다. DOB, plant RK4, CPU 로그까지 포함한 전체 simulation
control cycle p95는 평균 약 29.9~30.4ms이고 모든 run에서 20ms deadline
초과율은 100%다. 현재 결과로 전체 루프의 50Hz 실시간성을 주장할 수 없다.
Compilation warmup과 최초 5개 샘플은 시간 통계에서 제외했다.

관련 unit/regression 테스트 27개가 통과했다. 저장 데이터 audit는 27개 run 모두에서
CUDA 여부, raw 속도 동일성, horizon 최소 headroom, 방향 연결, margin 상·하한,
최소 내부 구간, raw 상태의 위반 집계를 통과했다. 대표 limited 그래프도 확인했다.
첫 pilot 이후 동일 tracker를 다시 실행할 때 margin을 초기화하도록 reset을 보강했고,
reset 테스트를 추가했다. 이번 pilot은 매 run 새 tracker여서 해당 보강의 수치 영향은 없다.

## 다음 실험에 필요한 결정

1. 기본 마진을 맞춘 통제 비교부터 수행하고 5/10/15% cap sweep 결과를 확인한다.
2. 실제 포화와 상태 경계 접근/위반이 함께 발생하는 중간 난이도 조건을 선별한다.
   현재 stress는 큰 실패, limited는 위반 0회여서 둘 사이의 조건이 필요하다.
3. 고정 worst-case margin과 같은 안전수준에서 추종 손실을 비교한다.
   seed tuning/검증 분리와 회복 가능한 braking 여유 검사가 필요하다.
4. 논문 시간 지표는 solve와 전체 cycle을 함께 제시하고, 필요 시 진단/plant 비용을
   별도로 프로파일링한다. hard constraint 만족 또는 하드웨어 실시간 보장은 주장하지 않는다.
