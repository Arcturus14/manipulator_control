# 보상 없는 DOB 동적 마진 위치 제약 분리 실험

## 외란 정의 정정

앞선 원 궤적의 `2x step`은 주기 외란이 아니다. 1.25초에 한 번 0에서
`[30, -24, 20, 6, -6, 3] Nm`로 바뀐 뒤 종료까지 유지한 단일 sustained step이다.

이번 위치 제약 분리 실험은 6번 관절 하나에만 다음 외란을 사용했다.

- 0.2초: ramp 시작
- 0.2--0.4초: 0에서 -9.5 Nm까지 선형 증가
- 0.4--2.0초: -9.5 Nm 유지
- plant convention: `tau_effective=tau_applied-d`; 음의 외란이 q6 상한 방향으로 민다.

## 구조

- noisy measured state: `sigma_q=0.001 rad`, `sigma_dq=0.01 rad/s`
- DOB: acceleration LPF 10 Hz, exact-ZOH Q-filter 8 Hz
- `d_hat` 출력 토크 보상: OFF
- `d_hat` rollout 반영: OFF
- `d_hat`은 동적 마진 생성에만 사용
- 정적군: native hinge-squared, full span의 고정 5%
- 동적군: 동일 비용, 5% floor + 방향성 stopping-distance, cap 10%
- q6 초기값: `0.92*qmax = 5.7805 rad`
- q6 물리 상한: `2*pi = 6.2832 rad`
- validation seeds: 17, 27, 37, 47, 57

Audit에서 10/10 CUDA, `estimate_only`, `u_applied==u_mppi`, measured/true state 분리,
finite data를 확인했다.

## 결과

| 지표 | 정적 5% | 동적 5--10% |
|---|---:|---:|
| q6 peak [rad] | 6.4571 ± 0.0442 | 6.2102 ± 0.0200 |
| 물리 상한까지 최소 거리 [rad] | -0.1739 ± 0.0442 | +0.0730 ± 0.0200 |
| 위반 step | 30.6 ± 5.8 | 0.0 ± 0.0 |
| 최대 위반량 [rad] | 0.1739 ± 0.0442 | 0.0 ± 0.0 |
| q6 peak velocity [rad/s] | 2.6078 ± 0.2177 | 1.5709 ± 0.2788 |
| 최대 상한 마진 [% full span] | 5.00 | 7.43 ± 0.22 |
| actuator saturation [step] | 0 | 0 |
| tracking RMSE [mm] | 10.778 ± 3.928 | 11.256 ± 3.731 |
| solve mean [ms] | 17.393 ± 0.046 | 17.623 ± 0.026 |

정적 5%는 5/5 seed에서 물리 위치 상한을 넘었고, 동적 5--10%는 5/5 seed에서
상한 이내였다. 대표 seed 17에서 `d_hat`이 음의 외란을 추정하면서 상한 마진이
5%에서 약 7.37%까지 증가했고, 이후 약 6.5% 수준으로 천천히 감소했다.

두 군 모두 토크 포화는 없었다. 따라서 이 결과는 `입력 포화 방지`가 아니라,
`외란 방향 추정 -> 비용 경계 조기 축소 -> 상한 방향 속도와 위치 peak 감소`를
보이는 결과다. 입력 포화가 원인인 별도 주장을 하려면 torque-limited 시나리오가
추가로 필요하다.

이 시나리오는 분리 경계를 찾기 위해 탐색한 좁은 조건이며 일반적인 안전 보장은
아니다. 또한 초기 q6는 이미 native 5% soft zone 내부에 있으므로, 논문에서는
`near-limit targeted stress case`라고 명시해야 한다.

## 산출물

- final data/plot: `kroc/results/estimator_only_position_validation_amp9p5_20261111/`
- benchmark: `kroc/benchmark_estimator_only_position_margin.py`
- coarse search: `kroc/results/estimator_only_position_search_20261111/`
- fine search: `kroc/results/estimator_only_position_search_fine_20261111/`
- boundary search: `kroc/results/estimator_only_position_search_boundary_20261111/`
