# 4단계: nominal false activation 개선

## 목적

Noisy-filtered Post-DOB에서 실제 외란이 없을 때 DOB 추정 잡음이 동적 상태
마진을 계속 확대하는 문제를 줄이되, 실제 외란 검출과 제약 성능은 유지한다.

Nominal 궤적에서도 현재 관절 속도에 따른 정지거리 마진은 의도된 동작이다.
따라서 이번 평가는 전체 마진 활성률이 아니라 `a_dhat`가 만드는 가속도 성분과
속도 마진 초과량을 false activation 지표로 사용했다.

## 설계

관절별 soft deadband를 다음과 같이 적용했다.

```text
a_margin = sign(a_raw) * max(abs(a_raw) - delta_a, 0)
```

Deadband는 margin 생성에만 사용한다. DOB 출력 보상과 rollout plant에는 적용하지
않는다. 기존의 빠른 마진 증가(`beta_up=1`)와 느린 감소(`beta_down=0.1`)도
유지한다.

Calibration은 zero-disturbance noisy-filtered nominal의 seed 17/27/37만 사용했다.

| Profile | joint 1--6 deadband [rad/s^2] |
|---|---|
| p95 | 0.749, 0.644, 0.844, 1.100, 1.057, 2.366 |
| p99 | 0.942, 0.874, 1.192, 1.447, 1.389, 2.951 |

검증은 calibration에 사용하지 않은 seed 47/57/67에서 수행했다. 모든 실행은
CUDA, position noise 0.001 rad, velocity noise 0.01 rad/s, acceleration LPF 10 Hz,
exact-ZOH DOB 8 Hz, Post-DOB output compensation 조건이다.

## Nominal 검증 결과

| Profile | `a_margin` channel activation | 속도 마진 평균 초과 [rad/s] | 속도 마진 최대 초과 [rad/s] | margin variation L1 | RMSE [mm] |
|---|---:|---:|---:|---:|---:|
| none | 99.17% | 0.06278 | 0.28078 | 16.347 | 8.350 |
| p95 | 4.21% | 0.00299 | 0.06675 | 3.295 | 8.350 |
| p99 | 0.65% | 0.00034 | 0.02405 | 2.497 | 8.350 |

p99는 no-deadband 대비 가속도 채널 활성률을 약 99.35%, 속도 마진 평균 초과를
약 99.45%, 전체 margin variation을 약 84.7% 줄였다. Nominal 추종 RMSE는 모든
seed에서 동일했다.

## 실제 외란 보존 결과

검증 시나리오는 정적 task reference, wrist-3 26 Nm, 5-step ramp + 5-step hold다.

| Profile | 외란 전 wrist 활성 | 검출 지연 | peak `a_margin` [rad/s^2] | wrist peak 속도 [rad/s] | hard violation |
|---|---:|---:|---:|---:|---:|
| none | 96.67% | 0 ms* | 88.92 | 2.564 | 0 |
| p95 | 1.11% | 20 ms | 86.55 | 2.564 | 0 |
| p99 | 0% | 20 ms | 85.97 | 2.564 | 0 |

`*` no-deadband의 0 ms는 실제 즉시 검출이 아니라 외란 전에 이미 계속
활성화되어 있었기 때문에 나타난 값이다. p99는 세 seed 모두 외란 다음 제어
주기에 검출했으며 raw peak 가속도 88.92 rad/s^2의 약 96.7%를 보존했다.
세 설정 모두 hard velocity violation은 0이었다. p99의 ramp RMSE 평균은
2.582 mm로 no-deadband의 2.611 mm보다 악화되지 않았다.

## 결정과 한계

현재 고정된 noisy-filtered Post-DOB 환경에는 **관절별 nominal p99 soft
deadband**를 4단계 후보로 채택한다. 다만 전역 기본값은 0으로 유지한다.
Deadband 수치는 센서와 DOB cutoff에 종속되므로 noise/DOB 설정이 바뀌면 다시
calibration해야 하기 때문이다.

이번 결과는 false activation 개선을 확인한 것이며 동적 마진 자체가 peak
속도를 낮춘다는 증거는 아니다. 다음 단계는 p99 deadband를 고정한 상태에서
fixed/directional/headroom이 구분되는 제한 접근 시나리오를 다시 qualification
하는 것이다. 특히 headroom은 현재 ramp에서 활성화되지 않으므로 torque reserve가
실제로 부족해지는 별도 조건이 필요하다.

## 산출물

- 구현: `kroc/state_margin.py`
- benchmark profile: `kroc/benchmark_noise_aware_state_margin.py`
- 분석기: `kroc/analyze_deadband_ablation.py`
- no-deadband: `kroc/results/noise_aware_state_margin_20261108_232828_696226/`
- p95: `kroc/results/noise_aware_state_margin_20261108_232900_698177/`
- p99: `kroc/results/noise_aware_state_margin_20261108_232933_279381/`
- 집계: `kroc/results/deadband_ablation_20261108_233100_verified/`
