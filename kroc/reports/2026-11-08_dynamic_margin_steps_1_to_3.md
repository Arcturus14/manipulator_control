# 동적 상태 마진 개선 1--4단계 경과

## 완료 범위

1. noise-aware Post-DOB와 동적 상태 마진 결합
2. fixed/adaptive 최소 마진을 전체 상태 범위의 5%로 통일
3. 위치·속도 제한 접근 시나리오 추가(효과 검증은 미완료)
4. nominal false activation 정량 진단

3번은 시나리오 코드가 실행된다는 의미의 smoke까지만 통과했다. 동적 마진의
효과가 여러 seed에서 재현된다는 의미의 qualification은 통과하지 못했다.

## 1. Noise-aware 동적 마진 결합

`NoiseAwareKrocTracker`를 추가해 실제 상태와 측정 상태를 분리했다. 제어기, DOB,
동적 마진은 측정 상태를 사용하고 plant 적분과 제약 위반 평가는 실제 상태를
사용한다.

Noise-aware 환경은 앞서 동결한 다음 후보를 사용한다.

- position noise: 0.001 rad
- velocity noise: 0.01 rad/s
- acceleration LPF: 10 Hz
- exact-ZOH DOB: 8 Hz
- compensation: Post-DOB output compensation

Clean continuity 환경은 zero sensor noise와 legacy Euler DOB를 사용한다.
`run_mppi`에는 reference와 일치하는 6관절 초기상태를 명시할 수 있도록
`initial_q` 옵션을 추가했다. 기본값은 기존 IK 초기화이므로 기존 실행 경로는
변하지 않는다.

## 2. 공정한 마진 비교

이전 pilot의 fixed 5%, adaptive minimum 2.5% 차이를 제거했다.

```text
fixed margin        = 5% of full physical span
dynamic minimum     = 5%
dynamic maximum     = 10%
reaction time       = 0.10 s
```

따라서 directional/headroom 방식은 fixed보다 내부 구간을 넓게 쓰는 이점을
갖지 않고, 위험 신호가 있을 때만 5%보다 크게 확장될 수 있다.

## 3. 제한 접근 시나리오

### Position-limit 접근

UR5 elbow joint(index 2)의 reference를 +0.82 pi에서 +0.96 pi까지 smoothstep으로
이동한다. 물리 한계 +pi는 넘지 않지만 fixed 5% soft boundary인 +0.90 pi를
통과한다. 10 Nm의 양의 elbow disturbance를 1/4 지점부터 가한다.

120-step, seed 7 smoke 결과에서 실제 물리 위치 한계까지 최소 거리는
0.330--0.376 rad였고, 내부 soft boundary까지 최소 거리는 0.007--0.062 rad였다.
즉 hard limit을 인위적으로 바꾸지 않고 soft boundary에 실제로 접근했다.
모든 조건에서 위치·속도 위반과 torque saturation은 0이었다.

Clean 조건에서는 fixed RMSE 3.96 mm, directional/headroom 3.38 mm였고,
noisy-filtered 조건에서는 fixed 3.83 mm, directional/headroom 3.93 mm였다.
단일 seed smoke이므로 성능 우열로 해석하지 않는다.

### Velocity-limit 접근

최초 설계는 1.8 Hz joint reference로 속도 경계에 접근했지만 tracking RMSE가
86--106 mm여서 폐기했다. 해당 폴더에는 `DESIGN_REJECTED.md`를 추가했다.

두 번째 설계는 정적인 task reference에서 wrist-3 joint에 30 Nm 외란을 두
control step(40 ms)만 가했다. Wrist 회전은 position-only task에 미치는 영향이
작으므로 속도 제약을 tracking 난이도와 분리했다.

120-step, seed 7 smoke 결과:

| Environment | Mode | RMSE [mm] | max velocity [rad/s] | hard-limit distance [rad/s] | violation |
|---|---|---:|---:|---:|---:|
| clean legacy | fixed | 1.788 | 2.133 | 1.009 | 0 |
| clean legacy | directional | 1.788 | 2.133 | 1.009 | 0 |
| clean legacy | headroom | 1.788 | 2.133 | 1.009 | 0 |
| noisy filtered | fixed | 2.979 | 3.085 | 0.057 | 0 |
| noisy filtered | directional | 2.800 | 3.085 | 0.057 | 0 |
| noisy filtered | headroom | 2.800 | 3.085 | 0.057 | 0 |

수정 시나리오는 tracking RMSE를 3 mm 이내로 유지하면서 noisy-filtered 조건에서
물리 속도 한계 pi rad/s에 0.057 rad/s까지 접근했다. 다만 세 마진 방식의 최대
속도가 같았다. 외란 pulse의 첫 반응은 DOB와 마진 갱신 전에 발생하므로 현재의
반응형 soft margin이 순간 peak를 낮추지 못한 것으로 보인다. 이 결과는 scenario
qualification이며 동적 마진 효과의 증거가 아니다.

새 velocity smoke에서는 headroom gain 최대값이 1이라 directional과 headroom이
동일했다. Headroom 차이는 별도의 torque-limited 조건에서 평가해야 한다.

### 회복 가능한 ramp 시나리오 재탐색

impulse보다 DOB와 마진에 반응시간을 주기 위해 wrist-3 외란을 ramp-and-hold로
바꾸고 amplitude 26, 30, 34, 38 Nm와 ramp 3, 5, 8 step을 탐색했다. Seed 7에서는
26 Nm, 5-step ramp, 5-step hold가 hard violation 없이 다음 차이를 보였다.

- fixed peak: 3.049 rad/s
- directional peak: 3.027 rad/s
- 0.9-limit 초과 step: 3 -> 2
- 초과 면적: 약 33% 감소

그러나 탐색에 쓰지 않은 seed 17, 27, 37에서 재검증한 결과는 일관되지 않았다.

| Seed | fixed peak [rad/s] | directional peak [rad/s] | 결과 |
|---:|---:|---:|---|
| 17 | 2.618 | 2.625 | 위험구간 미진입, 소폭 악화 |
| 27 | 3.125 | 3.037 | 0.089 rad/s 감소 |
| 37 | 3.107 | 3.107 | 동일 |

27--28 Nm, 5/7-step ramp 추가 탐색에서도 seed 17의 wrist-3 속도는 0.9-limit
위험구간에 진입하지 않았다. 따라서 현재 ramp는 유용한 후보이지만 동적 마진
효과를 입증하는 qualified scenario는 아니다.

## 계산시간 참고

마진 갱신과 진단용 GPU clone을 포함한 solve 평균은 약 17.1--17.3 ms였다.
`full_cycle`은 Python/CPU 로그 및 실험 계측까지 포함해 약 30 ms이므로 하드웨어
제어주기와 동일한 값으로 해석하지 않는다. 실시간성 평가는 계측 오버헤드를
분리한 뒤 다시 수행해야 한다.

## 4. Nominal false activation 진단 및 개선

Noisy-filtered Post-DOB, 실제 외란 0, seed 17/27/37에서 directional margin을
측정했다. 5% baseline보다 큰 마진을 activation으로 정의했다.

| 지표 | 3-seed 결과 |
|---|---:|
| 한 채널이라도 활성화된 step 비율 | 100% |
| 전체 q/dq x lower/upper x joint 채널 활성 비율 | 평균 98.77% |
| 위치 마진 최대 초과량 | 평균 0.151 rad |
| 속도 마진 최대 초과량 | 평균 0.309 rad/s |
| nominal `abs(a_dhat)` p95 | 평균 1.350 rad/s^2 |
| nominal `abs(a_dhat)` p99 | 평균 2.192 rad/s^2 |

위 수치는 위치·속도에 의한 의도된 정지거리 마진까지 포함한 최초 진단이다.
Nominal 궤적도 움직이므로 전체 마진 활성률을 모두 false activation으로 부르는
것은 과대평가다. 후속 4단계에서는 `a_dhat` 가속도 성분만 분리해 재평가했다.

Calibration seed 17/27/37의 관절별 nominal p95/p99로 soft acceleration
deadband를 구성하고, 독립 seed 47/57/67에서 검증했다. p99는 가속도 채널
activation을 99.17%에서 0.65%, 속도 마진 평균 초과량을 0.06278에서
0.00034 rad/s로 줄였다. 26 Nm ramp는 세 seed 모두 20 ms에 검출했고 외란 전
wrist 오활성은 0%였다. 상세 결과는
`kroc/reports/2026-11-08_step4_acceleration_deadband.md`에 기록했다.

따라서 4단계 후보는 nominal p99 soft deadband로 결정한다. 이제 이 후보를
고정하고 동일 ramp 계열에서 3번 scenario qualification을 다시 수행해야 한다.

4번을 확인하기 전에는 deadband를 기본 로직으로 확정하지 않는다.

## 산출물

- 구현: `kroc/state_margin.py`, `experiment_runner.py`
- benchmark/scenario: `kroc/benchmark_noise_aware_state_margin.py`
- scenario tests: `tests/test_kroc_constraint_scenarios.py`
- position smoke: `kroc/results/noise_aware_state_margin_20261108_221732_539490/`
- revised velocity smoke: `kroc/results/noise_aware_state_margin_20261108_222034_628498/`
- ramp scenario search: `kroc/results/dynamic_margin_velocity_search_20261108_231357_789713/`
- new-seed ramp validation: `kroc/results/noise_aware_state_margin_20261108_231737_194275/`
- scenario refinement: `kroc/results/dynamic_margin_velocity_search_20261108_231846_573046/`
- nominal false activation: `kroc/results/noise_aware_state_margin_20261108_231935_234990/`
- false activation analyzer: `kroc/analyze_false_activation.py`
