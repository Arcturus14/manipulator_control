# Zero-floor 동적 제약과 정적 soft constraint 비교

## 결론

기존 5% floor를 제거하고 네 비교군을 동일한 smooth soft-limit 비용으로 다시
검증했다. 선정된 입력 포화 시나리오에서 dynamic headroom은 기본 soft constraint
대비 위치 위반을 평균 14.0 step에서 0.6 step으로 줄였다. 외란 전에는 기본 soft와
사실상 같은 0 margin과 상태를 사용했고, 외란 검출 후 margin을 평균 약 6.4%까지
확대했다.

다만 고정 5%와 고정 10%는 모든 seed에서 위반 0이었고, dynamic은 한 seed에서
3 step의 작은 위반이 남았다. Dynamic은 입력 포화를 없앤 것이 아니라 더 일찍
제동 포화를 사용해 상태 위반을 줄였다. 따라서 결과는 핵심 메커니즘의 1차 증거지만
고정 5%보다 우월하다는 증거는 아니다.

## 공통 실험 조건

- controller: noise-aware Post-DOB
- validation seeds: 17, 27, 37, 47, 57
- 시나리오 탐색 seed 7과 validation seed 분리
- joint: wrist-3, 초기 위치 `0.88 q_max = 5.5292 rad`
- disturbance: -30 Nm, onset step 10, 20-step ramp + 15-step hold
- physical torque limit: ±28 Nm
- sensor noise: q 0.001 rad, dq 0.01 rad/s
- DOB: acceleration LPF 10 Hz, exact-ZOH cutoff 8 Hz
- P99 acceleration deadband
- 동일 softplus limit cost: sharpness 8, 기존 5%-span 정규화 길이
- state hard clamp 없음
- headroom: 실제 Post-DOB 명령 경로를 근사하기 위해 `U_center + d_hat` 사용

## 네 주 비교군

| 비교군 | 명시적 margin |
|---|---:|
| basic soft | 0% |
| fixed small | 5% |
| fixed conservative | 10% |
| dynamic headroom | 0~10% |

Directional 0~10%는 headroom multiplier의 추가 효과만 확인하는 보조 ablation으로
실행했으며 주 비교군에는 포함하지 않았다.

## 결과

값은 5개 seed의 mean ± sample standard deviation이다. Upper distance는 양수일수록
물리 상한 안쪽이고 음수이면 위반이다.

| 방식 | upper distance [rad] | hard violation [step] | violation integral [rad·s] | saturation [step] | tracking RMSE [mm] |
|---|---:|---:|---:|---:|---:|
| basic soft 0% | -0.3016 ± 0.3813 | 14.0 ± 10.2 | 0.07855 ± 0.08582 | 19.2 ± 1.3 | 2.982 ± 1.898 |
| fixed 5% | 0.3194 ± 0.0863 | 0 | 0 | 18.4 ± 0.5 | 2.519 ± 0.637 |
| fixed 10% | 0.7540 ± 0.0000 | 0 | 0 | 17.8 ± 0.8 | 2.900 ± 0.649 |
| dynamic 0~10% | 0.1464 ± 0.1704 | 0.6 ± 1.3 | 0.00031 ± 0.00070 | 21.0 ± 0.7 | 10.497 ± 3.894 |

Dynamic과 basic의 paired 차이:

- 상한 거리: +0.4480 rad
- 위반 step: -13.4 step
- 위반 적분: -0.07823 rad·s
- saturation: +1.8 step
- tracking RMSE: +7.515 mm

Basic은 5개 seed 중 4개에서 물리 상한을 넘었다. Dynamic은 5개 중 4개에서 위반
0이었고 seed 37에서만 최대 0.0359 rad, 3 step의 작은 위반이 남았다.

## 비보수성 확인

외란 전 10 step의 upper margin 평균:

| 방식 | 외란 전 margin [rad] | onset 위치 [rad] |
|---|---:|---:|
| basic soft | 0 | 5.4916 |
| fixed 5% | 0.6283 | 5.4282 |
| fixed 10% | 1.2566 | 5.2683 |
| dynamic | 0.0034 | 5.4916 |

Dynamic의 외란 전 작은 0.0034 rad은 측정 속도 노이즈에 의한 것으로 전체 span의
약 0.027%다. 상태는 basic soft와 수치상 동일했다. 즉 기존 5% floor를 제거한 뒤
안전 구간에서는 정적 축소를 사실상 사용하지 않았다.

외란 onset부터 peak 직전까지 dynamic margin 평균은 0.8034 rad로 전체 범위의
약 6.4%였고, episode 마지막에는 평균 0.0158 rad로 감소했다. 같은 구간에서
fixed 5%는 항상 0.6283 rad, fixed 10%는 항상 1.2566 rad였다.

## `tau_required > tau_max`와 시간 순서

Dynamic run에서 실제 Post-DOB requested torque `u_mppi + d_hat`의 한계 초과 최대값은
평균 18.05 Nm였다. wrist 한계가 28 Nm이므로 일부 시점의 requested torque 크기는
약 46 Nm까지 올라갔다. 최소 방향 headroom도 음수가 되었고 headroom gain 최대값은
평균 4.86으로 명확히 활성화됐다.

대표적인 평균 시간 순서는 다음과 같다.

1. disturbance onset: step 10
2. directional margin 0.05 rad 초과: 약 step 13
3. headroom gain 활성: 약 step 24
4. dynamic의 torque saturation 시작: 평균 step 25.8
5. basic의 torque saturation 시작: 평균 step 27.6
6. position peak: 약 step 46

즉 외란 추정으로 margin이 먼저 증가하고, torque reserve가 부족해지면서 headroom이
추가 확대되며, dynamic은 basic보다 약 1.8 step 일찍 최대 제동을 사용했다. 그 결과
포화 횟수 자체는 늘었지만 이후 상태 제약 위반은 크게 줄었다.

따라서 이 실험이 지지하는 문장은 다음과 같다.

> 요구 제동 토크가 actuator 한계를 넘는 ramp 외란에서, zero-floor dynamic margin은
> 외란 전에는 제약을 축소하지 않다가 외란 및 headroom 위험을 검출한 뒤 경계를
> 축소하고 제동을 앞당겨 위치 제약 위반을 감소시켰다.

반대로 다음 문장은 아직 지지되지 않는다.

> Dynamic margin이 actuator saturation 자체를 예방했다.

## Headroom ablation

| 방식 | upper distance [rad] | 위반 step | 최대 margin [rad] |
|---|---:|---:|---:|
| directional only | 0.0711 ± 0.1379 | 1.2 ± 1.8 | 0.7028 ± 0.0340 |
| directional + headroom | 0.1464 ± 0.1704 | 0.6 ± 1.3 | 1.2566 ± 0.0000 |

모든 seed에서 headroom 방식의 upper distance가 directional-only보다 증가했으며 paired
평균 증가는 0.0753 rad였다. 위반 step 차이는 두 seed에서만 나타나 평균 0.6 step
감소했다. 따라서 headroom의 추가 효과는 방향성은 일관되지만 표본 수와 효과 크기상
강한 통계적 결론으로 사용하기에는 아직 이르다.

## 해석과 남은 문제

1. Dynamic은 basic soft보다 제약 안전성을 크게 개선했고, fixed 10%보다 안전 구간의
   제약 축소가 훨씬 작았다. 의도한 중간 성격은 재현됐다.
2. 그러나 이 시나리오에서는 fixed 5%가 위반 0과 낮은 RMSE를 동시에 보여 dynamic보다
   유리했다. Dynamic의 늦고 급격한 회피가 약 10.5 mm의 RMSE를 만들었다.
3. Wrist-3는 end-effector position task에서 일부 중복 자유도를 가지므로 fixed margin의
   보수성을 EE RMSE만으로 평가하면 과소평가된다. 외란 전 joint 이동량과 사용 가능한
   상태 범위를 함께 보고해야 한다.
4. Dynamic이 requested torque를 더 크게 만들고 saturation을 앞당겼다. 다음 개선은
   margin 확대뿐 아니라 torque-feasible braking cost 또는 required-torque deficit을
   직접 비용에 넣어 과도한 불가능 명령을 줄이는 것이다.
5. 다음 검증에서는 elbow/shoulder 또는 orientation tracking을 포함해 고정 마진의
   추종 손실이 실제 task metric에도 드러나는 시나리오가 필요하다.

## 재현 파일

- benchmark: `kroc/benchmark_saturation_aware_constraints.py`
- analyzer/plot: `kroc/analyze_saturation_aware_constraints.py`
- scenario search: `kroc/search_saturation_aware_position_scenario.py`
- result: `kroc/results/saturation_aware_constraints_20261109_031119_001304/`
