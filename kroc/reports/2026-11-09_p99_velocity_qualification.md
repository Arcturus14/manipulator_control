# P99 deadband 환경의 동적 속도 마진 다중-seed 검증

## 결론

P99 soft deadband를 적용한 noise-aware Post-DOB 환경에서 동적 상태 마진의
효과가 재현되는 velocity recovery 시나리오를 선정했다. 탐색에 사용하지 않은
5개 seed(47, 57, 67, 77, 87)에서 fixed margin과 directional margin을 paired
비교한 결과, directional margin은 모든 seed에서 peak velocity를 낮췄고 hard
velocity violation을 3회에서 0회로 줄였다. 사전에 정한 수용 기준을 모두
만족했으므로 **directional dynamic margin은 현재 시나리오에서 qualification
통과**로 판단한다.

반면 headroom gain은 전체 실험에서 1.0에 머물러 directional과 완전히 같은
결과를 냈다. 따라서 **headroom의 추가 효과는 아직 qualification되지 않았다.**

## 일반적인 Post-DOB 대비 현재 논문 구조의 변경점

1. Plant 적분과 제약 평가는 true state를 사용하고, MPPI·DOB·동적 마진은
   sensor noise가 포함된 measured state를 사용한다.
2. 측정 가속도는 10 Hz LPF를 통과하고, DOB는 exact-ZOH 8 Hz 후보를 사용한다.
3. Nominal noise로 생기는 `a_dhat`를 외란으로 오인하지 않도록 관절별 P99 soft
   deadband를 적용한다.
4. DOB 추정값은 rollout의 Post-DOB 보상뿐 아니라 상태 제약 비용의 방향성
   마진 생성에도 사용한다.
5. 속도 hard clamp 전 raw state를 별도로 기록해 clamp가 위반을 숨기지 않도록
   했다.
6. Fixed와 adaptive 방식 모두 최소 마진을 상태 범위의 5%로 통일해 baseline
   이점을 제거했다.

동적 마진 경로는 다음과 같다.

```text
d_hat -> a_raw = -M(q)^(-1) d_hat
      -> joint-wise P99 soft deadband
      -> disturbance-direction state margin
      -> MPPI state-constraint cost
```

Torque 기준은 `U_center = U_ff + R`이며 `d_hat`를 다시 더하지 않는다. 이로써
이미 `R`에 포함될 수 있는 보상 입력과 외란 추정값의 이중 계산을 피한다.

## 고정한 실험 환경

- UR5 wrist-3 joint: index 5
- position sensor noise: 0.001 rad
- velocity sensor noise: 0.01 rad/s
- acceleration LPF: 10 Hz
- exact-ZOH DOB cutoff: 8 Hz
- Post-DOB output compensation
- fixed margin: 상태 범위의 5%
- directional/headroom margin: 최소 5%, 최대 10%
- reaction time: 0.10 s
- controller period: 20 ms
- P99 acceleration deadband [rad/s^2]:
  `[0.94216007, 0.87384290, 1.19243760, 1.44695040, 1.38893270, 2.95120070]`

P99 후보는 nominal calibration seed 17/27/37로 만들고 독립 seed 47/57/67에서
false activation을 확인했다. 가속도 채널 activation은 99.17%에서 0.65%로,
속도 마진 평균 초과량은 0.06278에서 0.00034 rad/s로 감소했으며 26 Nm ramp는
세 seed 모두 20 ms에 검출됐다.

## 시나리오 재선정 과정

기존의 zero-initial-velocity impulse/ramp는 첫 속도 peak가 DOB 추정 및 마진
갱신보다 먼저 발생해 반응형 soft margin의 효과를 공정하게 평가하지 못했다.
또한 일부 seed는 위험구간에 진입하지 않아 결과가 재현되지 않았다.

따라서 이번에는 한계 방향으로 이미 움직이는 관절이 외란을 받았을 때 MPPI가
제동할 수 있는지를 평가하는 controlled velocity-recovery 시나리오를 사용했다.

- initial wrist-3 velocity: +2.5 rad/s
- applied disturbance: -16 Nm
- disturbance ramp: 5 control steps (100 ms)
- hold: 5 control steps (100 ms)
- onset: step 0
- simulation length: 100 steps

현재 plant 식은 `tau_effective = tau_applied - disturbance`이므로 음의 외란이
양의 관절 가속도를 만든다. 이 부호 규약을 확인한 뒤, 양의 초기속도를 상한
방향으로 더 미는 -16 Nm를 선택했다. 이 시나리오는 impulse와 달리 DOB 및 마진이
반응할 시간을 주면서도 fixed 방식이 속도 위험구간에 진입하도록 한다.

탐색 seed 7/17/27/37에서 후보를 선정한 뒤, 최종 평가는 탐색에 사용하지 않은
seed 47/57/67/77/87로 수행했다.

## 다중-seed 최종 결과

| Mode | mean peak dq6 [rad/s] | soft-zone steps | hard violations | soft excess integral [rad] | tracking RMSE [mm] | torque-rate RMS [Nm/s] | mean solve [ms] |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 3.1045 | 19 | 3 | 0.070400 | 1.823 | 294.20 | 14.15 |
| directional | 2.8630 | 10 | 0 | 0.008405 | 2.033 | 295.89 | 14.37 |
| headroom | 2.8630 | 10 | 0 | 0.008405 | 2.033 | 295.89 | 14.37 |

Directional 대비 fixed의 변화는 다음과 같다.

- mean peak velocity: 0.2416 rad/s 감소(-7.78%)
- soft-limit excess integral: 88.1% 감소
- soft-zone 체류: 19 step에서 10 step으로 감소(-47.4%)
- hard violation: 3회에서 0회로 감소
- tracking RMSE: 11.52% 증가
- torque-rate RMS: 0.57% 증가
- mean solve time: 1.56% 증가
- torque saturation: 두 방식 모두 0회

Seed별 peak 감소량은 47: 0.0720, 57: 0.2983, 67: 0.2625, 77: 0.2128,
87: 0.3623 rad/s였다. 모든 독립 seed에서 감소 방향이 일치했다.

## 수용 기준과 판정

사전 기준은 (1) 모든 seed에서 peak velocity 감소, (2) excess integral 50% 이상
감소, (3) hard violation 비증가, (4) tracking RMSE 악화 15% 이내, (5) torque
saturation 비증가였다. Directional margin은 다섯 조건을 모두 만족했다.

따라서 현재 단계의 판정은 다음과 같다.

- **Directional dynamic margin: PASS**
- **Headroom의 distinct effect: FAIL / 별도 torque-limited 시나리오 필요**

## 해석 범위와 한계

이 결과는 P99 deadband와 현재 noise/DOB 설정 아래의 controlled recovery
scenario에서 동적 속도 마진이 fixed margin보다 위험 노출을 줄였다는 증거다.
임의의 trajectory 전반에 대한 우월성이나 formal safety guarantee를 뜻하지
않는다. 또한 wrist-3를 관절별로 해석한 `joint-wise approximation`이며 UR5의
관절 결합 효과를 완전히 분리한 것은 아니다.

Headroom이 동작하지 않은 것은 구현 오류의 증거가 아니라, 현재 시나리오에서
계획 토크가 설정한 unsafe headroom 영역에 들어가지 않아 gain이 1.0을 유지했기
때문이다. Headroom 주장은 torque-limited 조건을 별도로 설계해 검증하기 전에는
논문 결론에 포함하지 않는다.

## 다음 작업

1. 같은 noise/DOB/deadband 설정으로 position-limit 시나리오를 다중 seed 검증한다.
2. Torque 포화에 가까우면서 제동 여유가 부족해지는 headroom 전용 시나리오를
   설계하고 directional과 headroom을 분리한다.
3. P99 deadband, 최대 마진, reaction time에 대한 민감도 분석을 수행한다.
4. 논문용 표와 그림에서는 true/raw constraint metric, measured state, solve time,
   logging을 포함한 full-cycle time을 구분한다.

## 재현 산출물

- 최종 benchmark 결과:
  `kroc/results/noise_aware_state_margin_20261108_235922_275343/`
- qualification 요약:
  `kroc/results/noise_aware_state_margin_20261108_235922_275343/QUALIFICATION.md`
- machine-readable 판정:
  `kroc/results/noise_aware_state_margin_20261108_235922_275343/qualification.json`
- benchmark:
  `kroc/benchmark_noise_aware_state_margin.py`
- scenario search:
  `kroc/search_dynamic_margin_velocity_scenario.py`
- analyzer:
  `kroc/analyze_velocity_qualification.py`
- tests:
  `tests/test_kroc_constraint_scenarios.py`
