# P99 deadband 환경의 동적 위치 마진 중간 검증

## 결론

위치 상한 방향 외란 시나리오의 부호 오류를 수정한 뒤, noise-aware Post-DOB와
P99 deadband 환경에서 fixed 5%, directional 5--10%, fixed 10%를 독립 5개
seed로 비교했다.

Directional은 fixed 5%보다 elbow 물리 상한까지의 평균 최소거리를
0.3246 rad에서 0.3533 rad로 0.0287 rad(8.84%) 늘렸다. 다섯 seed 중 네 seed에서
거리가 증가했고 한 seed에서는 동일했으며, 감소한 seed는 없었다. 모든 방식에서
hard position violation과 torque saturation은 0이었다.

따라서 현재 결과는 **동적 위치 마진이 상한 접근 위험을 줄이는 방향으로
작동한다는 중간 증거**다. 다만 fixed 5%가 물리 한계를 넘고 directional만
위반을 막는 hard-prevention 결과는 아직 재현되지 않았으므로 그 주장은 하지
않는다.

## 발견하고 수정한 시나리오 오류

Plant 부호는 다음과 같다.

```text
tau_effective = tau_applied - disturbance
```

기존 position-limit 시나리오는 elbow 상한 reference에 `+10 Nm` 외란을 사용했다.
이 부호는 실제로 elbow를 하한 방향으로 미는 외란이므로 의도와 반대였다.
외란을 `-10 Nm`로 수정해 reference 이동과 외란이 모두 상한 방향을 향하게 했다.

## 최종 중간 검증 조건

- Joint: elbow, index 2
- Reference: 0.82π에서 0.96π까지 smoothstep
- Physical upper limit: π
- Disturbance: -10 Nm, step 25부터 지속
- Plant convention: `tau_effective = tau_applied - disturbance`
- Sensor noise: q 0.001 rad, dq 0.01 rad/s
- Acceleration LPF: 10 Hz
- Exact-ZOH DOB: 8 Hz
- P99 acceleration deadband 사용
- Position-limit cost weight: 50,000
- Controller period: 20 ms
- Validation seeds: 27, 37, 47, 57, 67
- 모든 실행 CUDA

## 비교 결과

| Mode | 평균 최소 상한거리 [rad] | Tracking RMSE [mm] | Torque-rate RMS [Nm/s] | Mean solve [ms] | Hard violation | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 5% | 0.3246 | 4.019 | 253.10 | 14.47 | 0 | 0 |
| Directional 5--10% | 0.3533 | 3.876 | 252.13 | 14.61 | 0 | 0 |
| Fixed 10% | 0.5655 | 18.469 | 310.05 | 14.39 | 0 | 0 |

Directional 대비 fixed 5%:

- 평균 물리 상한거리: +0.0287 rad, +8.84%
- Tracking RMSE: -3.58%
- Torque-rate RMS: -0.38%
- Mean solve time: +0.97%
- Hard position violation: 모두 0
- Torque saturation: 모두 0

Seed별 `directional - fixed 5%` 상한거리 차이:

| Seed | 차이 [rad] |
|---:|---:|
| 27 | 0.0000 |
| 37 | +0.0407 |
| 47 | +0.0238 |
| 57 | +0.0367 |
| 67 | +0.0423 |

Fixed 10%는 상한거리를 가장 크게 확보했지만 tracking RMSE가 fixed 5% 대비 약
4.6배로 증가했다. Directional은 fixed 10% 수준의 보수성을 항상 적용하지 않고,
fixed 5%와 거의 같은 추종 성능을 유지하면서 상한거리를 일부 추가 확보했다.
이는 Tube MPC 수준의 강건 보장이 아니라, 사용자가 의도한 **비보수적 방식과
고정 보수적 축소 사이의 중간 지점**이라는 연구 방향과 부합한다.

## 실패한 탐색과 제외 이유

1. Wrist-3 정적 reference: 유효 관성이 작아 fixed 5% 경계 안에서도 충분히
   정지해 물리 상한 위반이 발생하지 않았다.
2. 상한 바로 앞에서 시작: fixed와 directional 모두 즉시 최대 제동해 마진
   차이가 입력 차이로 이어지지 않았다.
3. Wrist-1 초기속도 및 -24/-26 Nm 외란: 일부 단일 seed에서는 fixed만 넘는
   경계가 있었으나 독립 seed에서 재현되지 않았다.
4. Position gain 2.0: torque saturation 중에는 효과가 없어 채택하지 않았다.
5. Elbow -140 Nm: tracking RMSE 33--40 mm와 포화가 발생해 보고용 조건에서
   제외했다.
6. Position weight 500/5,000 sweep: 상한거리 차이는 있었지만 hard-prevention을
   만들지 못했고 기본 50,000보다 논문 설정 근거가 약해 최종 비교에 사용하지
   않았다.

## 현재 판정

- 외란 방향 수정: 완료
- P99 환경에서 directional 위치 마진 활성화: 확인
- Fixed 5% 대비 상한 접근 완화: 5 seed에서 비감소, 평균 8.84% 개선
- Fixed 10% 대비 추종 성능 이점: 확인
- Fixed가 위반하고 directional만 방지하는 hard-prevention: 미확인
- Formal robust constraint guarantee: 연구 목표가 아니며 주장하지 않음

## 다음 개선 방향

Hard-prevention을 추가로 보이려면 외란 세기만 키우는 방식은 부적절하다. 다음
단계에서는 물리 한계 안에 별도의 task/operational position constraint를 설정하고,
그 제한을 controller cost와 평가 양쪽에 동일하게 적용하는 시나리오가 적절하다.
이 방식은 기계적 한계 직전의 과도한 외란을 만들지 않고도 task-level 제약에서
fixed 5%, directional, fixed 10%의 차이를 검증할 수 있다.

## 산출물

- 최종 5-seed 결과:
  `kroc/results/noise_aware_state_margin_20261109_005304_640749/`
- Benchmark:
  `kroc/benchmark_noise_aware_state_margin.py`
- Position scenario search:
  `kroc/search_dynamic_margin_position_scenario.py`
- Scenario tests:
  `tests/test_kroc_constraint_scenarios.py`
