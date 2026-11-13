# 센서 노이즈 환경의 단순 방향성 마진: Post-DOB와 Model-only 비교

## 목적

사용자가 제시한 정지거리 기반 위치·속도 마진식만 사용하여 Post-DOB와
Model-only DOB를 비교했다. Torque headroom 증폭은 제외했다.

비교군은 각 DOB 구조마다 다음 세 가지이다.

1. `basic_soft_0pct`: 물리 경계에만 soft constraint, 상태 제약 축소 0%
2. `static_conservative_10pct`: 전체 물리 상태 범위의 10%를 항상 축소
3. `dynamic_directional_0_to_10pct`: 방향성 정지거리 마진, 기본 마진 0%, 최대 10%

## 동적 마진

P99 soft deadband가 적용된 외란 가속도를 `a_tilde`라 하면 다음 식을 사용했다.

```text
m_dq^+ = ReLU(a_tilde) Tr
m_dq^- = ReLU(-a_tilde) Tr

m_q^+ = ReLU(dq) Tr + 0.5 ReLU(a_tilde) Tr^2
m_q^- = ReLU(-dq) Tr + 0.5 ReLU(-a_tilde) Tr^2
```

- `Tr = 0.1 s`
- `m0 = 0`
- 최대 마진: 전체 상태 범위의 10%
- 증가 `beta_up=1.0`, 감소 `beta_down=0.1`
- headroom multiplier: OFF

## 공통 실험 조건

- GPU: NVIDIA GeForce GTX 1650, CUDA 강제
- paired seeds: 17, 27, 37, 47, 57
- 총 60회: 2개 시나리오 × 2개 DOB 구조 × 3개 제약 방식 × 5 seeds
- simulation: 100 steps, MPPI K=136, H=10
- sensor noise: position 0.001 rad, velocity 0.01 rad/s
- DOB: acceleration LPF 10 Hz, exact-ZOH cutoff 8 Hz
- acceleration deadband: nominal joint-wise P99
- 공통 constraint cost: softplus, sharpness 8, weight 50000
- state hard clamp: OFF
- 위치 시나리오: wrist-3를 `0.88 q_max`에서 시작, -30 Nm, onset 10,
  20-step ramp + 15-step hold
- 속도 시나리오: wrist-3 초기속도 +2.5 rad/s, -16 Nm,
  5-step ramp + 5-step hold

## 위치 상한 시나리오

값은 5 seeds의 mean ± sample standard deviation이다.

| 구조 | 제약 방식 | RMSE [mm] | 위치 위반 [step] | 최대 위치 초과 [rad] | 위치 위반 적분 [rad·s] | 속도 위반 [step] |
|---|---|---:|---:|---:|---:|---:|
| Post-DOB | basic soft 0% | 4.273 ± 1.491 | 21.4 ± 3.9 | 0.5811 ± 0.1299 | 0.13975 | 4.4 ± 2.1 |
| Post-DOB | dynamic 0–10% | 4.897 ± 3.046 | 1.2 ± 1.8 | 0.0129 ± 0.0222 | 0.00057 | 0 |
| Post-DOB | static 10% | 2.615 ± 0.394 | 0 | 0 | 0 | 0 |
| Model-only | basic soft 0% | 145.746 ± 80.012 | 68.4 ± 4.4 | 3.1821 ± 0.5469 | 2.49792 | 47.8 ± 8.0 |
| Model-only | dynamic 0–10% | 130.828 ± 51.176 | 65.6 ± 6.2 | 2.9411 ± 0.5609 | 2.20073 | 41.8 ± 10.5 |
| Model-only | static 10% | 105.071 ± 49.320 | 40.0 ± 9.8 | 1.3556 ± 0.3972 | 0.64570 | 17.6 ± 2.0 |

Post-DOB에서 동적 마진은 basic soft 대비 위치 위반 step을 94.4%, 위반 적분을
99.6%, 최대 초과량을 97.8% 줄였다. 5 seeds 중 seed 17/37/47은 위반 0,
seed 27은 2 step, seed 57은 4 step이었다.

고정 10%는 모든 seed에서 위반이 0이었다. 그러나 Post-DOB 동적 방식의 평균 위치
상한 마진은 0.2208 rad, 고정 10%는 항상 1.2566 rad이었다. 동적 방식은 평균적으로
약 82.4% 작은 위치 마진을 사용하면서 거의 모든 위반을 제거했다. 이번 시나리오의
추종 RMSE는 고정 10%가 가장 낮았으므로, 동적 방식이 추종 성능까지 우수하다고
주장할 수는 없다.

Model-only에서는 동적 마진이 basic soft보다 일부 개선됐지만 위치 위반이
65.6 step 남았다. 고정 10%도 40.0 step 위반했다. 현재 Model-only 구조는 추정
외란을 rollout model에는 넣지만 actuator 출력에는 직접 보상하지 않으므로, 이
강한 외란에서 soft constraint만으로 필요한 제동 입력을 만들지 못했다.

## 속도 상한 시나리오

| 구조 | 제약 방식 | RMSE [mm] | 속도 위반 [step] | 최대 속도 초과 [rad/s] | 속도 위반 적분 [rad] |
|---|---|---:|---:|---:|---:|
| Post-DOB | basic soft 0% | 2.010 ± 0.296 | 1.4 ± 1.9 | 0.1444 ± 0.1990 | 0.00635 |
| Post-DOB | dynamic 0–10% | 1.914 ± 0.148 | 0.4 ± 0.9 | 0.0140 ± 0.0313 | 0.00048 |
| Post-DOB | static 10% | 1.753 ± 0.315 | 0 | 0 | 0 |
| Model-only | basic soft 0% | 12.641 ± 5.268 | 8.2 ± 0.4 | 1.5450 ± 0.3495 | 0.17086 |
| Model-only | dynamic 0–10% | 12.373 ± 7.416 | 8.0 ± 0.7 | 1.4610 ± 0.4061 | 0.15907 |
| Model-only | static 10% | 16.062 ± 9.270 | 7.0 ± 0.0 | 1.0429 ± 0.3366 | 0.10375 |

Post-DOB 동적 마진은 basic soft 대비 속도 위반 step을 71.4%, 위반 적분을
92.4%, 최대 초과량을 90.3% 줄였다. 위반은 seed 17에서 2 step만 남았고 나머지
4개 seed에서는 0이었다. 고정 10%는 모든 seed에서 위반이 0이었다.

Model-only에서는 동적 마진의 개선이 작았다. 위반 step은 8.2에서 8.0으로만
감소했고, 고정 10%도 7 step 위반했다. 따라서 이 시나리오에서도 제약 방식보다
외란 보상 구조 차이가 지배적이었다.

## Post-DOB와 Model-only 비교

동일한 동적 방향성 마진에서:

- 위치 시나리오: Post-DOB 1.2 step 대 Model-only 65.6 step
- 속도 시나리오: Post-DOB 0.4 step 대 Model-only 8.0 step
- 위치 RMSE: Post-DOB 4.897 mm 대 Model-only 130.828 mm
- 속도 RMSE: Post-DOB 1.914 mm 대 Model-only 12.373 mm

현재 설정에서는 Post-DOB가 명확히 더 안정적이다. 이는 Model-only 일반의
열등성을 의미하지 않고, 현재의 H=10, K=136, 동일 cost와 별도 재튜닝이 없는
`rollout_model_only` 구현에 대한 결과이다.

## 결론

1. 사용자가 제시한 방향성 마진식은 센서 노이즈 환경의 Post-DOB에서 실제 물리
   위치·속도 위반을 크게 줄였다.
2. 보수적 고정 10%는 가장 확실하게 위반을 제거했지만 항상 최대 수준의 제약
   축소를 적용한다.
3. 동적 방식은 Post-DOB에서 훨씬 작은 평균 마진으로 고정 10%에 가까운 제약
   성능을 냈지만, 소수 seed에서 작은 위반이 남았으므로 hard safety guarantee는
   아니다.
4. 현재 Model-only에서는 세 제약 방식 모두 강한 위치 외란과 속도 pulse를 충분히
   억제하지 못했다. Model-only 비교를 개선하려면 horizon, terminal velocity cost,
   disturbance-aware nominal shift를 별도로 튜닝해야 한다.

## 재현 파일

- benchmark: `kroc/benchmark_simple_directional_architectures.py`
- raw/aggregate/report: `kroc/results/simple_directional_architectures_20261109_042300_261089/`
