# 동일 위치 외란에서 DOB 구조 × 상태 마진 비교

## 공통 조건

- q6 초기값: `0.92*qmax = 5.7805 rad`
- q6 물리 상한: `2*pi = 6.2832 rad`
- 0.2초부터 -9.5 Nm ramp 시작, 0.4초 도달 후 2.0초까지 유지
- sensor noise: `sigma_q=0.001 rad`, `sigma_dq=0.01 rad/s`
- DOB: acceleration LPF 10 Hz, exact-ZOH Q 8 Hz
- seeds: 17, 27, 37, 47, 57
- CUDA 30 runs

구조:

- `estimate_only`: d_hat 추정만 수행; 출력/rollout 보상 없음
- `post_dob`: MPPI 출력에 d_hat을 더한 후 actuator limit 적용
- `rollout_only`: 출력 보상 없이 MPPI rollout의 effective torque에 d_hat 반영

각 구조에서 native fixed 5%와 directional dynamic 5--10%를 비교했다.

## 결과

| 구조 | 마진 | 위반 step | 최대 위반 [rad] | 상한 최소거리 [rad] | RMSE [mm] | 포화 step | torque-rate [Nm/s] | solve mean [ms] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| estimate only | fixed 5% | 30.6 ± 5.8 | 0.174 ± 0.044 | -0.174 ± 0.044 | 10.778 ± 3.928 | 0 | 340.29 | 17.374 |
| estimate only | dynamic 5--10% | 0 | 0 | +0.073 ± 0.020 | 11.256 ± 3.731 | 0 | 345.09 | 17.580 |
| Post-DOB | fixed 5% | 0 | 0 | +0.504 ± 0.000 | 1.818 ± 0.199 | 0 | 284.68 | 17.320 |
| Post-DOB | dynamic 5--10% | 0 | 0 | +0.504 ± 0.000 | 1.668 ± 0.219 | 0 | 291.54 | 17.571 |
| rollout only | fixed 5% | 0 | 0 | +0.504 ± 0.000 | 3.219 ± 0.782 | 0 | 316.94 | 17.345 |
| rollout only | dynamic 5--10% | 0 | 0 | +0.504 ± 0.000 | 2.911 ± 0.530 | 0 | 321.98 | 17.604 |

`상한 최소거리`는 episode 중 q6 최고값에 대한 값이다. Post-DOB와 rollout-only는
초기값이 episode 최고값이라 두 margin 모두 같은 0.504 rad로 나타난다.

최종 q6 평균은 다음과 같다.

| 구조 | fixed 5% | dynamic 5--10% |
|---|---:|---:|
| estimate only | 6.457 ± 0.044 | 6.210 ± 0.020 |
| Post-DOB | 5.437 ± 0.080 | 5.237 ± 0.102 |
| rollout only | 4.999 ± 0.104 | 4.403 ± 0.044 |

## 해석

1. 보상 없는 구조에서는 dynamic margin 자체의 인과 효과가 분명하다. fixed는 5/5
   seed에서 상한을 넘고 dynamic은 5/5에서 상한 이내였다.
2. Post-DOB와 rollout-only는 이 외란 크기에서 보상 자체가 충분해 fixed 5%도
   위반하지 않았다. 따라서 dynamic margin의 추가적인 `위반 방지` 효과는 식별되지
   않는다.
3. dynamic margin은 Post-DOB와 rollout-only의 RMSE를 각각 약 8.3%, 9.6%
   개선했지만 torque-rate와 연산시간은 소폭 증가했다.
4. rollout-only + dynamic은 q6를 가장 멀리 후퇴시켰다. task-space RMSE는 낮지만
   final q6가 4.403 rad까지 감소하므로, 관절 구성 측면에서는 과도하게 보수적인
   반응일 수 있다.
5. 모든 군에서 actuator saturation은 0이었다. 이 비교는 saturation avoidance를
   검증하는 실험이 아니다.

Audit는 30/30 CUDA, finite data와 각 구조의 command relation을 확인했다.

## 산출물

- result: `kroc/results/position_margin_architectures_20261111/`
- benchmark: `kroc/benchmark_position_margin_architectures.py`
