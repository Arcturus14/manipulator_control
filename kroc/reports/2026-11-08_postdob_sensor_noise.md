# Post-DOB 센서 노이즈 1차 비교

## 목적

이상적인 실제 상태를 직접 사용하는 기존 Post-DOB와 센서 측정 노이즈를 반영한
Post-DOB를 비교했다. 양쪽 모두 KRoC 동적 상태 마진은 사용하지 않았다. 현재
제어기의 기존 90% 고정 soft state-limit cost는 유지했다.

## 상태 분리

물리 플랜트의 `true_state`와 제어기가 받는 `measured_state`를 분리했다.

```text
true_state -> sensor model -> measured_state -> MPPI/Post-DOB
     |
     +-> plant integration and constraint evaluation
```

MPPI와 DOB는 측정 상태만 사용하고, 플랜트 적분과 상태 제약 위반 평가는 실제
상태만 사용한다. DOB는 연속한 두 측정 속도의 차분으로 가속도를 계산한다.
센서 난수는 전용 generator를 사용하므로 MPPI sampling RNG를 소비하지 않는다.

## 노이즈 조건

이번 값은 UR5 제조사 사양이 아니라 1차 민감도 조건이다.

- joint position: 독립 zero-mean Gaussian, `sigma_q = 1e-3 rad`
- joint velocity: 독립 zero-mean Gaussian, `sigma_dq = 1e-2 rad/s`
- bias, colored noise, quantization, latency: 이번 실험에서는 제외

Universal Robots의 공개 UR5 technical specification에는 TCP repeatability
`±0.1 mm`가 제시되어 있지만 joint encoder resolution이나 관절 위치·속도 측정
noise 표준편차는 확인되지 않았다. TCP repeatability를 joint sensor Gaussian
noise로 직접 변환하지 않았다.

공식 자료:
https://www.universal-robots.com/media/1828033/ur5_tech_spec_web_en.pdf

## 실험 조건

- GPU: NVIDIA GeForce GTX 1650
- trajectory: circle, 250 steps, `dt=0.02 s`
- seeds: 7, 17, 27
- controller: Post-DOB output compensation
- MPPI: `H=10`, `K=136`, temperature 1, temporal noise rho 0.8
- dynamic state margin: OFF
- state clamp: OFF
- nominal: disturbance OFF, original torque limits
- stress: 1.24 s부터 4x step disturbance, original torque limits
- limited: 1.24 s부터 2x step disturbance, torque limits 40%

총 18회 CUDA run을 완료했다. Audit는 18/18 run에서 실제/측정 상태 분리,
실현 noise RMS, 동적 마진 비활성화, raw 상태 위반, DOB 오차 및 torque 포화를
재계산해 통과했다. 관련 단위 테스트를 포함한 30개 테스트도 통과했다.

## 평균 결과

| Scenario | 측정 | tracking RMSE [mm] | torque-rate RMS [Nm/s] | DOB RMSE [Nm] | saturation steps | q/dq violation |
|---|---|---:|---:|---:|---:|---:|
| nominal | clean | 3.592 | 255.775 | 0.0004 | 0.0 | 0 / 0 |
| nominal | noisy | 3.698 | 360.225 | 2.516 | 0.0 | 0 / 0 |
| stress | clean | 4.250 | 279.463 | 0.466 | 0.0 | 0 / 0 |
| stress | noisy | 4.061 | 380.714 | 2.567 | 0.0 | 0 / 0 |
| limited | clean | 5.145 | 249.907 | 0.233 | 60.0 | 0 / 0 |
| limited | noisy | 6.941 | 334.652 | 2.547 | 72.3 | 0 / 0 |

## 해석

nominal과 stress에서는 현재 noise 크기로 tracking RMSE가 크게 악화되지는 않았다.
그러나 nominal에서도 존재하지 않는 외란을 DOB가 평균 RMS 약 2.52 Nm로 추정했고,
torque-rate RMS가 약 40.8% 증가했다. stress에서도 DOB 추정 RMSE는 약 0.47 Nm에서
2.57 Nm로 증가하고 torque-rate RMS는 약 36.2% 증가했다.

limited에서는 noise가 실제 성능에도 나타났다. tracking RMSE는 5.15 mm에서
6.94 mm로 약 34.9% 증가했고, saturation step은 60.0에서 72.3으로 증가했다.
상태 제약 위반은 두 조건 모두 없었다.

따라서 기존 이상적 시뮬레이션은 DOB 추정 정확도와 torque smoothness를 과대평가한
것이 맞다. 다만 이 한 noise 크기만으로 실제 UR5 성능을 예측할 수는 없다.

## 다음 단계

동적 상태 마진을 다시 적용하기 전에 다음 순서로 noise 문제를 다룬다.

1. DOB에 입력되는 velocity 또는 acceleration에 causal low-pass filter 적용
2. filter cutoff와 DOB alpha를 함께 sweep해 tracking/noise trade-off 확인
3. direct velocity noise와 encoder-position 차분 velocity 모델을 분리 비교
4. 1/2-step latency, bias, colored noise 추가
5. 제조사 또는 실기 로그에서 관절 측정 noise를 산출하면 그 값으로 재실행

동적 마진은 noisy `d_hat`에 직접 반응하므로 위 절차가 끝나기 전에는 비활성화한다.

## 산출물

- sensor model: `kroc/sensor_noise.py`
- benchmark: `kroc/benchmark_postdob_sensor_noise.py`
- audit: `kroc/audit_postdob_sensor_noise.py`
- raw/aggregate/plots: `kroc/results/postdob_sensor_noise_20261108_033315_108501/`
