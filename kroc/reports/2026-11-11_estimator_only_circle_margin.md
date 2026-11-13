# DOB 필터 변경 및 estimator-only 동적 마진 비교

## 결론

센서 노이즈가 있는 원 궤적에서 기존 DOB와 변경 DOB를 먼저 비교한 뒤, 변경 DOB가
추정한 `d_hat`을 토크 보상이나 rollout 모델에 사용하지 않고 상태 마진 생성에만
사용해 정적 5%와 동적 5--10%를 비교했다.

- 변경 DOB는 무외란 원 궤적에서 기존 noisy legacy 대비 tracking RMSE를 유지하면서
  가짜 `d_hat` RMS와 torque-rate를 낮췄다.
- 2배 step 외란의 estimator-only 비교에서 두 방법 모두 물리 위치/속도 위반과
  토크 포화가 0이었다.
- 동적 마진은 물리 위치 및 속도 한계까지의 최소 거리를 늘렸지만, 추종 RMSE와
  연산시간, torque-rate가 소폭 증가했다.
- 따라서 이 원 궤적 실험은 동적 마진의 **안전 여유 확대와 성능 비용**은 보이지만,
  제약 초과 방지 또는 입력 포화 방지를 증명하지는 않는다.

## 1. DOB 필터 변경

공통 센서 노이즈는 `sigma_q=0.001 rad`, `sigma_dq=0.01 rad/s`이다.

기존 버전:

- 차분 가속도 LPF 없음
- Euler Q-filter, `alpha=40`, `dt=0.02 s`
- 갱신 이득 `g=dt*alpha=0.8`
- 실제 digital -3 dB cutoff 약 17.62 Hz

변경 버전:

- DOB inverse dynamics에만 causal 10 Hz acceleration LPF 적용
- exact-ZOH Q-filter의 설계 cutoff 8 Hz
- `beta=exp(-2*pi*8*0.02)=0.365931`
- 실제 digital -3 dB cutoff 약 8.78 Hz

별도 validation seed 37/47/57의 무외란 원 궤적 평균은 다음과 같다.

| 조건 | tracking RMSE [mm] | torque-rate [Nm/s] | 가짜 d_hat RMS [Nm] | solve mean [ms] |
|---|---:|---:|---:|---:|
| noisy legacy | 3.739 ± 0.052 | 356.103 ± 14.216 | 2.491 ± 0.160 | 15.022 ± 0.016 |
| LPF 10 Hz + exact-ZOH Q 8 Hz | 3.650 ± 0.133 | 283.763 ± 5.250 | 1.557 ± 0.067 | 14.984 ± 0.015 |

변경형은 legacy 대비 tracking RMSE 2.4%, torque-rate 20.3%, 가짜 `d_hat` RMS
37.5%가 감소했다. 무외란 `DOB RMSE`는 실제 외란이 0이므로 가짜 `d_hat` RMS와
같은 의미이다.

## 2. Estimator-only 동적 마진 실험

### 제어 구조

- MPPI 입력: noisy measured `q`, `dq`
- DOB 입력: 같은 noisy measured `q`, `dq`, 실제 actuator applied torque
- DOB: acceleration LPF 10 Hz + exact-ZOH Q 8 Hz
- `d_hat` 출력 보상: OFF
- `d_hat` rollout 모델 반영: OFF
- `d_hat` 사용처: 동적 마진 계산만
- 비교 비용: 동일한 native hinge-squared state-limit cost
- 정적군: full physical span의 고정 5%
- 동적군: 5% floor + 방향성 stopping-distance 항, 최대 10%, P99 deadband

저장 데이터 audit에서 10/10 CUDA 실행, `u_applied == u_mppi`, compensation mode
`estimate_only`, true/measured state 분리, finite data를 확인했다. 실제 동적 위치
마진 최대값은 8.95%였다.

### 외란

5초 원 궤적에서 1.25초부터 2배 step 외란을 계속 인가했다.

`d = [30, -24, 20, 6, -6, 3] Nm`

Plant convention은 `tau_effective=tau_applied-d`이다. 1배 외란에서는 두 군이
동일하게 동작해 마진 효과를 식별할 수 없었고, 4배 외란에서는 보상 없는 두 군이
모두 약 0.6 m RMSE와 대규모 상태 위반으로 붕괴했다. 2배 외란은 폐루프 추종을
유지하면서 동적 마진의 작동 차이를 볼 수 있어 본 비교 강도로 사용했다.

### 5-seed 결과

| 지표 | 정적 5% | 동적 5--10% | 해석 |
|---|---:|---:|---|
| tracking RMSE [mm] | 41.091 ± 1.412 | 43.284 ± 2.268 | 동적 +5.34% |
| 최대 tracking error [mm] | 58.466 ± 2.680 | 69.053 ± 12.864 | 동적 증가 |
| 최소 위치 여유 [rad] | 0.550 ± 0.010 | 0.659 ± 0.029 | 동적 +0.109 rad |
| 최소 속도 여유 [rad/s] | 0.093 ± 0.015 | 0.391 ± 0.018 | 동적 +0.298 rad/s |
| 위치/속도 위반 [step] | 0 / 0 | 0 / 0 | 차이 없음 |
| 토크 포화 [step] | 0 | 0 | 차이 없음 |
| torque RMS [Nm] | 21.353 ± 1.001 | 21.255 ± 1.064 | 거의 동일 |
| torque-rate RMS [Nm/s] | 305.017 ± 6.436 | 311.889 ± 10.981 | 동적 +2.25% |
| solve mean [ms] | 17.414 ± 0.042 | 17.653 ± 0.025 | +0.239 ms |
| solve p95 [ms] | 17.721 ± 0.059 | 17.961 ± 0.066 | +0.241 ms |

동적 마진은 이 시나리오에서 보수적인 위치·속도 여유를 더 확보했지만, 이미 정적
5%도 제약을 위반하지 않았고 actuator utilization도 1 미만이었다. 따라서 현재
결과로 주장할 수 있는 이점은 `constraint clearance 증가`이며, `violation 방지`나
`saturation 방지`는 아니다. 또한 안전 여유 증가의 대가로 tracking과 torque-rate가
악화됐다.

## 산출물

- filter plot: `kroc/figures/2026-11-11_noise_filter_circle_comparison.png`
- margin result/plot: `kroc/results/estimator_only_circle_margin_scale2_20261111_multiseed/`
- invalid 4x stress diagnostic: `kroc/results/estimator_only_circle_margin_20261111_multiseed/`
- benchmark: `kroc/benchmark_estimator_only_circle_margin.py`
- filter plot script: `kroc/plot_noise_filter_circle_comparison.py`
