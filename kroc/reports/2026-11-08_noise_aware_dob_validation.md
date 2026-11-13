# Noise-aware Post-DOB 별도 seed 검증

## 목적

기존 cutoff 선정에 사용하지 않은 seed에서 다음 세 조건을 같은 CUDA 실행 묶음으로
직접 비교했다.

1. `clean`: 센서 노이즈 없음, legacy Euler DOB
2. `noisy_legacy`: 센서 노이즈 있음, 가속도 LPF 없음, legacy Euler DOB
3. `noisy_exact_8hz`: 센서 노이즈 있음, 가속도 LPF 10 Hz, exact-ZOH DOB 8 Hz

이 비교는 이전 실험과의 간접 비교 문제를 제거하고, cutoff 선정 seed 7/17/27과
평가 seed가 겹치는 문제를 제거하기 위한 validation이다.

## 조건

- validation seeds: 37, 47, 57
- scenarios: nominal, ramp, sinusoidal, impulse, strong step, torque-limited
- sensor noise: position 0.001 rad, velocity 0.01 rad/s의 독립 Gaussian
- dynamic state margin: OFF
- state clamp: OFF
- GPU: NVIDIA GeForce GTX 1650, CUDA only
- 총 54회

Audit는 54/54 실행에서 CUDA 사용, true/measured state 분리, 실현 noise RMS,
가속도 LPF 재귀식, exact-ZOH DOB 재귀식, 저장 지표 재계산 및 동적 마진 비활성화를
확인해 통과했다. 전체 단위 테스트 42개도 통과했다.

## noisy legacy 대비 필터 후보 평균 변화

| Scenario | Tracking RMSE | Torque-rate RMS | DOB RMSE | saturation steps |
|---|---:|---:|---:|---:|
| nominal | -2.40% | -20.31% | -37.52% | 0 -> 0 |
| ramp | +0.03% | -21.53% | -37.68% | 0 -> 0 |
| sinusoidal | +5.12% | -23.79% | -34.98% | 0 -> 0 |
| impulse | +3.01% | -23.07% | -34.14% | 0 -> 0 |
| strong step | +4.21% | -19.53% | -18.13% | 0 -> 0 |
| limited | -23.01% | -17.69% | -34.23% | 62.33 -> 56.00 |

모든 조건에서 평균 torque-rate와 DOB RMSE가 감소했고 상태 위치·속도 위반은
0이었다. 추종은 nominal과 limited에서 개선되고 ramp는 사실상 동일했지만,
sinusoidal/impulse/strong-step에서는 평균 3--5% 악화됐다.

paired seed 기준 tracking 변화의 최악값은 nominal +0.85%, ramp +15.93%,
sinusoidal +7.65%, impulse +6.28%, strong-step +10.71%, limited +21.34%였다.
Limited 평균은 크게 개선됐지만 seed 37에서는 악화됐으므로 평균 개선만으로
강건성을 주장하지 않는다.

## 판단과 다음 연구 순서

현재 10 Hz acceleration LPF + exact-ZOH 8 Hz DOB를 **동적 마진 실험용으로
동결한 후보**로 사용한다. 추가 cutoff tuning을 같은 합성 Gaussian 환경에서
반복하면 동적 마진 연구보다 observer tuning에 과적합될 위험이 있다.

동적 마진 실험은 다음 두 환경을 분리해 수행한다.

- continuity baseline: clean + legacy Euler DOB
- realistic main candidate: noisy + acceleration LPF 10 Hz + exact-ZOH DOB 8 Hz

`noisy_legacy`는 필터 효과를 분리하는 진단군으로 유지한다. 동적 마진의 첫
factorial 비교가 끝나기 전에는 cutoff나 noise 크기를 다시 조정하지 않는다.

현재 noise가 UR5 사양 기반이 아니므로 colored noise, bias, quantization, latency는
동적 마진의 1차 효과를 확인한 뒤 robustness 실험으로 추가한다. 실기 로그가
확보되면 이 순서를 앞당겨 noise 모델과 DOB 후보를 다시 동결한다.

## 산출물

- raw/aggregate/report: `kroc/results/exact_dob_cutoff_20261108_213049_571177/`
- benchmark: `kroc/benchmark_exact_dob_cutoff.py`
- audit: `kroc/audit_postdob_sensor_noise.py`
