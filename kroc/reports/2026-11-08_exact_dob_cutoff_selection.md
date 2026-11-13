# 센서 노이즈와 정확 이산 DOB cutoff 비교

## 버전 분리

기존 무노이즈 controller와 Euler DOB는 삭제하거나 변경하지 않았다.

- `dob_cutoff_hz=None`: 기존 `DOB_ALPHA=40` Euler 갱신
- `dob_cutoff_hz>0`: 실험용 exact-ZOH 1차 DOB 필터
- 센서 노이즈: `SensorNoiseTracker`에서만 활성화
- 가속도 LPF: 실험 옵션으로만 10 Hz 활성화

따라서 일반 실행은 기존 경로를 유지하며, 새 구성이 검증되기 전에는 자동으로
적용되지 않는다.

## 필터

요청한 연속시간 cutoff `f_c`로부터 정확한 pole retention을 계산했다.

\[
\beta=\exp(-2\pi f_c\Delta t),\qquad
\hat d_k=\beta\hat d_{k-1}+(1-\beta)r_k
\]

이는 연속 1차 필터의 time constant를 ZOH로 정확히 이산화한다. `dt=0.02 s`에서
설계 cutoff와 이산 필터 자체의 실제 -3 dB 지점은 다음과 같다.

| 설계 cutoff [Hz] | beta | 실제 digital -3 dB [Hz] |
|---:|---:|---:|
| 6 | 0.470489 | 6.307 |
| 8 | 0.365931 | 8.780 |
| 10 | 0.284610 | 11.696 |
| 12 | 0.221360 | 15.511 |

따라서 최종 `dob_cutoff_hz=8`은 실제 digital -3 dB 약 8.78 Hz에 해당한다.
앞단의 acceleration LPF 설계 cutoff는 10 Hz이다.

## 실험 구성

- GPU: GTX 1650, CUDA only
- seeds: 7, 17, 27
- 총 90회
- candidate DOB cutoff: 6, 8, 10, 12 Hz
- sensor noise: `sigma_q=1e-3 rad`, `sigma_dq=1e-2 rad/s`
- acceleration LPF: 10 Hz
- dynamic state margin: OFF
- state clamp: OFF

외란은 다음처럼 구성했다.

1. nominal: 외란 없음
2. ramp: 0.5초 동안 1배 외란까지 증가
3. sinusoidal: 기존 1배 저주파 정현파 외란
4. impulse: 1배 외란을 0.1초 인가
5. strong step: 기존 4배 step 외란
6. limited: 2배 step과 40% torque limit

## 사전 선정 기준

각 disturbed scenario의 후보 중 최저 RMSE를 1.0으로 놓았다.

- 5개 disturbed scenario 평균 열화 5% 이하
- 어떤 단일 scenario도 열화 10% 이하
- 위치·속도 상태 위반 0
- 통과 후보 중 nominal `d_hat` RMS와 torque-rate RMS의 동일 가중 정규화
  점수가 가장 작은 cutoff 선택

## 평균 결과

| Scenario | 6 Hz RMSE | 8 Hz RMSE | 10 Hz RMSE | 12 Hz RMSE |
|---|---:|---:|---:|---:|
| nominal | 3.740 | 3.713 | 3.700 | 3.631 |
| ramp | 3.732 | 3.736 | 3.745 | 3.663 |
| sinusoidal | 4.771 | 4.492 | 4.470 | 4.324 |
| impulse | 3.792 | 3.813 | 3.769 | 3.675 |
| strong step | 4.853 | 4.418 | 4.347 | 4.126 |
| limited | 4.439 | 4.580 | 6.608 | 5.967 |

Nominal noise 관련 값은 다음과 같다.

| cutoff | 가짜 `d_hat` RMS [Nm] | torque-rate RMS [Nm/s] |
|---:|---:|---:|
| 6 Hz | 1.282 | 273.155 |
| 8 Hz | 1.552 | 284.932 |
| 10 Hz | 1.810 | 285.772 |
| 12 Hz | 1.939 | 284.328 |

## 선정 결과

| cutoff | 평균 tracking 열화 | 최악 열화 | 기준 통과 |
|---:|---:|---:|---|
| 6 Hz | 6.60% | 17.63% | 실패 |
| 8 Hz | 3.98% | 7.07% | **통과** |
| 10 Hz | 12.47% | 48.85% | 실패 |
| 12 Hz | 6.88% | 34.42% | 실패 |

**최종 실험 후보는 exact-ZOH `dob_cutoff_hz=8 Hz`이다.** 6 Hz는 noise 억제가
더 좋지만 sinusoidal과 strong-step 응답이 느리다. 10/12 Hz는 빠른 외란에는
유리하지만 torque-limited 조건의 RMSE가 크게 증가한다. 8 Hz만 사전 tracking
기준과 상태 위반 0을 동시에 만족했다.

기존 clean/Euler 기준과 비교하면 8 Hz noisy 조건은 nominal RMSE가
3.592 -> 3.713 mm로 약 3.4% 증가했고, strong-step은 4.250 -> 4.418 mm로
약 4.0% 증가했다. limited는 5.145 -> 4.580 mm로 감소했다.

## 다음 단계

사용자가 제외한 실기 로그 기반 noise 추정과 별도 상태 관측기는 진행하지 않았다.
이제 주목적인 동적 상태 마진 실험에서는 다음 두 군을 분리 비교한다.

```text
legacy control:
  sensor noise OFF
  acceleration LPF OFF
  legacy Euler DOB_ALPHA=40

noise-aware candidate:
  sensor noise ON
  acceleration LPF=10 Hz
  exact-ZOH DOB cutoff=8 Hz
```

동적 마진에는 raw residual이 아니라 검증된 `d_hat`을 사용하되, 마진 자체의
필터와 deadband는 별도 ablation으로 다룬다.

## 산출물

- 결과: `kroc/results/exact_dob_cutoff_20261108_204933_249095/`
- 자동 요약: 위 결과 디렉터리의 `REPORT.md`
- 선정 근거: 위 결과 디렉터리의 `selection.json`
- aggregate: 위 결과 디렉터리의 `aggregate.json`
- audit: 위 결과 디렉터리의 `audit.json`
