# DOB alpha 좁은 비교와 최종값

## alpha와 cutoff frequency의 관계

현재 DOB 갱신식은 다음과 같다.

\[
\hat d_k=(1-g)\hat d_{k-1}+g r_k,\qquad g=\Delta t\alpha
\]

연속시간 1차 Q-filter `alpha/(s+alpha)`로 해석하면 근사 cutoff는

\[
f_{c,continuous}=\frac{\alpha}{2\pi}
\]

이다. 그러나 실제 코드는 forward Euler 이산 필터이므로 정확한 디지털
주파수 응답은

\[
H(z)=\frac{g}{1-(1-g)z^{-1}}
\]

이고 -3 dB cutoff는 다음 식으로 계산했다.

\[
\omega_c=\cos^{-1}\left(
\frac{1+(1-g)^2-2g^2}{2(1-g)}
\right),\qquad
f_{c,digital}=\frac{\omega_c}{2\pi\Delta t}
\]

`dt=0.02 s`에서 값은 다음과 같다.

| alpha | `g=dt*alpha` | 연속 근사 cutoff [Hz] | 실제 디지털 cutoff [Hz] |
|---:|---:|---:|---:|
| 32 | 0.64 | 5.093 | 8.953 |
| 35 | 0.70 | 5.570 | 11.033 |
| 38 | 0.76 | 6.048 | 14.129 |
| 40 | 0.80 | 6.366 | 17.621 |

가속도 LPF cutoff 10 Hz와 위 DOB Q-filter cutoff는 서로 다른 필터의 값이다.

## 실험 조건

- acceleration LPF: 10 Hz
- DOB alpha: 32, 35, 38, 40
- noise: `sigma_q=1e-3 rad`, `sigma_dq=1e-2 rad/s`
- scenario: nominal, 4x step disturbance, 40% torque limit
- seeds: 7, 17, 27
- dynamic state margin: OFF
- CUDA 45회, audit 45/45 통과

## 전체 평균

| Scenario | alpha | RMSE [mm] | torque-rate [Nm/s] | DOB RMSE [Nm] | saturation step |
|---|---:|---:|---:|---:|---:|
| nominal | 32 | 3.771 | 282.248 | 1.575 | 0.0 |
| nominal | 35 | 3.696 | 283.344 | 1.729 | 0.0 |
| nominal | 38 | 3.721 | 293.232 | 1.926 | 0.0 |
| nominal | 40 | 3.760 | 287.951 | 2.011 | 0.0 |
| stress | 32 | 4.561 | 286.646 | 2.069 | 0.0 |
| stress | 35 | 4.309 | 299.841 | 2.145 | 0.3 |
| stress | 38 | 4.455 | 304.522 | 2.248 | 0.0 |
| stress | 40 | **4.135** | 308.289 | 2.326 | 0.0 |
| limited | 32 | 7.599 | 263.952 | 1.682 | **61.0** |
| limited | 35 | 5.781 | 271.808 | 1.834 | 68.3 |
| limited | 38 | 6.086 | 287.089 | 1.996 | 66.3 |
| limited | 40 | **4.908** | 273.585 | 2.036 | 69.0 |

## step 직후 응답

| Scenario | alpha | DOB 90% 도달 [ms] | 0.5 s 최대 추종오차 [mm] | 0.5 s RMSE [mm] |
|---|---:|---:|---:|---:|
| stress | 32 | 40 | 15.670 | 8.694 |
| stress | 35 | 20 | 13.641 | 7.350 |
| stress | 38 | 20 | 14.226 | 7.611 |
| stress | 40 | 20 | **11.492** | **6.406** |
| limited | 32 | 40 | 7.181 | 4.087 |
| limited | 35 | 20 | 7.034 | 3.955 |
| limited | 38 | 20 | 7.279 | 3.992 |
| limited | 40 | 20 | **4.954** | **3.150** |

## 최종 결정

현재 실험 범위의 최종값은 **DOB_ALPHA=40**으로 유지한다.

alpha 32는 noise와 torque-rate가 작지만 외란 응답이 느리고 limited tracking이
크게 악화됐다. alpha 35는 nominal/stress 절충이 좋지만 limited RMSE와 외란 직후
peak가 alpha 40보다 크다. alpha 38도 alpha 40보다 tracking 이점이 없었다.

alpha 40은 nominal DOB RMSE가 가장 크다는 단점이 있지만, 10 Hz acceleration
LPF가 이미 무필터 대비 noise 영향을 줄였고, 강한 외란과 torque-limited 조건에서
가장 안정적인 tracking 회복을 보였다. 따라서 논문용 현재 조합은 다음과 같다.

```text
acceleration LPF cutoff = 10 Hz
DOB_ALPHA = 40
dynamic state margin = OFF (noise validation 완료 전)
```

이 결정은 현재의 시험용 Gaussian sensor noise와 step disturbance에 대한 것이며,
실기 sensor log로 noise 모델을 다시 정한 뒤 재검증해야 한다.

## 산출물

- 결과: `kroc/results/postdob_sensor_noise_20261108_163459_945940/`
- 자동 요약: 위 결과 디렉터리의 `REPORT.md`
- aggregate: 위 결과 디렉터리의 `aggregate.json`
- audit: 위 결과 디렉터리의 `audit.json`
