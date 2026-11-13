# 10 Hz 가속도 LPF 조건의 DOB alpha 비교

## 목적과 고정 조건

차분 가속도 LPF cutoff를 10 Hz로 고정하고 DOB Q-filter gain인
`DOB_ALPHA`만 비교했다.

- `DOB_ALPHA`: 10, 20, 30, 40
- `dt=0.02 s`
- sensor noise: `sigma_q=1e-3 rad`, `sigma_dq=1e-2 rad/s`
- scenario: nominal, 4x step disturbance, 40% torque limit
- seeds: 7, 17, 27
- dynamic state margin: OFF
- GPU: GTX 1650, CUDA 45회
- audit: 45/45 통과

DOB 갱신식의 새 residual 반영 비율은 `dt * alpha`이므로 각각 0.2, 0.4,
0.6, 0.8이다. alpha가 작을수록 noise는 억제하지만 실제 외란 반응이 느려진다.

## 평균 결과

| Scenario | alpha | RMSE [mm] | torque-rate [Nm/s] | DOB RMSE [Nm] | saturation step |
|---|---:|---:|---:|---:|---:|
| nominal | 10 | 3.813 | 264.067 | 0.467 | 0.0 |
| nominal | 20 | 3.818 | 266.140 | 0.955 | 0.0 |
| nominal | 30 | 3.797 | 273.107 | 1.458 | 0.0 |
| nominal | 40 | 3.760 | 287.951 | 2.011 | 0.0 |
| stress | 10 | 8.882 | 272.899 | 3.194 | 0.0 |
| stress | 20 | 5.675 | 286.843 | 2.212 | 0.0 |
| stress | 30 | 4.603 | 287.351 | 2.045 | 0.0 |
| stress | 40 | 4.135 | 308.289 | 2.326 | 0.0 |
| limited | 10 | 6.769 | 256.380 | 1.644 | 69.0 |
| limited | 20 | 7.428 | 260.958 | 1.370 | 70.0 |
| limited | 30 | 5.558 | 271.027 | 1.612 | 61.0 |
| limited | 40 | 4.908 | 273.585 | 2.036 | 69.0 |

모든 실행에서 위치·속도 상태 제약 위반은 0이었다.

## 해석

- alpha 10은 nominal noise 억제가 가장 강하지만 stress RMSE가 8.882 mm로
  alpha 40보다 114.8% 크다. 강한 step 외란에 너무 느리다.
- alpha 20도 stress와 limited 추종 성능이 부족하다.
- alpha 30은 alpha 40 대비 nominal torque-rate를 5.2%, nominal DOB RMSE를
  27.5% 줄이고 limited 포화를 69회에서 61회로 줄였다. 대신 stress RMSE는
  11.3%, limited RMSE는 13.2% 증가했다.
- alpha 40은 stress/limited tracking RMSE가 가장 작지만 noise-induced torque와
  nominal 가짜 외란 추정은 가장 크다.

## 결론

tracking 성능을 최우선으로 하면 alpha 40을 유지한다. noise robustness와 포화를
함께 고려한 균형 후보는 alpha 30이다. alpha 10과 20은 현재 4x step disturbance
조건에는 느리므로 제외한다.

현재 결과만으로 기본값을 변경하지 않는다. 다음에는 alpha 30과 40 사이인
`32, 35, 38, 40`을 비교하고, step 직후 rise time과 0.1/0.2/0.5초 구간의 tracking
peak를 기록해 최종값을 결정한다.

후속 좁은 비교와 최종 결정은
`kroc/reports/2026-11-08_dob_alpha_final.md`에 기록했다.

## 산출물

- 결과: `kroc/results/postdob_sensor_noise_20261108_162317_336960/`
- 자동 요약: 위 결과 디렉터리의 `REPORT.md`
- aggregate: 위 결과 디렉터리의 `aggregate.json`
- audit: 위 결과 디렉터리의 `audit.json`
