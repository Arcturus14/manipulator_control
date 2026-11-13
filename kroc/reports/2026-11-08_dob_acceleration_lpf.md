# Post-DOB 차분 가속도 LPF 비교

## 구현

측정 속도의 차분으로 계산한 가속도에 causal 1차 LPF를 추가했다.

\[
\ddot q_k^{raw}=\frac{\dot q_k^m-\dot q_{k-1}^m}{\Delta t},\qquad
\ddot q_k^f=\beta\ddot q_{k-1}^f+(1-\beta)\ddot q_k^{raw}
\]

\[
\beta=\exp(-2\pi f_c\Delta t)
\]

필터는 DOB의 inverse-dynamics 계산에만 사용한다. MPPI는 기존과 같이 측정
`q`, `dq`를 직접 입력받는다. `dob_acceleration_lpf_cutoff_hz=0`이면 기존
무필터 동작을 정확히 유지한다. 동적 상태 마진은 모든 비교에서 껐다.

## 실험

- GPU: GTX 1650, CUDA 필수
- `dt=0.02 s`, `DOB_ALPHA=40`
- noise: `sigma_q=1e-3 rad`, `sigma_dq=1e-2 rad/s`
- cutoff: raw, 5, 10, 15 Hz
- scenario: nominal, 4x step disturbance, 40% torque limit
- seed: 7, 17, 27
- 총 45회, audit 45/45 통과

2 Hz는 사전 단일-seed 시험에서 위상 지연으로 폐루프가 불안정해 본 비교에서
제외했다.

## 평균 결과

| 조건 | LPF | RMSE [mm] | torque-rate [Nm/s] | DOB RMSE [Nm] | saturation step |
|---|---:|---:|---:|---:|---:|
| nominal | raw | 3.698 | 360.225 | 2.516 | 0.0 |
| nominal | 5 Hz | 3.803 | 267.534 | 2.826 | 0.0 |
| nominal | 10 Hz | 3.760 | 287.951 | 2.011 | 0.0 |
| nominal | 15 Hz | 3.814 | 313.562 | 2.069 | 0.0 |
| stress | raw | 4.061 | 380.714 | 2.567 | 0.0 |
| stress | 5 Hz | 4.424 | 292.752 | 3.454 | 0.0 |
| stress | 10 Hz | 4.135 | 308.289 | 2.326 | 0.0 |
| stress | 15 Hz | 4.194 | 328.958 | 2.228 | 0.0 |
| limited | raw | 6.941 | 334.652 | 2.547 | 72.3 |
| limited | 5 Hz | 5.665 | 264.304 | 2.858 | 86.7 |
| limited | 10 Hz | 4.908 | 273.585 | 2.036 | 69.0 |
| limited | 15 Hz | 4.901 | 301.202 | 2.141 | 64.3 |

상태 위치·속도 위반은 모든 45회 실행에서 0이었다.

## 결론

10 Hz가 현재 가장 균형 잡힌 후보이다. raw 대비 nominal에서 torque-rate와
DOB RMSE를 각각 약 20% 줄였다. limited에서는 tracking RMSE를 약 29%,
torque-rate를 약 18% 줄였다. stress tracking RMSE는 약 1.8% 증가했지만
DOB RMSE와 torque-rate는 감소했다.

5 Hz는 torque smoothing은 강하지만 위상 지연 때문에 DOB RMSE와 limited 포화를
악화시켰다. 15 Hz는 limited 포화가 가장 적지만 torque smoothing이 10 Hz보다
약하다. 따라서 10 Hz를 확정 기본값이 아닌 다음 `DOB_ALPHA` sweep의 중심 후보로
사용한다.

## 다음 단계

1. cutoff 10 Hz를 고정하고 `DOB_ALPHA = 10, 20, 30, 40` 비교
2. cutoff 8, 10, 12, 15 Hz의 좁은 범위 확인
3. 외란 step 직후 rise time과 peak error를 명시적으로 기록
4. 실제 센서 로그 확보 후 noise 모델과 cutoff 재설정
5. 위 절차가 끝난 후 동적 상태 마진 재활성화

## 산출물

- 결과: `kroc/results/postdob_sensor_noise_20261108_160104_979051/`
- 구현: `observers/dob.py`
- benchmark: `kroc/benchmark_postdob_sensor_noise.py`
- audit: `kroc/audit_postdob_sensor_noise.py`
- tests: `tests/test_dob_acceleration_filter.py`
