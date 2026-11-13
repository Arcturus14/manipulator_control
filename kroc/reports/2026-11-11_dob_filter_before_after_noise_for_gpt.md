# DOB 필터 구성 비교 검토 자료

## 1. 검토 목적

이 문서는 다음 두 DOB(Disturbance Observer) 구성을 비교하고, 제3자가 설계와
실험의 타당성을 검토할 수 있도록 정리한 자료이다.

1. **노이즈 도입 전 기존 구성**: 가속도 LPF 없이 `DOB_ALPHA=40`인 Euler
   Q-filter 사용
2. **노이즈 도입 후 후보 구성**: 차분 가속도에 10 Hz LPF를 적용하고,
   외란 추정값에는 설계 cutoff 8 Hz의 exact-ZOH Q-filter 사용

여기서 센서 노이즈는 실제 UR5 사양으로 확정한 값이 아니라, 차분 가속도와 DOB의
노이즈 민감도를 확인하기 위한 시험용 조건이다. 두 구성 모두 이 보고서에 수록한
필터 선정 실험에서는 동적 상태 마진을 끄고 비교했다.

---

## 2. 공통 제어 구조와 상태 분리

샘플링 주기는 `dt=0.02 s`, 즉 제어 주파수는 50 Hz이고 Nyquist 주파수는 25 Hz이다.
Post-DOB 구조에서는 MPPI가 만든 토크 명령에 DOB가 추정한 외란 보상량을 반영한다.

노이즈 실험에서는 물리 플랜트의 상태와 제어기에 전달되는 측정 상태를 분리했다.

```text
true_state ──> sensor model ──> measured_state ──> MPPI와 DOB
     │
     └────────────────────────> plant 적분과 실제 상태 제약 판정
```

- MPPI와 DOB는 `measured_state`를 사용한다.
- 플랜트 적분 및 위치·속도 제약 위반 판정은 `true_state`를 사용한다.
- 센서 노이즈용 난수 생성기는 MPPI sampling 난수와 분리했다.
- “MPPI 입력은 그대로 두고 DOB 입력만 필터링한다”는 말은 측정 상태 자체를
  평활화해서 MPPI에 넣는 것이 아니라, DOB 내부에서 속도를 차분해 만든
  **가속도 신호에만 LPF를 적용한다**는 뜻이다.

---

## 3. 노이즈 도입 전: 기존 Euler Q-filter

### 3.1 구조

기존 무노이즈 기준 구성은 다음과 같다.

```text
sensor noise: OFF
acceleration LPF: OFF
Q-filter: forward Euler, DOB_ALPHA=40
dynamic state margin: OFF
```

시뮬레이터의 실제 상태가 측정 상태로 그대로 사용되므로 측정 오차가 없다. DOB는
연속한 두 속도의 차분으로 가속도를 구하지만, 이상적인 상태에서는 센서 노이즈가
차분 과정에서 증폭되지 않는다.

### 3.2 Q-filter 식

DOB의 모델 잔차를 `r_k`, 필터링된 외란 추정값을 `d_hat_k`라고 하면 기존 식은
다음과 같다.

\[
\hat d_k=(1-g)\hat d_{k-1}+g r_k,\qquad g=\Delta t\alpha
\]

`dt=0.02 s`, `alpha=40 rad/s`이므로 `g=0.8`이며,

\[
\hat d_k=0.2\hat d_{k-1}+0.8r_k
\]

가 된다. 새 잔차를 80% 반영하므로 외란 응답은 빠르지만 고주파 성분도 많이
통과시킨다.

연속시간 필터 `alpha/(s+alpha)`로 해석한 근사 cutoff는

\[
f_{c,continuous}=\frac{\alpha}{2\pi}=6.366\ \mathrm{Hz}
\]

이다. 그러나 실제 구현은 forward Euler 이산 필터이므로 실제 주파수 응답은

\[
H(z)=\frac{g}{1-(1-g)z^{-1}}
\]

이고, `alpha=40`, `dt=0.02 s`에서 실제 디지털 −3 dB 주파수는 약
**17.621 Hz**이다. 따라서 6.366 Hz는 현재 코드의 실제 digital cutoff가 아니다.

### 3.3 기존 무노이즈 성능

아래 값은 seed 7, 17, 27의 평균이다.

| 시나리오 | tracking RMSE [mm] | torque-rate RMS [Nm/s] | DOB RMSE [Nm] | saturation steps | q/dq 위반 |
|---|---:|---:|---:|---:|---:|
| nominal, 외란 없음 | 3.592 | 255.775 | 0.0004 | 0.0 | 0 / 0 |
| strong step, 4배 외란 | 4.250 | 279.463 | 0.466 | 0.0 | 0 / 0 |
| limited, 2배 외란·토크 한계 40% | 5.145 | 249.907 | 0.233 | 60.0 | 0 / 0 |

---

## 4. 센서 노이즈 도입 후 발견된 문제

### 4.1 시험용 센서 노이즈

- 관절 위치: 독립 영평균 Gaussian, `sigma_q=1e-3 rad`
- 관절 속도: 독립 영평균 Gaussian, `sigma_dq=1e-2 rad/s`
- bias, colored noise, quantization, latency는 제외

이 값은 제조사 센서 사양이 아니라 민감도 시험용 가정이다.

### 4.2 차분 가속도의 노이즈 증폭

기존 가속도 계산은

\[
a_{raw,k}=\frac{\dot q_{m,k}-\dot q_{m,k-1}}{\Delta t}
\]

이다. 속도 측정 노이즈도 같은 방식으로 차분되므로 작은 오차가 `dt=0.02 s`로
나누어져 크게 나타난다. 이 가속도가 동역학 모델의 관성항에 들어가면, 실제 외란이
없어도 DOB 잔차가 커지고 DOB가 이를 외란으로 추정할 수 있다. 이 보고서에서는 이를
**가짜 외란 추정**이라고 부른다.

### 4.3 필터가 없는 noisy 결과

| 시나리오 | 조건 | RMSE [mm] | torque-rate [Nm/s] | DOB RMSE 또는 nominal `d_hat` RMS [Nm] | saturation steps |
|---|---|---:|---:|---:|---:|
| nominal | clean | 3.592 | 255.775 | 0.0004 | 0.0 |
| nominal | noisy, 필터 없음 | 3.698 | 360.225 | 2.516 | 0.0 |
| strong step | clean | 4.250 | 279.463 | 0.466 | 0.0 |
| strong step | noisy, 필터 없음 | 4.061 | 380.714 | 2.567 | 0.0 |
| limited | clean | 5.145 | 249.907 | 0.233 | 60.0 |
| limited | noisy, 필터 없음 | 6.941 | 334.652 | 2.547 | 72.3 |

nominal에서는 실제 외란이 없는데도 약 2.52 Nm RMS의 외란을 추정했고,
torque-rate RMS가 약 40.8% 증가했다. limited 조건에서는 tracking RMSE가 약
34.9% 증가했다. 이 결과 때문에 Q-filter만 조절하기 전에 차분 가속도 신호를
별도로 필터링할 필요가 생겼다.

---

## 5. 노이즈 도입 후: 가속도 LPF와 exact-ZOH Q-filter

### 5.1 두 필터의 역할

후보 구조에는 역할이 다른 1차 필터 두 개가 직렬로 들어간다.

```text
측정 속도 dq_m
   │ 차분
   ▼
raw acceleration ──> acceleration LPF 10 Hz ──> 모델 잔차 r_k
                                                    │
                                                    ▼
                                      exact-ZOH Q-filter 8 Hz
                                                    │
                                                    ▼
                                                 d_hat
```

- **가속도 LPF**: 속도 차분에서 증폭된 고주파 노이즈가 동역학 잔차를 오염시키는
  것을 줄인다.
- **Q-filter**: 잔차로부터 최종 외란 추정값 `d_hat`을 만들며, 노이즈 억제와
  실제 외란 추종 속도 사이의 절충을 담당한다.
- MPPI가 받는 `q_m`, `dq_m`에는 이 LPF를 적용하지 않았다. 따라서 필터 지연은
  DOB 경로에만 추가된다.

### 5.2 가속도 LPF

\[
a_{f,k}=\beta_a a_{f,k-1}+(1-\beta_a)a_{raw,k}
\]

\[
\beta_a=\exp(-2\pi f_{c,a}\Delta t)
\]

설계 cutoff는 `f_c,a=10 Hz`이다. `dt=0.02 s`에서
`beta_a=exp(-0.4*pi)≈0.28461`이다. 이 필터도 아래 Q-filter와 같은 지수형
이산화를 사용하므로, 필터 자체의 실제 digital −3 dB 주파수는 약 **11.696 Hz**이다.
즉 여기서 “가속도 LPF 10 Hz”는 구현에 입력한 연속시간 설계 cutoff를 뜻한다.

### 5.3 exact-ZOH Q-filter

기존 `alpha` 기반 Euler 식 대신, 양의 `dob_cutoff_hz`가 지정된 실험 경로에서는
다음 식을 사용한다.

\[
\beta_q=\exp(-2\pi f_{c,q}\Delta t)
\]

\[
\hat d_k=\beta_q\hat d_{k-1}+(1-\beta_q)r_k
\]

이는 연속 1차 필터의 pole을 ZOH 방식으로 이산화한 것이다. 최종 후보의 설계
cutoff는 `f_c,q=8 Hz`이고, `beta_q≈0.365931`이다. 이 이산 필터의 실제 digital
−3 dB 주파수는 약 **8.780 Hz**이다.

| Q-filter 설계 cutoff [Hz] | beta_q | 실제 digital −3 dB [Hz] |
|---:|---:|---:|
| 6 | 0.470489 | 6.307 |
| 8 | 0.365931 | 8.780 |
| 10 | 0.284610 | 11.696 |
| 12 | 0.221360 | 15.511 |

주의할 점은 exact-ZOH 식에 넣는 `8 Hz`가 연속 pole을 정하는 설계값이고,
샘플링 후 필터 주파수 응답에서 직접 측정되는 −3 dB 지점은 약 8.78 Hz라는 것이다.

---

## 6. Q-filter cutoff 좁은 비교

### 6.1 실험 조건

- GPU: NVIDIA GeForce GTX 1650, CUDA
- trajectory: circle, 250 steps, `dt=0.02 s`
- seeds: 7, 17, 27
- 총 90 runs, audit 90/90 통과
- MPPI: `H=10`, `K=136`, temperature 1, temporal noise rho 0.8
- 센서 노이즈: `sigma_q=1e-3 rad`, `sigma_dq=1e-2 rad/s`
- 가속도 LPF 설계 cutoff: 10 Hz
- Q-filter 후보: 6, 8, 10, 12 Hz
- dynamic state margin: OFF
- state clamp: OFF

외란 시나리오는 nominal, 0.5초 ramp, 저주파 sinusoidal, 0.1초 impulse,
4배 strong step, 2배 step과 40% torque limit의 limited 조건이다.

### 6.2 Tracking 결과

| 시나리오 | 6 Hz RMSE | 8 Hz RMSE | 10 Hz RMSE | 12 Hz RMSE |
|---|---:|---:|---:|---:|
| nominal | 3.740 | 3.713 | 3.700 | 3.631 |
| ramp | 3.732 | 3.736 | 3.745 | 3.663 |
| sinusoidal | 4.771 | 4.492 | 4.470 | 4.324 |
| impulse | 3.792 | 3.813 | 3.769 | 3.675 |
| strong step | 4.853 | 4.418 | 4.347 | 4.126 |
| limited | 4.439 | 4.580 | 6.608 | 5.967 |

단위는 모두 mm이다.

### 6.3 Nominal 노이즈 지표

| Q-filter cutoff | 가짜 `d_hat` RMS [Nm] | torque-rate RMS [Nm/s] |
|---:|---:|---:|
| 6 Hz | 1.282 | 273.155 |
| 8 Hz | 1.552 | 284.932 |
| 10 Hz | 1.810 | 285.772 |
| 12 Hz | 1.939 | 284.328 |

cutoff가 낮을수록 가짜 외란 추정은 줄지만, 실제 외란에 대한 응답도 느려지는
전형적인 절충이 보인다.

### 6.4 사전 선정 기준과 결과

각 외란 시나리오에서 가장 낮은 후보 RMSE를 기준으로 다음 gate를 사용했다.

- 5개 disturbed scenario의 평균 열화 5% 이하
- 단일 scenario의 최악 열화 10% 이하
- 위치·속도 상태 위반 0
- 통과 후보 중 nominal 가짜 `d_hat` RMS와 torque-rate RMS를 함께 고려

| cutoff | 평균 tracking 열화 | 최악 열화 | gate 통과 |
|---:|---:|---:|---|
| 6 Hz | 6.60% | 17.63% | 실패 |
| 8 Hz | 3.98% | 7.07% | **통과** |
| 10 Hz | 12.47% | 48.85% | 실패 |
| 12 Hz | 6.88% | 34.42% | 실패 |

따라서 현재 시험 범위에서는 설계 cutoff 8 Hz가 유일한 통과 후보였다. 6 Hz는
노이즈 억제가 가장 좋지만 sinusoidal 및 strong-step 추종이 느렸고, 10/12 Hz는
빠른 외란 응답에 유리하지만 torque-limited 조건에서 RMSE가 크게 증가했다.

---

## 7. 최종 두 구성 비교

| 항목 | 노이즈 도입 전 legacy | 노이즈 도입 후 candidate |
|---|---|---|
| 센서 입력 | ideal clean state | noisy measured state |
| 시험 노이즈 | OFF | q 1e-3 rad, dq 1e-2 rad/s |
| MPPI 입력 필터 | 없음 | 없음 |
| DOB 차분 가속도 LPF | 없음 | 설계 10 Hz, 실제 digital −3 dB 약 11.696 Hz |
| Q-filter 구현 | Euler | exact-ZOH |
| Q-filter 파라미터 | alpha=40, g=0.8 | 설계 cutoff=8 Hz, beta=0.365931 |
| 실제 digital −3 dB | 약 17.621 Hz | 약 8.780 Hz |
| 동적 상태 마진 | OFF | OFF |

대표 조건의 tracking 결과는 다음과 같다.

| 시나리오 | legacy clean RMSE [mm] | candidate noisy RMSE [mm] | 변화 |
|---|---:|---:|---:|
| nominal | 3.592 | 3.713 | +3.4% |
| strong step | 4.250 | 4.418 | +4.0% |
| limited | 5.145 | 4.580 | −11.0% |

후보 구성은 노이즈가 존재함에도 clean legacy 대비 nominal과 strong-step RMSE를
약 4% 이내로 유지했고, limited 조건에서는 평균 RMSE가 더 작았다. nominal의
가짜 외란 추정은 필터가 없는 noisy 구성의 2.516 Nm에서 1.552 Nm로 약 38.3%
감소했고, torque-rate도 360.225 Nm/s에서 284.932 Nm/s로 약 20.9% 감소했다.
다만 clean legacy의 nominal `d_hat` RMS 0.0004 Nm 수준까지 제거되지는 않았다.

---

## 8. 해석과 현재 결론

1. 속도 차분으로 가속도를 얻는 현재 DOB에서는 측정 속도 노이즈가 크게 증폭된다.
2. Q-filter만 낮추면 가짜 외란은 줄지만 실제 외란 응답이 느려질 수 있다.
3. 따라서 차분 직후 가속도 LPF로 노이즈 발생 지점을 먼저 처리하고, 그 다음
   Q-filter로 외란 추정 대역폭을 정하는 2단 구성이 합리적인 후보이다.
4. 현재 데이터에서는 `acceleration LPF=10 Hz`, `exact-ZOH Q-filter=8 Hz`가
   노이즈 억제와 외란 추종 사이의 실험적 절충점이다.
5. 기존 무노이즈/Euler 경로는 삭제하거나 덮어쓰지 않았고, `dob_cutoff_hz=None`
   및 가속도 LPF 0 Hz 설정으로 보존되어 있다.

---

## 9. 비교 해석 시 중요한 한계

이 절은 결론을 과대해석하지 않기 위해 반드시 함께 봐야 한다.

1. **최종 표는 순수한 필터 단독 A/B가 아니다.** legacy에는 노이즈가 없고,
   candidate에는 노이즈가 있으며, 동시에 가속도 LPF와 Q-filter 이산화 방식도
   달라졌다. 따라서 limited 성능 개선을 특정 필터 하나의 효과로 단정할 수 없다.
2. 센서 노이즈 크기는 실제 UR5 로그나 encoder 사양에서 식별한 값이 아니다.
3. q와 dq에 독립 Gaussian noise를 직접 더했다. 실제 센서의 상관관계, bias,
   colored spectrum, 양자화, 통신 지연은 반영하지 않았다.
4. cutoff 비교는 circle trajectory와 3개 seed에 기반한다.
5. 선정 gate의 열화율은 clean legacy가 아니라 각 시나리오의 후보 중 최저 RMSE를
   기준으로 계산했다.
6. candidate는 가짜 외란과 torque chatter를 줄였지만 nominal `d_hat` RMS
   1.552 Nm가 남아 있다.
7. 설계 cutoff와 이산 필터의 실제 digital −3 dB 주파수는 동일하지 않다.
8. 동적 상태 마진을 켜면 남아 있는 `d_hat` 노이즈가 마진을 불필요하게 활성화할
   수 있으므로, 이 필터 결과만으로 전체 KRoC 성능이 검증됐다고 볼 수 없다.

더 엄밀한 필터 효과 분리를 위해서는 다음과 같은 factorial ablation이 필요하다.

```text
noise:       clean / noisy
accel LPF:   OFF / 10 Hz
Q-filter:    legacy Euler alpha=40 / exact-ZOH 8 Hz
```

그러나 현재 목적은 실제 센서 모델을 확정하는 것이 아니라, 시험용 노이즈 조건에서
동적 상태 마진 실험에 사용할 수 있는 noise-aware DOB 후보를 정하는 것이다.

---

## 10. GPT에게 검토받고 싶은 질문

아래 질문과 이 문서 전체를 함께 전달하면 된다.

1. 속도 차분 직후의 10 Hz 가속도 LPF와 외란 추정단의 8 Hz Q-filter를 직렬로
   두는 설계가 DOB 구조상 타당한가?
2. `beta=exp(-2*pi*f_c*dt)`를 사용한 exact-ZOH 구현과, 설계 8 Hz가 실제
   digital −3 dB 약 8.78 Hz가 된다는 해석이 맞는가?
3. 50 Hz sampling에서 10 Hz와 8 Hz라는 대역폭이 지나치게 높은지 또는 낮은지,
   판단할 때 어떤 시간응답·주파수응답 지표를 추가로 확인해야 하는가?
4. 제시한 6/8/10/12 Hz 결과와 선정 gate를 기준으로 8 Hz 선택이 충분히
   방어 가능한가? 선정 기준에 구조적인 편향은 없는가?
5. clean legacy와 noisy candidate 비교는 여러 변수가 동시에 달라진다. 위의
   factorial ablation 중 동적 상태 마진 실험 전에 반드시 수행해야 할 최소 비교는
   무엇인가?
6. nominal `d_hat` RMS 1.552 Nm가 동적 상태 마진을 오작동시키지 않게 하려면
   추가 저역통과필터, joint-wise deadband, hysteresis 중 무엇이 가장 적절한가?
7. q와 dq에 독립 Gaussian noise를 직접 주는 모델 대신 encoder position으로부터
   속도와 가속도를 일관되게 생성하는 센서 모델이 필요한가?
8. 제한된 3개 seed와 circle trajectory 결과만으로 cutoff를 고정하기 전에 어떤
   추가 검증이 가장 정보 가치가 높은가?

---

## 11. 재현 및 원자료 위치

- DOB 구현: `observers/dob.py`
- 센서 모델: `kroc/sensor_noise.py`
- noisy benchmark: `kroc/benchmark_postdob_sensor_noise.py`
- noisy audit: `kroc/audit_postdob_sensor_noise.py`
- 최초 noise 비교 보고서: `kroc/reports/2026-11-08_postdob_sensor_noise.md`
- alpha 비교 보고서: `kroc/reports/2026-11-08_dob_alpha_final.md`
- exact cutoff 선정 보고서: `kroc/reports/2026-11-08_exact_dob_cutoff_selection.md`
- 최초 noise 결과: `kroc/results/postdob_sensor_noise_20261108_033315_108501/`
- alpha 비교 결과: `kroc/results/postdob_sensor_noise_20261108_163459_945940/`
- exact cutoff 결과: `kroc/results/exact_dob_cutoff_20261108_204933_249095/`

## 12. 한 문장 요약

이상적 상태에서는 Euler `alpha=40` Q-filter가 빠르게 동작했지만, 시험용 센서
노이즈를 넣자 차분 가속도가 가짜 외란과 토크 떨림을 만들었고, 가속도 LPF 10 Hz와
exact-ZOH Q-filter 설계 8 Hz를 함께 사용한 후보가 현재 시험 범위에서 가장 균형
잡힌 결과를 보였다. 다만 이 결론은 실제 UR5 센서 특성에 대한 확정값이 아니며,
필터 단독 효과를 분리한 완전한 A/B 결과도 아니다.
