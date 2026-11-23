# 논문 작성용 연구 정리: 외란 보상으로 인한 입력 포화와 MPPI 입력 제약 처리

이 폴더는 학부 논문을 쓰는 데 필요한 연구 내용, 최종 수치, 그림과 데이터를 한곳에 모은
것이다. 모든 수치는 `data/`의 CSV와 일치하며, 그림과 CSV는 저장소에 추적되는 근거 자료에서
`python -m kroc.make_paper_figures`로 다시 만들 수 있다. 논문에는 이 문서와 `data/`에 있는
값만 쓴다. 수치를 계산하거나 해석할 때는 7절의 주장 범위를 먼저 확인한다.

- 브랜치 `edit/constraint`. 근거 원본: `kroc/evidence/`, 상세 기록: `kroc/README.md` 10장
- 시뮬레이션 전용 연구다. 실제 로봇 실험은 없다.

## 1. 연구 주제

외란 관측기(DOB)로 외란을 보상하는 MPPI 궤적 추종에서, 실제 입력은 다음과 같다.

```
u_req = u_mppi + d_hat,      u_app = clip(u_req, -u_max, u_max)
```

외란 추정값 `d_hat`이 크면 요청 토크가 actuator 한계를 넘어 잘리고(포화), 보상하지 못한
양만큼 추종 성능이 떨어진다. 그런데 기본 MPPI는 이 잘림을 모른 채 명목 입력 `u_mppi`를
계획한다. 본 연구는 이 상황에서 MPPI가 보상 후 포화를 고려하도록 하는 세 방식을 비교한다.

| 방식 | 내용 |
|---|---|
| **N0** | 기본 MPPI. 입력 제약 처리 없음 |
| **N1** | soft constraint. 보상 후 요청 토크가 한계를 넘는 양에 제곱 비용 추가 |
| **D0** | hard constraint. 명목 입력의 허용 범위를 보상값만큼 이동시켜 투영 |

이전 단계(하중 없음, J3 외란)에서는 포화를 만들기 위해 외란이 토크 한계의 약 89%
(133 Nm)나 필요했다. 이는 비현실적이라고 판단해 말단에 5 kg 하중을 달고 빠른 원궤적을
추종하게 하여 명목 입력 `u_mppi` 자체를 키웠다. 그 결과 **토크 한계의 약 26–33%인 J2
외란**에서 포화와 성능 저하가 나타나는 조건을 만들었다.

이전 단계 결과(J3 133 Nm, 하중 없음)는 코드 버전과 조건이 달라 본 결과와 수치를 직접
비교하지 않는다. 동기 설명에만 쓴다.

## 2. 시뮬레이션 설정

| 항목 | 값 |
|---|---|
| 로봇 | UR5 6축, RNEA 기반 강체 동역학, 감속기 관성(armature)·점성 마찰 포함 |
| 토크 한계 | J1–J3 ±150 Nm, J4–J6 ±28 Nm |
| 하중 | 5 kg 정육면체(한 변 0.063 m), flange에 강체 부착, TCP = 정육면체 중심 |
| 기준 궤적 | YZ 평면 원, 반경 0.20 m, 주기 2 s, 중심 (0.717, 0.069, 0.300) m |
| 궤적 시작 | 0–3 s에 속도를 5차 다항식으로 서서히 증가 |
| 초기 자세 | q0 = (3.3857, −0.7568, 0.6252, −1.1786, −1.7568, 5.5920) rad |
| 외란 | J2에 음의 방향 상수 외란. 0.5–2.5 s에 raised-cosine으로 증가 후 유지 |
| 제어 주기 / 시간 | 20 ms (50 Hz) / 16 s |
| plant 적분 | RK4. MPPI 예측은 semi-implicit Euler |
| 센서 노이즈 | 관절 위치 σ = 0.001 rad, 속도 σ = 0.01 rad/s |
| 평가 | TCP 위치 오차 RMSE와 최대 오차, 5 < t ≤ 15 s (정상 상태 5주기) |
| seed | 최종 평가 7207, 7217, …, 7297 (10개). 민감도 분석은 별도 7307/7317/7327 |

외란은 J2에 준다. J2에 하중의 중력과 궤적 관성 부하가 집중되기 때문이다.
외란이 없을 때 J2 요청 토크의 최대 크기는 약 113 Nm(한계의 75%)다.

**유효 실행:** 로봇·하중 mesh 충돌이 없고, 관절 위치·속도 한계를 넘지 않으며, 팔꿈치가
뒤집히지 않아야 한다(전 구간 q3 > 0.2 rad). 하나라도 어기면 "제약 위반(무효)"이다.

## 3. 제어 구조

### 3.1 MPPI

| 항목 | 값 |
|---|---|
| sample 수 K / horizon H | 136 / 10 (0.2 s) |
| 온도 λ | 64 |
| 탐색 noise | 관절별 분산 (9, 9, 9, 0.3136, 0.3136, 0.3136) Nm², horizon 전체에 같은 값 |
| 가중치 | ω_k ∝ exp(−(S_k − min S)/λ) |
| warm start | 기준 궤적의 역동역학 feedforward + 직전 최적 잔차의 shift |
| 실현 방식 | GPU(GTX 1650), float32 |

sample 비용 S_k는 각 rollout stage t(상태 x_{t+1}, 입력 v_t)의 합과 종단 비용이다.

- TCP 추종: 1.2×10⁶ ‖p(x_{t+1}) − p_ref(t+1)‖², 종단 3.6×10⁶
- 자세 유지: 5000 ‖q_{t+1} − q_ref(t+1)‖², 종단 15000
- 입력 크기: 5×10⁻⁴ ‖v_t‖², 관절 속도: 5 ‖q̇_{t+1}‖²
- 관절 위치·속도 한계: 한계의 2.5% 안쪽부터 hinge 제곱 penalty (가중치 5×10⁴)
- 입력 평활: 첫 입력 4 ‖v_0 − v_prev‖² (v_prev는 직전 step의 u_mppi), 이후 4/(H−1) Σ‖v_t − v_{t−1}‖²
- 충돌: 로봇 자기충돌·바닥, 하중 정육면체와 로봇·바닥, 여유 5 mm, 가중치 5×10⁴

stage t의 상태 비용은 같은 시각의 기준 `ref(t+1)`과 비교한다. 사용한 라이브러리
(pytorch_mppi 0.9.1)는 stage t에 x_{t+1}을 넘기기 때문이다.

importance-sampling 보정항은 쓰지 않는다. 이는 Williams et al.의 정보이론 MPC에서 기저 분포를
현재 명목 입력으로 둔 경우(α = 1, γ = λ(1 − α) = 0)에 해당한다.

### 3.2 DOB와 출력 보상

```
residual = u_app − [ M(q) a_f + C(q, q̇) q̇ + g(q) + B q̇ ]     (a_f: 차분 가속도에 6 Hz 1차 LPF)
d_hat    = Q(residual)                                        (Q: 3 Hz 1차 저역통과)
u_req    = u_mppi + d_hat,     u_app = clip(u_req)
```

MPPI 예측 모델에는 외란을 넣지 않는다. 보상은 출력에서만 한다.
N1과 D0는 한 번의 MPPI 계산 동안 `d_hat`을 상수로 본다(저주파 외란 가정).

### 3.3 N0, N1, D0

```
N0:  S_k = J(v)
N1:  S_k = J(v) + 1 · Σ_t ‖ relu(v_t + d_hat − u_max) + relu(u_min − v_t − d_hat) ‖²
D0:  S_k = J(v),   v_t ∈ [u_min − d_hat, u_max − d_hat]
```

- N1은 보상 후 요청 토크가 한계를 넘는 양에만 비용을 준다. 한계 안에서는 N0와 같다.
- D0는 warm start, 모든 sample, 가중 평균 후의 입력열을 이동된 범위로 투영한다.
  그래서 계산 시점의 `d_hat` 기준으로 요청 토크가 한계를 넘지 않는다.

### 3.4 실시간 구현

같은 수식을 GPU kernel로 다시 구현해 MPPI 한 번 계산이 19.10 ms에서 0.45 ms로 줄었다.

- 동역학은 RNEA 7회의 6×6 행렬 연산을 0 항과 중복을 제거한 스칼라 식(약 2,200개)으로 펼쳤다.
- 동역학, FK, 비용을 K×H 전체에 대해 kernel 하나로 계산한다.
- 충돌은 link 좌표계에서 계산하고, 비용이 정확히 0인 쌍만 bounding box로 건너뛴다.

수식, K, H, 가중치는 바꾸지 않았다. 동일성 검증 결과는 다음과 같다.
- 동역학은 float64 기준 1e-12 이내로 일치한다.
- 충돌 비용은 68만 자세에서 누락·오검출이 없었다.
- 최종 168회 실행의 RMSE 차이는 최대 0.0004 mm다.

## 4. 실험

| 실험 | 조건 | seed | 데이터 |
|---|---|---|---|
| E1 최종 비교 | 무외란(DOB on/off), 45, 47.5 Nm × N0/N1/D0 | 10 | `final_runs.csv`, `final_statistics.csv` |
| E2 외란 크기 sweep | 0, 35–60 Nm 21단계 × N0/N1/D0 (630회) | 10 | `sweep_*.csv` |
| E3 원인 확인 | actuator clamp 제거, DOB LPF·센서 노이즈, 첫 입력 평활 비용 제거 | 10 | `final_runs.csv`, `dob_*.csv`, `ablation_*.csv` |
| E4 실시간 | 50 Hz 주기, 세 방식 × 8,000 step | 10 | `realtime_latency_summary.csv` |
| E5 민감도 | K, H, λ를 하나씩 변경 (사후 확인) | 3 | `sensitivity_K_H_lambda.csv` |

보조 강건성 확인(seed 3개): 40 s 장시간 실행과 외란의 0.1 Hz ±5% 변동.

## 5. 결과

### 5.1 최종 비교 (E1, 10 seed, 평균 ± 표본표준편차, mm)

| 조건 | N0 | N1 | D0 |
|---|---:|---:|---:|
| 무외란, DOB off | 1.34 ± 0.07 | | |
| 무외란, DOB on | 2.44 ± 0.09 | | |
| 45 Nm (한계의 30%) RMSE | 5.91 ± 0.62 | 3.98 ± 0.29 | 3.52 ± 0.19 |
| 45 Nm 최대 오차 평균 | 17.80 | 10.41 | 8.16 |
| 47.5 Nm (31.7%) RMSE | 8.81 ± 0.61 | 5.55 ± 0.31 | 4.40 ± 0.19 |
| 47.5 Nm 최대 오차 평균 | 26.57 | 15.96 | 11.38 |

- 포화로 N0의 RMSE는 무외란 대비 45 Nm에서 2.4배, 47.5 Nm에서 3.6배가 된다.
- N0 대비 RMSE 감소는 45 Nm에서 N1 32.7%, D0 40.4%, 47.5 Nm에서 N1 36.9%, D0 50.0%다.
- 늘어난 오차 중 줄인 비율은 N1 51–56%, D0 69%다.
- RMSE와 최대 오차 모두 모든 비교에서 10/10 seed가 N0 > N1 > D0다. 검정은 Wilcoxon
  부호순위, Holm 보정 p = 0.041이다(21개 비교 family, 그중 정렬 전 코드 비교 6개는 CSV에서 제외).
- 보조 확인(3 seed): 40 s 실행 6.55 / 4.25 / 3.67 mm, 0.1 Hz ±5% 변동 외란 6.18 / 4.06 / 3.47 mm로
  순서가 유지된다.

### 5.2 외란 크기 sweep (E2, 10 seed)

| 외란 [Nm] (한계 대비) | RMSE N0 / N1 / D0 [mm] | 개선 seed N1<N0, D0<N0, D0<N1 | 유효 N0 / N1 / D0 |
|---|---|---|---|
| 35 (23%) | 2.44 / 2.44 / 2.44 | 차이 없음 | 10 / 10 / 10 |
| 40 (27%) | 2.44 / 2.44 / 2.45 | 8, 2, 2 | 10 / 10 / 10 |
| 41 (27%) | 2.48 / 2.46 / 2.49 | 8, 1, 0 | 10 / 10 / 10 |
| 42 (28%) | 2.67 / 2.55 / 2.61 | 10, 7, 0 | 10 / 10 / 10 |
| 43 (29%) | 3.23 / 2.81 / 2.83 | 10, 10, 2 | 10 / 10 / 10 |
| 44 (29%) | 4.39 / 3.31 / 3.16 | 10, 10, 10 | 10 / 10 / 10 |
| 45 (30%) | 5.91 / 3.98 / 3.52 | 10, 10, 10 | 10 / 10 / 10 |
| 46 (31%) | 7.27 / 4.62 / 3.87 | 10, 10, 10 | 10 / 10 / 10 |
| 47 (31%) | 8.33 / 5.23 / 4.21 | 10, 10, 10 | 10 / 10 / 10 |
| 48 (32%) | 9.52 / 6.05 / 4.73 | 10, 10, 10 | 10 / 10 / 10 |
| 49 (33%) | (무효 포함) | 유효한 쌍만 | 8 / 8 / **3** |
| 50–60 (33–40%) | (무효) | – | 0 / 0 / 0 |

전체 21단계와 J_clip, 포화 시간 비율은 `data/sweep_summary.csv`에 있다.

- **포화 전(≤ 40 Nm):** 세 방식의 성능이 같다. 요청 토크가 한계를 넘는 순간은 37.5 Nm부터
  생기지만 추종 저하로 이어지지 않는다.
- **포화 시작(41–43 Nm):** N0가 나빠지기 시작하고 N1은 대부분의 seed에서 낫다.
  D0는 아직 N1보다 나쁜 seed가 많다.
- **포화 구간(44–48 Nm):** 모든 쌍이 10/10 seed에서 N0 > N1 > D0다. 외란이 커질수록 차이가 커진다.
- **제약 위반:** 49 Nm에서 유효 seed가 N0 8, N1 8, D0 3으로 **D0가 먼저 무너진다.**
  50 Nm부터는 세 방식 모두 무효다. 무효 실행은 모두 팔꿈치 뒤집힘을 포함하며, 일부는 충돌이나
  관절 한계 위반을 동반한다.
- 누적 clipping은 D0가 구조상 항상 0이고, N1은 N0의 약 40%다(예: 47.5 Nm에서 107.6 → 42.5 Nm·s).

### 5.3 원인 확인 (E3, 10 seed)

| 대조 실험 | 결과 | 해석 |
|---|---|---|
| actuator clamp 제거(비물리 진단), 45 Nm N0 | 2.44 mm, 무외란과 같음 | 이 조건의 저하는 포화에서 비롯된다 |
| 첫 입력 평활 비용 제거, 45 Nm | 무외란 2.84 → N0 5.48 / N1 4.06 / D0 3.76 | 순서와 개선 비율(N1 54%, D0 65%)이 평활 비용 설정에 의존하지 않는다 |
| DOB 가속도 LPF 끔, 무외란 | 2.44 → 1.40 mm | DOB on의 RMSE 증가는 주로 가속도 LPF에서 생긴다 |

DOB on 무외란 RMSE 증가에 대한 주의점이다.
- LPF를 끄면 RMSE 증가분의 94%가 사라지지만 추정값 RMS는 1.84 → 2.47 Nm로 커진다.
  센서 노이즈를 끄면 추정값은 1.06 Nm로 줄어든다.
- 따라서 "추정값의 크기"는 LPF만으로 설명되지 않는다. LPF는 차분 가속도의 노이즈를 줄이려고
  필요한 것이며, 이 실험은 원인을 분리하기 위한 것이지 LPF 제거를 제안하는 것이 아니다.

### 5.4 실시간성 (E4)

측정값은 예정된 주기 시작부터 잰 **대기 종료 후 제어 계산 지연**이다. 깨어남 지연, 상태와
토크의 CPU–GPU 전송, MPPI, 보상·clamp, DOB 갱신을 포함하고 plant 모사는 제외한다.
센서 취득과 로봇 통신 지연은 포함하지 않는다.

| 구현 | 방식 | 평균 [ms] | p99 [ms] | 최악 [ms] | 20 ms 초과 (8,000 step 중) |
|---|---|---:|---:|---:|---:|
| GPU kernel | N0 | 3.57 | 4.36 | 6.43 | 0 |
| GPU kernel | N1 | 3.67 | 4.41 | 6.55 | 0 |
| GPU kernel | D0 | 4.05 | 4.86 | 7.89 | 0 |
| 기존 구현 | N0 | 20.50 | 21.03 | 311.3 | 7,998 |

측정 환경은 GTX 1650, i7-13700K, Linux(powersave)다. 실시간 OS가 아니므로 hard real-time
보장은 아니다.

### 5.5 민감도 (E5, 사후 확인, 3 seed)

K = 136, H = 10, λ = 64는 이 분석 이전에 고정한 값이다. 이 분석은 선정 근거가 아니다.

- K를 136에서 2048로 늘려도 무외란 RMSE는 같다(2.39 mm). 45, 47.5 Nm에서도 N0 > N1 > D0 순서는
  유지되지만, **47.5 Nm의 N0는 K에 대해 수렴하지 않는다**(K = 2048에서 +18%).
- H ≤ 8이면 외란 시 무효가 되고, H = 9는 47.5 Nm에서 일부 seed가 무효다.
  H ≥ 11이면 추종이 둔해지고, H = 16에서는 45 Nm에서 세 방식의 차이가 사라진다.
- λ = 32–96에서 순서는 유지되지만 λ = 128은 47.5 Nm에서 무효다.
- 유효 sample 수 지표 η(≈ 가중치가 의미 있는 sample 수)는 45 Nm에서 약 5로 문헌의 참고 범위
  5–10(Pezzato et al.; Biased-MPPI) 하한이고, 47.5 Nm에서는 4.4–4.8이다.

## 6. 그림

모든 그림은 `figures/`에 있다. 영문 축 라벨이며 N0 파랑, N1 주황, D0 청록이다.

| 파일 | 내용 | 권장 위치 |
|---|---|---|
| `setup_robot_payload_trajectory.png` | UR5 초기 자세, 5 kg 하중 정육면체, YZ 원 기준 궤적 | 실험 설정 |
| `disturbance_profiles.png` | J2 외란 파형(35, 42, 45, 48, 50 Nm). 0.5–2.5 s 증가 후 유지 | 실험 설정 |
| `final_rmse_comparison.png` | 45, 47.5 Nm의 RMSE 막대(10 seed 평균, 표준편차, 개별 seed 점), 무외란 기준선 | 핵심 결과 |
| `rmse_vs_disturbance.png` | 외란 크기별 RMSE(log 축). 채운 점은 10 seed 모두 유효, 빈 점은 무효 포함 | 핵심 결과 |
| `timeseries_45Nm.png` | 45 Nm의 TCP 오차, 누적 clipping, J2 요청 토크(평균선, seed 최소–최대 범위) | 결과 해석 |
| `timeseries_42Nm.png` | 포화 시작 구간의 같은 그림 | 결과 해석 |
| `timeseries_48Nm.png` | 모든 방식이 유효한 마지막 단계의 같은 그림 | 결과 해석 |
| `timeseries_50Nm.png` | 모든 방식이 무효(팔꿈치 뒤집힘)인 단계의 같은 그림 | 한계 |
| `timeseries_35Nm.png` | 포화 전 단계의 같은 그림(세 방식 동일) | 부록 |
| `realtime_latency_cdf.png` | 50 Hz에서 제어 계산 지연의 누적분포(kernel 세 방식, 기존 구현 N0), 20 ms 선 | 실시간성 |

시계열 그림의 요청 토크는 음수 방향이며 −150 Nm가 한계다. 오차 그림은 0–15 s만 그린다.
기준 궤적 끝(15.7 s 이후)을 정지값으로 채워 생기는 인공 오차를 빼기 위해서다.

## 7. 주장 범위와 한계

논문에 쓸 수 있는 주장은 다음과 같다.
1. 외란 보상으로 요청 토크가 포화되면 기본 MPPI(N0)의 추종 성능이 떨어진다. clamp 제거 대조로
   원인이 포화임을 확인했다.
2. 보상량을 고려해 명목 입력을 제약하면(N1 soft, D0 hard) 저하가 줄어든다. 44–48 Nm에서
   N0 > N1 > D0가 10/10 seed에서 성립한다.
3. GPU kernel 구현으로 세 방식 모두 20 ms 주기 안에서 제어 계산을 마친다(5.4의 측정 정의에 한함).

반드시 함께 밝혀야 하는 한계는 다음과 같다.
- **유효 범위가 좁다:** 이득이 나타나는 외란 범위는 약 41–48 Nm(한계의 27–32%)이고,
  49 Nm부터 제약 위반이 생긴다.
- **D0의 강건성:** D0는 포화가 약한 41–43 Nm에서 N1보다 나쁜 seed가 많고, 49 Nm에서는 가장 먼저
  무효가 된다(유효 3/10). "D0가 모든 조건에서 최선"이라고 쓰지 않는다.
- **N1/D0와 자세 안정성:** N1과 D0가 자세 안정성을 높인다고 주장하지 않는다. 최소 q3는 N0보다
  작아질 수 있다(48 Nm에서 N0 0.33, N1 0.29, D0 0.27 rad).
- **일반화:** 한 궤적, 한 하중, 한 관절(J2)의 시뮬레이션 결과이며 다른 조건으로 일반화하지 않는다.
- **파라미터:** K, H, λ의 민감도는 사후 확인이며 47.5 Nm N0는 K에 수렴하지 않았다.
- **실시간성:** 센서·통신 지연을 포함하지 않으며 hard real-time 보장이 아니다.
- **외란 가정:** 외란은 저주파(상수 유지)로 가정했다. 빠르게 변하는 외란은 다루지 않는다.

## 8. 논문에 쓰지 말 것

- **직전 입력 기준값 투영:** 평활 비용의 v_prev를 실현 가능 범위로 자르는 진단 실험이다.
  저장소의 `kroc/README.md` 10.4와 `kroc/evidence/controls_20261123/causes.json`에는 기록이
  남아 있지만, 이 논문에서는 언급하거나 사용하지 않는다.
- **D0의 sample 투영 제거 등 그 밖의 투영 변형 실험:** 같은 이유로 쓰지 않는다.
- **이전 단계 수치(J3 133 Nm 등)와의 직접 비교:** 코드와 조건이 다르다.
- **K, H, λ 표를 "파라미터 선정 근거"로 쓰는 것:** 사후 민감도 분석이다.
- **`kroc/README.md` 1–5절의 수치:** 비용 시점 정렬 이전 코드의 결과다.

## 9. 데이터 사전 (`data/`)

| 파일 | 행 단위 | 주요 열 |
|---|---|---|
| `final_runs.csv` | 조건 × 방식 × seed (108행) | condition, case, seed, rmse_mm, max_error_mm, q3_min_rad, valid |
| `final_statistics.csv` | 쌍 비교 15개 | mean_difference_mm(앞 − 뒤), ci95_half_width_mm, positive_seeds, wilcoxon_p, holm_p |
| `sweep_summary.csv` | 외란 단계 21개 | 방식별 rmse 평균·표준편차, 최대 오차, j_clip_nm_s, clip_time_percent, valid_seeds, 개선 seed 수 |
| `sweep_runs.csv` | 단계 × 방식 × seed (630행) | rmse_mm, max_error_mm, j_clip_nm_s, q3_min_rad, elbow_ok, limits_ok, collision_free, valid, solve 시간 |
| `sweep_timeseries_selected_levels.csv` | 대표 5단계 × 방식 × 통계(mean/min/max) × 시각 | ee_error_mm, requested_j2_nm, cumulative_clip_nm_s, d_true_j2_nm (0.02 s 간격, 15 s까지) |
| `realtime_latency_summary.csv` | 구현 × 대기 방식 × 방식 | wake/command/update_done의 mean, p99, max [ms], 20 ms 초과 수 |
| `dob_no_disturbance_diagnostic.csv` | DOB·LPF·노이즈 조건 7개 | rmse_mean_mm, d_hat_j2_rms_nm |
| `ablation_first_input_smoothing_off.csv` | 첫 입력 평활 비용 제거 | rmse_mean_mm, valid_seeds |
| `sensitivity_K_H_lambda.csv` | 설정 × 방식 × 외란 | rmse 평균·표준편차, valid_seeds, ess_median, eta_median, solve_mean_ms |

정의는 다음과 같다.
- RMSE와 최대 오차는 5 < t ≤ 15 s 구간의 TCP 위치 오차다.
- J_clip은 전 관절 |요청 − 적용| 토크를 실행 전체(0–16 s)에 대해 적분한 값[Nm·s]이다.
- clip_time_percent는 평가 구간에서 J2 요청이 150 Nm를 넘은 step의 비율이다.
- 개선 seed는 두 실행이 모두 유효한 seed 중 RMSE가 더 낮은 수다.
- sweep_summary의 평균은 무효 실행을 포함한다. 유효가 10 미만인 단계는 성능 비교에 쓰지 않는다.

## 10. 재현

```bash
P=/home/arclab-04/miniconda3/envs/manipulator/bin/python
$P -m kroc.make_paper_figures                 # 이 폴더의 그림과 CSV (근거 자료에서, 수십 초)
$P -m kroc.sweep_saturation                   # E2 재실행, 약 50분 (CUDA)
$P -m kroc.validate_aligned_circle --output D # E1 재실행, 약 20분
$P -m kroc.benchmark_realtime                 # E4 재측정, 약 40분
./execute.sh gui                              # preset: payload_circle_T2_J2_45_compare.yaml 등
```

참고 문헌 후보:
- G. Williams et al., "Information Theoretic Model Predictive Control: Theory and Applications to
  Autonomous Driving" (arXiv:1707.02342)
- C. Pezzato et al., "Sampling-Based Model Predictive Control Leveraging Parallelizable
  Physics Simulations," RA-L 2025 (arXiv:2307.09105)
- "Biased-MPPI: Informing Sampling-Based Model Predictive Control by Fusing Ancillary Controllers"
  (arXiv:2401.09241, 저자 확인 필요)
