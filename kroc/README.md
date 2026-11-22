# KRoC 최종 연구 기록

이 문서는 2026-11-18 데이터 정리 이후 KRoC 연구의 단일 기준 문서다. 상태 제약
동적 축소의 최선 진단 사례와 입력 제약 N0/N1/D0 비교에 필요한 결론, 구성, 데이터
위치와 재현 코드를 기록한다. 그 외 탐색 원자료·중간 보고서·일회성 스크립트는
삭제했다.

## 1. 최종 연구 범위

남긴 주제는 두 가지다.

1. **상태 제약 동적 축소**: 일반적인 성공으로 채택하지 않고, 가장 명확했던 J5
   estimator-only 사례를 제한된 진단 결과로 보존한다.
2. **입력 제약 축소**: N0/N1/D0를 최종 주제로 삼는다. N0은 기준, N1은 soft
   clipping-loss 비용, D0는 보상 추정값에 따라 이동하는 hard nominal-input bound다.

본 시스템은 시뮬레이션 UR5의 토크 기반 MPPI다. 결과는 사용한 궤적·초기 자세·센서
노이즈·DOB·외란 범위에 한정하며 하드웨어 안전성이나 전역 제약 만족을 보장하지
않는다.

## 2. 공통 최종 설정

| 항목 | 값 |
|---|---|
| MPPI samples / horizon | K=136 / H=10 |
| 제어 주기 | 20 ms |
| temperature | 64 |
| covariance diagonal | `[9, 9, 9, 0.3136, 0.3136, 0.3136]` Nm² |
| sampling | horizon 공유 Gaussian noise |
| warm start | inverse-dynamics feedforward + residual sequence |
| DOB | Q cutoff 3 Hz, acceleration LPF 6 Hz |
| 센서 노이즈 표준편차 | q=0.001 rad, dq=0.01 rad/s |
| 토크 한계 | ±`[150,150,150,28,28,28]` Nm |
| 입력 적용 | 최종 actuator physical clamp 유지 |

최종 GUI N0 설정은 `configs/gui_final_n0.yaml`에 있다. 공통 controller/runner와
GUI 변경도 최종 구현의 일부이며 `kroc` 폴더 밖에 위치한다.

## 3. 상태 제약 동적 축소

### 3.1 구성

고정 마진은 전체 관절 상태 범위의 한쪽 2.5% 또는 5%다. 동적 방식은 2.5%를
기본값으로 하고 위험 방향만 최대 5%까지 확대한다.

```text
d_margin = clip(d_hat, -tau_max, tau_max)
a_raw    = -solve(M(q), d_margin)
a        = sign(a_raw) * max(abs(a_raw) - nominal_p99_deadband, 0)

m_q+ = m_q0 + max(dq, 0) * Tr + 0.5 * max(a, 0) * Tr^2
m_v+ = m_v0 + max(a, 0) * Tr
```

- `Tr=0.10 s`
- 한쪽 마진을 전체 상태 범위의 2.5–5%로 제한
- 증가 `beta_up=1`, 감소 `beta_down=0.1`
- solve마다 갱신하고 해당 rollout horizon에서는 고정
- 위치·속도 soft-constraint 비용이 시작되는 경계만 이동
- 보존 사례에서는 headroom, input tightening, output/rollout DOB compensation 미사용

이는 반응시간 동안의 위험 방향 이동량을 이용한 heuristic이다. stopping-distance
증명, 불변 집합 또는 hard constraint가 아니다.

### 3.2 보존한 최선 사례

조건은 최종 N0, 반경 4 cm·6 s 원궤적, J5 음의 외란 12.5 Nm, 독립 seed
8707/8717/8727이다. DOB 추정값은 마진 생성에만 사용했다.

| 방식 | 위치 위반 실행 | 평균 RMSE [mm] | 최악 위치 초과 [rad] |
|---|---:|---:|---:|
| static 2.5% | 3/3 | 8.954 | 0.17123 |
| dynamic estimate 2.5–5% | 0/3 | 8.417 | 0 |
| static 5% | 0/3 | 9.677 | 0 |

이 결과는 제한된 estimator-only 조건의 최선 사례일 뿐이다. 일반적인 상태 제약
보장, post-DOB 포화 해결 또는 모든 관절·궤적에서의 우월성으로 주장하지 않는다.

보존 데이터:

```text
kroc/results/n0_other_joint_estimator_margin_20261118/final_refinement/
  plan.json
  verdict.json
  selected_condition_summary.json
  end_state_diagnostics.json
  validation/          # 3 seeds × 3 cases의 JSON/NPZ
```

재현 코드의 진입점은 `refine_n0_other_joint_estimator_margin.py`이며 핵심 구현은
`state_margin.py`다.

## 4. 입력 제약 N0/N1/D0

세 방식 모두 명목 rollout 상태 전이에 외란을 직접 넣지 않는다. 실제 요청 입력은
MPPI 명목 입력과 causal DOB 보상값으로 만들고 actuator physical clamp를 마지막에
적용한다.

### N0

- nominal input을 rollout 전에 자르지 않는다.
- 추가 입력 제약 비용이 없다.

### N1

- N0의 sampler, rollout 및 목적함수를 유지한다.
- causal `d_hat`을 더했을 때 예상되는 요청 토크 clipping loss의 제곱 비용만 추가한다.
- 최종 weight는 `w=1`이며 물리 한계 안에서는 추가 비용이 0이다.
- 95% headroom 비용은 최종 설정에서 사용하지 않는다.

```text
u_req_pred = v + d_hat_snapshot
loss       = relu(u_req_pred - u_max) + relu(u_min - u_req_pred)
J_N1       = J_N0 + w * sum(loss^2)
```

### D0

- 목적함수는 N0와 같고 N1 soft cost가 없다.
- solve 시작의 causal 보상 snapshot을 `c=d_hat`으로 둔다.
- nominal input 범위를 `[u_min-c, u_max-c]`로 이동한다.
- warm start, sampled sequence, weighted update 후 전체 sequence를 이동 범위로 투영한다.
- nominal rollout은 투영된 `v`로 진행하며 외란을 상태 전이에 중복 삽입하지 않는다.

D0가 동일 snapshot 기준 요청 clipping을 0으로 만드는 것은 구성에 따른 결과다.
추정 오차, 상태 제약, 충돌 또는 추종 오차까지 제거하는 보장은 아니다.

## 5. 최종 입력 제약 결과

J3 음의 외란은 0.5 s부터 raised cosine으로 증가해 2.5 s에 목표 크기에 도달하고
6 s까지 유지했다. 각 외란 크기에서 동일한 10 seed로 비교했다.

| 외란 [Nm] | N0 RMSE [mm] | N1 RMSE [mm] | D0 RMSE [mm] | 정상 실행 N0/N1/D0 |
|---:|---:|---:|---:|---:|
| 132.5 | 5.809 | 5.817 | 5.736 | 10/10/10 |
| 133.0 | 6.645 | 6.032 | 5.810 | 10/10/10 |
| 133.5 | 21.065 | 10.789 | 8.420 | 10/10/10 |
| 134.0 | 53.221 | 36.614 | 28.245 | 10/10/10 |
| 135.0 | 103.593 | 95.787 | 78.043 | 10/10/10 |
| 136.0 | 210.511 | 175.419 | 136.105 | 7/9/10 |
| 138.0 | 472.836 | 346.266 | 480.174 | 1/1/0 |

132.5–135 Nm에서 세 방식 모두 정상인 50쌍에서는 D0가 N1보다 RMSE가 낮았다.
D0 요청 clipping은 0이었다. 136 Nm부터 안전 실패가 나타났고 138 Nm에서는 세
방식 모두 유효 추종을 유지하지 못했으므로 해당 구간을 정상 성능 우위로 해석하지
않는다.

N1의 N0 대비 개선과 N1의 D0 대비 우위는 서로 다른 질문이다. 최종 결과는 직접
bound projection인 D0가 강한 비교 기준이며, 이 조건에서 N1의 D0 대비 추가 우위는
입증되지 않았음을 보여준다.

## 6. 보존 입력 데이터

전체 load sweep의 seed별 지표는 두 `summary.json`에 남겼다. 디스크 절감을 위해
−133 Nm 이외의 원시 NPZ는 삭제했다.

```text
kroc/results/final_n1_load_sweep_20261117/
  plan.json
  summary.json          # N0/N1 전체 7 loads × 10 seeds 통계
  source/
  raw/a133/             # N0/N1 원시 실행 일부

kroc/results/final_three_way_load_sweep_20261117/
  plan.json
  summary.json          # N0/N1/D0 전체 7 loads × 10 seeds 통계
  comparison_audit.json
  source/
  raw/a133/             # D0 원시 실행 일부

kroc/results/final_mppi_optimization_20261117/
  plan.json
  calibration.json
  margin_capture.json
  execution_choice.json
  selected_execution.json
  final_audit.json
  final_comparison.json
  margin_comparison_primary/   # −133 Nm의 나머지 N0/N1/D0 5 seeds
```

세 디렉터리의 −133 Nm 자료를 합치면 10 seed의 N0/N1/D0 원시 시계열이 완성된다.
요약 파일에는 삭제 전 전체 sweep의 집계와 paired 비교가 기록되어 있다. 삭제된
다른 load의 NPZ는 Git으로 복구할 수 없다.

## 7. 보존 코드

- 상태 제약: `state_margin.py`, `compare_n0_state_margins.py`,
  `search_n0_other_joint_estimator_margin.py`, `refine_n0_other_joint_estimator_margin.py`
- 입력 제약: `input_headroom_study.py`, `optimize_final_mppi.py`,
  `sweep_final_n1_load.py`, `extend_load_sweep_d0.py`
- 충돌 감사: `collision_geometry.py`, `collision_kernels.py`,
  `self_collision_cost.py`, `audit_circle_validation.py`
- 핵심 테스트: `test_state_margin.py`, `test_n0_other_joint_estimator_margin.py`,
  `test_input_headroom_study.py`, `test_final_input_headroom_checks.py`,
  `test_fast_final_mppi.py`, `test_self_collision_cost.py`, `test_sensor_noise.py`

위 진입점의 직접·간접 import에 필요한 보조 모듈만 함께 남겼다. 과거 탐색용
benchmark, plot, report 생성, Spark bundle, PDF와 중간 보고서는 삭제했다.
실행 코드가 공통 기준값을 읽을 수 있도록
`configs/pre_input_study_baseline.json` 하나만 동결 프리셋으로 보존했다.

## 8. 검증 명령

```bash
/home/arclab-04/miniconda3/envs/manipulator/bin/python -m unittest -v \
  kroc.test_state_margin \
  kroc.test_sensor_noise \
  kroc.test_self_collision_cost \
  kroc.test_input_headroom_study \
  kroc.test_final_input_headroom_checks \
  kroc.test_fast_final_mppi \
  kroc.test_n0_other_joint_estimator_margin
```

GPU 실험은 CUDA preflight 성공을 요구하며 CPU fallback 결과를 연구 데이터로
기록하지 않는다. 삭제된 sweep을 다시 만들려면 보존 driver의 출력 경로와 기존 파일
존재 검사를 먼저 새 경로에 맞춰야 한다.
메시 기반 충돌 검사와 `test_self_collision_cost`에는 별도 `python-fcl` 및
`trimesh` 설치가 필요하다.

## 9. 정리 이력

- 2026-11-18 1차 정리: `kroc` 약 979 MB에서 150 MB로 축소했다.
- 2026-11-21 최종 정리: 단일 문서, 최소 재현 코드, 상태 최선 사례와 −133 Nm
  N0/N1/D0 원시 자료 및 전체 sweep summary만 남겼다.
- `kroc/results/`, `*.npz`, `*.png`는 Git ignore 대상이다. 원시 결과 삭제는 Git
  history에 나타나지 않으며 별도 백업이 없으면 복구할 수 없다.

## 10. Payload 고부하 궤적의 입력 포화 비교 (2026-11-22)

외란을 정격의 약 30%로 낮추기 위해 하중과 궤적으로 명목 토크 `u_mppi`를 키운
조건이다. 5절(J3, 89% 외란)과는 관절, 궤적, 비용의 시점 정렬과 평가 구간이 모두
다르므로 두 절의 RMSE를 직접 비교하지 않는다.

### 10.1 조건

| 항목 | 값 |
|---|---|
| 설정 | `configs/payload_circle_T2_J2_45_n0.yaml` |
| 하중 / 궤적 | 5 kg / YZ 원, 반경 0.20 m, 주기 2 s, 시작 ramp 3 s, 총 16 s |
| 외란 | J2 음의 방향, 0.5–2.5 s raised-cosine 증가 후 유지 |
| 제어기 | 2절 설정 + payload 안정화 비용(posture 5000, boundary/smooth 4) |
| 시점 정렬 | `align_running_reference: true` |
| 평가 | 5 < t ≤ 15 s (5주기), seed 7207–7297 (10개) |
| 유효 실행 | 충돌 및 관절 위치·속도 위반 없음, 전 구간 q3 > 0.2 rad |

J2를 사용한 이유는 payload 중력과 궤적 관성 부하가 이 관절에 집중되기 때문이다.

### 10.2 비용 시점 정렬

`pytorch_mppi`는 stage t의 running cost를 전진된 상태 x(t+1)로 계산한다. 기존
코드는 이를 ref(t)와 비교해 이 궤적에서 약 14 ms의 추종 지연을 만들었다.
`align_running_reference`는 위치·자세 running cost의 기준만 ref(t+1)로 옮긴다.
terminal cost와 첫 입력의 boundary cost는 바꾸지 않는다.

이 정렬은 이후 모든 MPPI 경로(기본 tracker, research/final N0, fast/captured/graph,
payload, 동적 장애물 비용)의 기본값이 되었다. iLQR과 SQP는 원래 x(t)와 ref(t)를
비교하므로 변경하지 않았다.

1–5절과 2026-11-22 이전의 모든 MPPI 결과는 정렬 전 비용으로 만든 것이다. 재현하려면
`tracker_options`(payload 설정은 `experimental_stabilization`)에
`align_running_reference: false`를 명시하거나 커밋 `ad30c72`를 사용한다. 옵션 키가
없는 기존 `resolved_config.json`도 이제 정렬된 비용으로 실행되므로 주의한다.

5절 조건(−133 Nm, 느린 4 cm 원)을 3 seed로 다시 확인한 결과 RMSE 변화는 1% 미만이고
N0 > N1 > D0 순서는 유지되었다. 전체 sweep은 다시 실행하지 않았다.

기준 궤적 끝은 정지값으로 padding되어 마지막 약 0.3 s에 21–28 mm의 인공 오차가
생긴다. 따라서 평가에서 제외한다.

### 10.3 결과 (평균 ± 표본 표준편차, mm)

무외란 N0:

| 비용 시점 | DOB off | DOB on |
|---|---:|---:|
| 정렬 전 | 7.73 ± 0.08 | 8.91 ± 0.09 |
| 정렬 후 | 1.34 ± 0.07 | 2.44 ± 0.09 |

DOB on의 RMSE 증가와 무외란 추정값의 크기는 원인을 구분해서 본다.
아래는 시점 정렬 후 무외란 N0의 개입 실험이다(10 seed, `controls_20261123/causes.json`).

| DOB | 가속도 LPF | 센서 노이즈 | RMSE [mm] | J2 추정값 RMS [Nm] |
|---|---|---|---:|---:|
| off | - | 있음 | 1.34 | 0 |
| on | 6 Hz (현재) | 있음 | 2.44 | 1.84 |
| on | 12 Hz | 있음 | 1.68 | 1.96 |
| on | 없음 | 있음 | 1.40 | 2.47 |
| off | - | 없음 | 1.18 | 0 |
| on | 6 Hz | 없음 | 2.32 | 1.06 |
| on | 없음 | 없음 | 1.18 | 0.00 |

- 이 조건에서 **RMSE 증가**의 주요인은 가속도 LPF다. LPF를 끄면 증가분의 94%가 사라진다.
  입력 토크에는 없는 지연이 관성 토크 항에만 들어가는 것이 그 경로로 해석된다.
- **추정값 RMS**는 LPF만으로 설명되지 않는다. 센서 노이즈 성분이 포함되며, LPF를 끄면
  RMSE는 줄어도 추정값 RMS는 커진다.
- LPF는 차분 가속도의 노이즈를 줄이기 위해 필요하다. 이 실험은 원인 분리용이며 LPF
  제거를 제안하지 않는다. 외란 인가 시 LPF 제거의 영향은 시험하지 않았다.

외란 인가, DOB on:

| 비용 시점 | 외란 [Nm] | 지표 | N0 | N1 | D0 |
|---|---:|---|---:|---:|---:|
| 정렬 후 | 45 | RMSE | 5.91 ± 0.62 | 3.98 ± 0.29 | 3.52 ± 0.20 |
| 정렬 후 | 45 | 최대 오차 평균 | 17.80 | 10.41 | 8.16 |
| 정렬 후 | 47.5 | RMSE | 8.81 ± 0.61 | 5.55 ± 0.31 | 4.40 ± 0.19 |
| 정렬 후 | 47.5 | 최대 오차 평균 | 26.57 | 15.96 | 11.38 |
| 정렬 전 | 45 | RMSE | 11.56 ± 0.51 | 10.13 ± 0.24 | 9.83 ± 0.17 |

세 행의 RMSE와 정렬 후 최대 오차 모두 10/10 seed에서 N0 > N1 > D0였다
(Wilcoxon, 21개 비교 Holm 보정 p = 0.041, `validation_20261122/statistics.json`).

### 10.4 대조 실험

| 실험 (정렬 후, 45 Nm) | N0 | N1 | D0 | seed 수 |
|---|---:|---:|---:|---:|
| actuator clamp 제거 (비물리 진단) | 2.44 | - | - | 10 |
| sample은 두고 평균 입력열만 투영 | - | - | 4.04 | 10 |
| 40 s 장시간 (5–39 s 평가) | 6.55 | 4.25 | 3.67 | 3 |
| 0.1 Hz, ±5% 변동 외란 | 6.18 | 4.06 | 3.47 | 3 |

clamp만 제거하면 무외란 값으로 돌아가므로 이 조건의 성능 저하는 포화에서 비롯된다.

포화가 저하로 이어지는 경로를 나누기 위한 **원인 분석용 대조군**은 다음과 같다(10 seed,
DOB on, RMSE mm). 제안 기법이 아니며, N0/N1/D0의 정의와 결과는 바꾸지 않는다.

| 대조군 | 45 Nm | 47.5 Nm | N0→D0 개선량 대비 |
|---|---:|---:|---:|
| N0 (기준) | 5.91 | 8.81 | 0% |
| N1 | 3.98 | 5.55 | 81% / 74% |
| D0 | 3.52 | 4.40 | 100% |
| N0 + warm-start 중심 투영 | 4.44 | 5.81 | 62% / 68% |
| N0 + 직전 입력 기준값 투영 | **3.61** | **5.05** | 96% / 85% |
| N0 + 위 두 가지 | 3.97 | 5.35 | 81% / 78% |
| D0에서 sample 투영만 제거 | 4.04 | 5.34 | 78% / 79% |
| N1 + 직전 입력 기준값 투영 | **3.43** | 4.64 | 104% / 95% |

직전 입력 기준값 투영은 첫 입력 평활 비용 `4 (v0 - v_prev)^2`의 `v_prev`만
`[u_min-d_hat, u_max-d_hat]`로 투영한 것이다.

- 입력열 또는 그 기억을 보상 후 실현 가능한 범위로 투영하는 효과가 큰 것을 확인했다.
- 이것을 windup 방지라는 **단일 기전으로 분리하지는 못했다.** 각 투영은 기억을 없애는
  동시에 실현 가능한 쪽으로 당기는 제약으로도 작용하고, 효과가 더해지지도 않는다.
- 기준값 투영만 한 N0가 N1보다 두 외란 모두 10/10 seed에서 낮다. 45 Nm에서는
  N1+기준값 투영이 D0보다도 10/10 seed에서 낮고, 47.5 Nm에서는 D0가 10/10에서 낮다.
- 따라서 D0의 우위는 **N0, N1, D0 세 방식 사이에서만** 주장한다.
- 첫 입력 평활 비용을 완전히 꺼도 무외란 2.84 mm 대비 45 Nm에서 N0 5.48 / N1 4.06 / D0 3.76 mm로 저하와
  순서가 남는다. 세 방식의 차이가 이 비용 하나 때문에 생긴 것은 아니다.

### 10.5 주장 범위

- 검증한 외란 범위는 45–47.5 Nm이다. 단일 seed 탐색에서 40 Nm는 세 방식의 차이가
  없었고, 50 Nm는 N0의 팔꿈치가 뒤집혔다(q3 < 0).
- N1/D0가 자세 안정성을 높인다고 주장하지 않는다. 최소 q3는 N0보다 작아질 수 있다.
- 실시간성은 10.7절의 kernel backend와 측정 정의(대기 종료 후 제어 계산)에 한해 주장한다.
- N1/D0가 포화 완화 방법 중 최선이라거나 개선이 windup 방지 하나 때문이라고 주장하지 않는다.
- 45/47.5 Nm는 사전 탐색으로 고른 조건이며 다른 궤적·관절로 일반화하지 않는다.

### 10.6 데이터와 재현

```bash
P=/home/arclab-04/miniconda3/envs/manipulator/bin/python
$P -m kroc.validate_aligned_circle                     # 저장된 168회(captured) 재집계
$P -m kroc.validate_aligned_circle --output NEW_DIR    # kernel로 재실행, 약 20분 (--backend captured: 약 70분)
$P -m kroc.summarize_payload_statistics                # 추적된 실행별 값에서 통계 21개 재계산
$P -m kroc.analyze_controls --suites causes            # 10.3-10.4 원인 대조군, 약 25분
$P -m kroc.analyze_controls --suites sensitivity       # 10.8 민감도, 약 35분
$P -m kroc.benchmark_realtime --summarize              # 저장된 step별 시각에서 10.7 표 재계산
$P -m kroc.benchmark_realtime                          # 50 Hz 재측정, 약 40분
$P -m kroc.check_collision_kernel                      # 충돌 kernel 동일성, 약 2분
./execute.sh gui                                       # Preset: payload_circle_T2_J2_45_n0.yaml
```

| 자료 | 위치 | Git |
|---|---|---|
| 168회 실행별 지표, 통계 | `kroc/evidence/validation_20261122/` | 추적 |
| 실시간 step별 시각 72,000개 | `kroc/evidence/realtime_20261123/*.csv.gz` | 추적 |
| 원인 대조군, 민감도 실행별 지표 | `kroc/evidence/controls_20261123/` | 추적 |
| 충돌 kernel 동일성 | `kroc/evidence/equivalence_20261123/` | 추적 |
| 시점 수정 전후 확인 기록 (재현 driver 없음) | `kroc/evidence/alignment_20261122/` | 추적 |
| 168회 시계열 | `kroc/results/aligned_circle*_20261122/` | 제외, 별도 백업 |

각 파일에 환경과 소스 SHA-256을 기록했다. 실시간과 원인 대조군의 일부 hash가 커밋본과
다른 이유와 재현 확인은 `kroc/evidence/PROVENANCE_NOTE_20261123.json`에 있다.

### 10.7 실시간 성능 (kernel backend)

`controller.rollout_backend: kernel`은 N0/N1/D0의 수식, K=136, H=10, 비용과 가중치를
바꾸지 않고 계산 방식만 바꾼 payload 전용 backend다. preset에는 켜져 있고, 키가 없는
설정은 기존 `captured` backend를 쓴다.

- RNEA 7회의 6×6 행렬 연산을 스칼라 식으로 펼치고 LDLᵀ로 푼다.
- 동역학, FK, 추종 비용을 K×H 전체에 대해 GPU kernel 하나로 계산한다.
- 충돌은 link 좌표계에서 계산하고, 비용이 정확히 0인 쌍만 bounding box로 건너뛴다.
- 종단 충돌 비용은 같은 x(H)의 마지막 stage 값을 재사용한다.
- DOB도 kernel로 계산하고, 제어에 쓰이지 않는 고정 margin 진단은 episode 뒤로 옮겼다.

측정 환경: GTX 1650, i7-13700K, powersave governor, 45 Nm, seed 7207–7297, 방식별 8,000 step.

측정값은 **대기 종료 후 제어 계산 지연**이며, 깨어남 지연이 드러나도록 예정된 주기 시작
시각을 0으로 기록했다. 센서 측정부터 actuator까지의 전체 지연이 아니다.

- `토크 출력`: 깨어남 지연 + 상태 CPU→GPU + MPPI + 보상·clamp + 토크 GPU→CPU
- `DOB 완료`: 위에 DOB 갱신까지 포함, 모사 plant와 진단용 예측은 제외
- solve에 쓰는 측정값은 대기 전에 모사된다. 센서 취득과 로봇 통신은 포함하지 않는다.

| backend / 대기 | 방식 | 깨어남 최악 | 토크 출력 평균 / p99 / 최악 | >20 ms | DOB 완료 평균 / p99 / 최악 | >20 ms |
|---|---|---:|---:|---:|---:|---:|
| kernel / sleep | N0 | 0.25 | 2.95 / 3.43 / 5.82 | 0 | 3.57 / 4.36 / 6.42 | 0 |
| kernel / sleep | N1 | 0.70 | 3.04 / 3.55 / 5.61 | 0 | 3.67 / 4.41 / 6.55 | 0 |
| kernel / sleep | D0 | 0.55 | 3.46 / 4.04 / 6.76 | 0 | 4.05 / 4.86 / 7.89 | 0 |
| kernel / busy | N0 | 0.00 | 0.84 / 1.00 / 5.75 | 0 | 0.97 / 1.26 / 5.97 | 0 |
| kernel / busy | N1 | 0.01 | 0.84 / 1.00 / 5.93 | 0 | 0.98 / 1.25 / 6.14 | 0 |
| kernel / busy | D0 | 0.01 | 0.92 / 1.08 / 1.58 | 0 | 1.06 / 1.35 / 2.17 | 0 |
| captured / sleep | N0 | 0.03 | 19.56 / 19.73 / 29.23 | 17 | 20.50 / 21.03 / 311.25 | 7,998 |
| captured / sleep | N1 | 0.03 | 19.62 / 19.83 / 28.83 | 23 | 20.54 / 21.12 / 279.49 | 8,000 |
| captured / sleep | D0 | 0.02 | 19.72 / 19.89 / 25.78 | 28 | 20.65 / 21.13 / 293.64 | 8,000 |

- 단위는 ms다. captured의 주기 초과는 토크 출력 기준 0.2–0.4%, DOB 완료 기준 거의 전부다.
- captured에서 방식별로 한 번씩 나온 280–311 ms 튐의 원인은 규명하지 않았다.
- captured는 매 주기 밀리므로 대기가 거의 없어 깨어남 지연이 0에 가깝다.
- kernel에서 plant 모사와 진단을 포함한 loop 전체는 N0/N1의 첫 step에서 한 번 20 ms를
  넘었다(최대 25.09 ms). 제어 계산 자체는 모든 step에서 주기 안이다.
- 커밋 `645c447` 메시지의 "sensor-to-DOB latency, 7.12 ms"는 깨어남 지연과 입출력 전송을
  뺀 값이며 명칭도 잘못되었다. 이 표로 정정한다.

연속 실행 시 MPPI solve만은 19.10 ms에서 0.45 ms로 줄었다. 충돌 kernel은 모든 분기를
강제로 계산해도 1.08 ms(통상 0.28–0.37 ms)다.

동일성 검증:

- 동역학 식은 float64에서 1e-12 이내로 일치한다(`tests/test_scalar_codegen.py`).
- 충돌 비용은 680,000개 자세에서 누락·오검출이 없었다(1 초과 비용 기준, `equivalence_20261123/collision_kernel.json`).
- 168회 재실행의 실행별 RMSE 차이는 최대 0.0004 mm이고 10.3–10.4절 값은 모두 같다
  (`kroc/results/aligned_circle_kernel_20261122/`).

주장 범위는 이 PC의 시뮬레이션에서 20 ms 주기 안에 제어 계산이 끝난다는 것이다.
Linux는 실시간 OS가 아니므로 hard real-time 보장이 아니며, 로봇 통신 지연도 포함하지
않는다. 비-payload MPPI 경로와 장애물 조건에는 이 backend가 적용되지 않는다.

### 10.8 K, H, λ 민감도

K=136, H=10, λ=64는 이 분석 **이전에** 정해진 값이다. 아래는 사후 민감도 확인이며 선정
절차가 아니다. 평가와 겹치지 않는 seed 7307/7317/7327을 사용했다(RMSE mm, N0 / N1 / D0).
η와 ESS는 45 Nm N0의 중앙값이다.

| 설정 | 무외란 | 45 Nm | 47.5 Nm | η | ESS |
|---|---:|---|---|---:|---:|
| **K=136, H=10, λ=64** | 2.39 | 5.51 / 3.77 / 3.42 | 9.06 / 5.58 / 4.38 | 5.1 | 10.6 |
| K=256 | 2.40 | 5.71 / 3.85 / 3.49 | 9.59 / 5.70 / 4.50 | 8.1 | 19.2 |
| K=512 | 2.39 | 6.05 / 4.00 / 3.59 | 10.41 / 6.01 / 4.71 | 14.1 | 38.3 |
| K=1024 | 2.39 | 5.84 / 3.91 / 3.55 | 10.52 / 6.01 / 4.73 | 25.2 | 75.9 |
| K=2048 | 2.39 | 5.85 / 3.92 / 3.56 | 10.72 / 6.06 / 4.76 | 46.9 | 150.8 |
| H=6 | 무효 | 무효 | 무효 | - | - |
| H=8 | 1.66 | 무효 | 무효 | - | - |
| H=9 | 1.83 | 3.94 / 2.81 / 2.63 | 각 2/3만 유효 | 7.1 | 15.7 |
| H=11 | 3.16 | 5.78 / 4.45 / 4.10 | 11.97 / 7.51 / 5.61 | 3.9 | 7.4 |
| H=12 | 4.00 | 6.11 / 5.08 / 4.75 | 13.17 / 8.83 / 6.48 | 3.0 | 5.1 |
| H=16 | 8.13 | 8.75 / 8.49 / 8.68 | 13.48 / 11.56 / 10.15 | 1.4 | 1.7 |
| λ=32 | 2.45 | 6.33 / 4.42 / 3.79 | 12.12 / 7.26 / 5.24 | 2.5 | 4.1 |
| λ=80 | 2.36 | 4.85 / 3.48 / 3.24 | 8.13 / 5.17 / 4.21 | 6.6 | 14.6 |
| λ=96 | 2.34 | 4.55 / 3.38 / 3.18 | D0 2/3만 유효 | 8.1 | 18.3 |
| λ=128 | 2.31 | 5.25 / 4.05 / 3.77 | 무효 | 10.4 | 24.6 |

- 무외란 RMSE는 K에 대해 수렴했다. **47.5 Nm N0는 K=136에서 수렴하지 않았다**
  (K=2048에서 +18%). K를 바꿔도 45, 47.5 Nm 모두 N0 > N1 > D0 순서는 유지된다.
- H=16, 45 Nm에서는 세 방식의 차이가 사라진다. 순서는 H와 λ 전 범위에서 보장되지 않는다.
- η 5–10(Pezzato et al., Biased-MPPI)에 현재 값은 45 Nm에서 들지만 47.5 Nm에서는
  4.4–4.8로 벗어난다. 해당 문헌의 K는 136과 다르므로 참고 범위로만 쓴다.
