# KRoC 학부생 논문 실험

2026-10-31부터 새로 작성한 구현, benchmark, 보고서와 데이터의 저장 위치다.
기존 실험 자료는 이동하거나 복사하지 않는다. 공통 controller/runner에는
실험 subclass 주입과 solve 준비 hook 및 제어 주기 시간 기록만 추가한다.

## 확정한 설계

- Model-only, `U_center = U_ff + R`; headroom에 d_hat을 추가하지 않는다.
- unclamped center의 horizon 최소 positive/negative headroom 사용.
  범위를 초과한 계획은 negative headroom으로 보존한다.
- 상한 마진은 negative headroom, 하한 마진은 positive headroom으로 확대한다.
  이는 joint-wise approximation이며 coupled braking feasibility를 보장하지 않는다.
- `a_d_hat = -solve(M(q), clip(d_hat))`. rollout에는 원래 d_hat을 사용한다.
- 마진은 solve마다 갱신하며 horizon 전체에서 고정한다.
- 증가 즉시 반영(beta_up=1), 감소는 beta_down=0.1로 필터링한다.
- target clip/구간 보호 후 필터, 다시 구간 보호를 적용한다.
- running 및 terminal에 동일한 비대칭 위치/속도 penalty를 적용한다.
- 모든 방법에서 비용 정규화 분모는 기존 90% 경계의 고정 길이로 유지한다.
  동적 마진으로 나누면 마진 증가와 함께 penalty 기울기가 약해지므로 사용하지 않는다.

## 추가 수치 선택: 검증할 초기 가정

| 파라미터 | 초기값 | 의미 |
|---|---:|---|
| 반응시간 Tr | 0.10 s | 5 control periods, 검증된 제동시간은 아님 |
| c_q, c_dq | 1 | 상태 변화 근사 계수 |
| 최소 마진 | 전체 범위 2.5%/side | adaptive methods의 기본 마진 |
| 고정 비교군 | 전체 범위 5%/side | 기존 90% soft boundary와 일치 |
| 최대 마진 | 전체 범위 5%, 10%, 15%/side | sweep 가능 |
| 최소 내부 구간 | 전체 범위 50% | 상·하한 역전 방지 |
| r_safe | 관절별 양의 torque 한계 20% | half input range의 20% |
| k_tau | 1 | headroom 확대 계수 |
| margin용 d_hat cap | 관절별 torque 한계 100% | rollout estimate에는 미적용 |

분율의 분모는 **전체 물리 범위**다. 예: 속도 범위 [-pi,pi]에서 5%는
0.1pi rad/s이고 고정 상한은 0.9pi다. Position도 q_max-q_min을 사용한다.
최소 마진과 고정 마진의 차이도 성능에 영향을 준다. 후속 ablation에서는
min_fraction=fixed_fraction으로 맞춘 비교를 추가해야 한다.
이번 pilot의 adaptive 최소 2.5%는 덜 보수적인 계획을 탐색하기 위한 초기값이다.
따라서 fixed 대비 차이를 순수한 동적 조절 효과로 해석하지 않는다.

## 비교와 실행

```bash
/home/arclab-04/miniconda3/envs/manipulator/bin/python -m unittest -v kroc.test_state_margin tests.test_state_limits
MPLCONFIGDIR=/tmp/matplotlib-kroc /home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.benchmark_state_margin --seeds 7 17 27 --caps .10
# 최대 마진 sweep
MPLCONFIGDIR=/tmp/matplotlib-kroc /home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.benchmark_state_margin --seeds 7 17 27 --caps .05 .10 .15
# 기본 마진까지 맞춘 통제 비교 (아직 실행하지 않은 후속 실험)
MPLCONFIGDIR=/tmp/matplotlib-kroc /home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.benchmark_state_margin --seeds 7 17 27 --caps .05 .10 .15 --min-fraction .05 --fixed-fraction .05
# 저장 데이터 독립 검사 및 mean/median/min/max/std 집계
/home/arclab-04/miniconda3/envs/manipulator/bin/python -m kroc.audit_results kroc/results/20261031_220157_697646
# 동일한 headroom 동적 마진에서 Post-DOB와 model-only 비교
MPLCONFIGDIR=/tmp/matplotlib-kroc /home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.benchmark_postdob_vs_modelonly --seeds 7 17 27
# 동적 마진 없이 clean/noisy measurement Post-DOB 비교
MPLCONFIGDIR=/tmp/matplotlib-kroc /home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.benchmark_postdob_sensor_noise --seeds 7 17 27
# 별도 validation seed에서 clean/noisy legacy/noisy filtered 직접 비교
MPLCONFIGDIR=/tmp/matplotlib-kroc /home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.benchmark_exact_dob_cutoff --seeds 37 47 57 --cutoffs 8
# 공정한 5% baseline의 clean/noise-aware 동적 마진과 제한 접근 시나리오
MPLCONFIGDIR=/tmp/matplotlib-kroc /home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.benchmark_noise_aware_state_margin --seeds 7 --scenarios nominal position_limit velocity_limit
# horizon 전체 rollout compile의 수치 검증, 프로파일 및 폐루프 A/B
/home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.validate_compiled_rollout
/home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.profile_mppi_gpu --solves 400 --profile-solves 10 --compile-rollout-loop
/home/arclab-04/miniconda3/envs/manipulator/bin/python -u -m kroc.benchmark_compiled_rollout --seeds 7
```

별도 validation seed의 same-run DOB 비교와 이후 연구 순서는
`kroc/reports/2026-11-08_noise_aware_dob_validation.md`에 기록했다.

Post-DOB 비교에서도 margin용 외란은 rollout 변수가 아니라 DOB의 `d_hat`에서
직접 읽는다. 따라서 Post-DOB와 model-only 모두 같은 외란 추정으로 마진을 만든다.
공정한 구조 비교를 위해 torque headroom 기준도 둘 다 unclamped `U_ff + R`이며
`d_hat`을 다시 더하지 않는다. 단, Post-DOB는 최종 출력에서 `d_hat`을 더하므로
실제 actuator 포화는 model-only와 달라질 수 있으며 saturation 지표로 함께 평가한다.

비교군: fixed, directional, headroom. 모든 방법은 같은 seed, rho=0.8,
temperature=1, H=10, K=136, cost weights=50000을 사용한다.
학습/tuning/검증 seed를 분리한 최종 연구 결과가 아닌 초기 pilot이다.
고정 비교군은 cap sweep에서도 동일한 5%다.

시나리오: nominal(외란 0), stress(4x step, 원래 torque limits),
limited(2x step, torque limits 40%). limited는 포화를 유도하는 인위적
actuator derating 실험이며 실제 UR5 사양 변경 결과로 해석하지 않는다.
각 실행은 250 steps/5초, step 외란은 floor(steps/4)에서 시작한다.

결과는 `kroc/results/<timestamp>/`에 manifest, metrics, run별 NPZ, 비교 그래프,
REPORT.md로 저장한다. CPU fallback은 거부한다. 중단되면 저장된 run의
metrics/NPZ를 사용할 수 있지만 완성 REPORT는 생성되지 않을 수 있다.
기본 .gitignore에 의해 NPZ/PNG는 git에서 제외되지만 디스크에는 보존된다.

## Clamp와 평가 규칙

현재 rollout은 semi-implicit Euler, plant는 RK4이며 둘 다 상태 clamp가 없다.
`qd_raw == qd`는 unclamped plant 적분 결과를 뜻한다. 테스트는 effective torque
400, dt=.02에서 두 전이 모두 8 rad/s를 그대로 내는지 검사한다.
Actuator torque clamp만 유지한다. 20ms 샘플 사이 peak는 별도로 보장하지 않는다.

위반 steps는 하나 이상의 관절이 실제 물리 경계를 초과한 샘플 수이며,
관절별 횟수, 최대 위반량, 누적 위반량(sum_j,t violation*dt), 최소 signed
distance, 내부 경계 거리도 기록한다. Raw qd, unclipped rollout d_hat_used,
clipped d_hat_margin, a_d_hat, 방향별 headroom/gain/margin, horizon center를 보존한다.
Saturation은 실제 출력의 torque limit 도달로 측정하며 sample clamp fraction과
구분한다. Tracking, torque rate(Nm/s), solve 및 전체 simulation control cycle
시간을 기록한다. Solve 시간은 margin과 GPU 로그 복사를 포함하고 전체 cycle은
DOB/plant/CPU 로그도 포함한다. Compilation warmup 및 최초 5 samples는 시간
통계에서 제외한다. 실시간 하드웨어 보장으로 해석하지 않는다.

## 주장 한계

Soft penalty, 추정 지연, local M(q), joint-wise braking 근사 때문에 제약 만족을
보장하지 않는다. 이미 회복 불가능한 상태/torque 조건은 margin으로 해결되지 않는다.
최종 논문에는 포화 여부, 물리적 회복 가능성, 추종 trade-off, 다중 seed의
mean/median/worst-case를 함께 제시해야 한다. Tube MPC는 이번 구현에서 제외한다.
