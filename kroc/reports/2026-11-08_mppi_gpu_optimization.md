# MPPI GPU 프로파일과 complete rollout-loop compile 최적화

## 목적

현재 torque-based MPPI의 GPU 사용량과 병목을 확인하고, Im et al.의
torque-sampling MPPI와 계산 구조를 비교해 수학적 설정을 바꾸지 않는 범위에서
solve 시간을 줄였다.

참고 논문:
https://arxiv.org/html/2609.02020v1

## 논문과 현재 프로젝트의 조건

| 항목 | Im et al. | 현재 프로젝트 |
|---|---:|---:|
| Robot | FR3, 7 DoF | UR5, 6 DoF |
| GPU | RTX 2070 Super | GTX 1650 |
| Rollouts K | 128 | 136 |
| Horizon T | 30 | 10 |
| Rollout dt | 0.006 s | 0.02 s |
| Prediction window | 0.18 s | 0.20 s |
| Solver rate/time | 약 166 Hz / 6 ms | baseline 약 67 Hz / 15 ms |
| 구현 | tailored custom CUDA dynamics | PyTorch/Inductor + batched RNEA |

논문은 각 rollout을 GPU thread에 할당하고 robot constants를 constant memory에
두는 custom CUDA dynamics solver를 사용한다. 현재 코드는 mass matrix와 bias를
얻기 위해 rollout step마다 7개 RNEA scenario를 batch로 평가하고 `bmm` 및
`torch.linalg.solve`를 사용한다. 따라서 K와 horizon만으로 시간을 직접 비교하면
안 된다.

NVIDIA 공개 사양상 RTX 2070 Super는 2,560 CUDA cores, GTX 1650은 896 CUDA
cores다. GPU 자체의 처리량 차이도 논문과의 시간 차이에 크게 기여한다.

NVIDIA source:
https://www.nvidia.com/en-us/geforce/graphics-cards/compare/

## Baseline profile

- continuous MPPI-only solve: 15.0675 ms
- closed-loop solve mean/p95: 14.909 / 15.151 ms
- active GPU utilization: 78--79%
- active memory utilization: 3%
- process memory observed by nvidia-smi: 300--312 MiB
- PyTorch peak allocated/reserved: 9.48 / 44 MiB
- active power: 약 58 W

메모리는 병목이 아니다. horizon의 각 step에서 compiled dynamics와 FK/cost를
Python loop로 호출하면서 GPU 사이에 launch gap이 남아 있었다. CUDA profiler에서
동역학의 `bmm` 계열이 대부분의 GPU 시간을 사용했다.

## 적용한 최적화

`pytorch_mppi`의 deterministic rollout loop 전체를 `torch.compile(fullgraph=True)`
대상으로 만들었다. reference step이 매 제어 주기 바뀌어도 graph가 재컴파일되지
않도록 `(H+1,3)` fixed-address reference buffer를 만들고, 각 주기 graph 실행 전에
내용만 in-place 복사한다.

옵션은 `compile_rollout_loop=True`이며 기본값은 아직 `False`다. 현재 최적화 경로는
tracking-only MPPI에 한정하고 obstacle cost가 있으면 명시적으로 거부한다.

## 수치 검증

동일 state, nominal sequence, CUDA random state에서 baseline과 compiled solve를
비교했다.

- action: exact match
- optimized nominal sequence: exact match
- perturbed actions: exact match
- rollout states: exact match
- MPPI weights: exact match
- total cost maximum absolute difference: 4.8828125e-4 (float32)

따라서 dynamics와 MPPI update 식을 바꾸지 않았다.

## 폐루프 결과

Circle 250 steps, seed 7, Post-DOB, disturbance OFF, dynamic margin OFF,
`K=136`, `H=10`, temperature 1, temporal-noise rho 0.8 조건이다.

| Condition | RMSE [mm] | solve mean [ms] | p95 [ms] | max [ms] | 20 ms miss |
|---|---:|---:|---:|---:|---:|
| baseline | 3.616 | 14.909 | 15.151 | 18.180 | 0 |
| compiled rollout | 3.591 | 11.494 | 11.573 | 15.169 | 0 |

- speedup: 1.297x
- solve-time reduction: 22.91%
- compiled graph initial warm-up: 67.51 s (baseline 10.68 s)

추종 차이는 stochastic closed loop의 경로 차이 범위이며 두 조건 모두 같은 수준이다.
단일 solve 수치 검증에서는 control output과 state가 일치했다.

## Optimized GPU utilization

- continuous MPPI-only solve: 11.4627 ms
- active GPU utilization: 95%
- active memory utilization: 4%
- process memory: 258--270 MiB
- PyTorch peak allocated/reserved: 9.42 / 22 MiB
- active power: 약 63 W

GPU utilization 증가와 solve-time 감소를 함께 확인했다. 다만 compiled profile에서도
`bmm`가 self CUDA time의 약 80%이고, 한 solve에 417개의 `bmm` 호출이 남아 있다.
fullgraph compile은 launch queue를 더 연속적으로 공급하지만 rigid-body dynamics의
연산 자체를 줄이지는 않는다.

## 결론과 다음 최적화

현재 PyTorch 구현에서도 약 23% 단축할 수 있다. 그러나 논문의 약 6 ms에
접근하려면 다음 단계가 필요하다.

1. six-link forward dynamics를 custom CUDA/Triton kernel로 구현
2. 7-scenario RNEA mass-matrix 구성 대신 직접 ABA 또는 fused CRBA/solve 검토
3. dynamics, semi-implicit Euler, FK, running cost를 rollout thread 안에서 fuse
4. robot constants를 constant/read-only memory에 배치
5. K=128 동일 조건과 RTX 2070 Super 이상 GPU에서 hardware-normalized 비교

custom dynamics는 수치·동역학 검증 범위가 크므로 이번 변경에는 포함하지 않았다.
현재 compiled 경로는 초기 컴파일 67 s와 obstacle 미지원 때문에 실험 옵션으로
유지한다. graph cache/사전 컴파일과 obstacle cost 통합을 완료한 뒤 기본값 전환을
결정하는 것이 안전하다.

## 산출물

- implementation: `controllers/mppi_dob_controller.py`
- runner option: `experiment_runner.py`
- profiler: `kroc/profile_mppi_gpu.py`
- numerical validation: `kroc/validate_compiled_rollout.py`
- closed-loop benchmark: `kroc/benchmark_compiled_rollout.py`
- audit: `kroc/audit_compiled_rollout.py`
- utilization summary: `kroc/results/gpu_utilization_20261108.json`
- closed-loop raw result: `kroc/results/compiled_rollout_20261108_122831_240638/`
