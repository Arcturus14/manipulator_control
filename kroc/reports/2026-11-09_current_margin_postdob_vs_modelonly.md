# 현재 동적 마진: Post-DOB와 Model-only 비교

## 결론

현재의 noise-aware directional margin을 두 DOB 구조에 동일하게 적용한 결과,
위치 상한 접근에서는 두 구조 모두 고정 5% 마진보다 여유 거리를 늘렸다. 반면
속도 pulse에서는 Post-DOB만 물리 속도 위반을 제거했고, Model-only는 모든 seed에서
7~8 step의 위반이 남았다. 따라서 현재 설정에서는 동적 마진과 결합할 기본 구조로
Post-DOB가 더 적합하다.

이 결과는 Model-only라는 일반적 방법 전체의 열등성을 뜻하지 않는다. 현재의 짧은
horizon, 샘플 수, 반응형 soft cost, DOB 지연을 그대로 둔 구현에 대한 결과이다.

## 실험 조건

- GPU MPPI: `K=136`, `H=10`, 100 simulation steps
- paired seeds: 47, 57, 67, 77, 87
- 센서 노이즈: position 0.001 rad, velocity 0.01 rad/s
- DOB: acceleration LPF 10 Hz, exact-ZOH cutoff 8 Hz
- 동적 마진 deadband: zero-disturbance에서 보정한 joint-wise P99
- 비교 마진: fixed 5%, directional 5~10%, reaction time 0.1 s
- state hard clamp 없음; 물리 상태의 위반을 직접 집계
- 위치 시나리오: elbow가 상한으로 이동, -10 Nm 연속 외란
- 속도 시나리오: wrist-3 초기 속도 +2.5 rad/s, -16 Nm ramp/hold pulse

## 위치 상한 접근 결과

값은 5개 seed의 mean ± sample standard deviation이다. 상한 거리는 클수록 안전하다.

| 구조 | 마진 | elbow 상한 거리 [rad] | tracking RMSE [mm] | 물리 위치 위반 |
|---|---|---:|---:|---:|
| Post-DOB | fixed 5% | 0.3202 ± 0.0133 | 4.028 ± 0.243 | 0 |
| Post-DOB | directional | 0.3506 ± 0.0028 | 3.922 ± 0.238 | 0 |
| Model-only | fixed 5% | 0.3238 ± 0.0037 | 3.988 ± 0.643 | 0 |
| Model-only | directional | 0.3666 ± 0.0075 | 3.530 ± 0.328 | 0 |

Paired directional 효과:

- Post-DOB: 상한 거리 +0.0304 rad (+9.48%), RMSE -0.106 mm
- Model-only: 상한 거리 +0.0428 rad (+13.22%), RMSE -0.458 mm

두 구조 모두 위치 위험 방향의 마진 확대가 실제 궤적을 상한에서 멀어지게 했다.
다만 모든 조건에서 원래부터 물리 위반이 0이므로, 이 실험은 “위반 방지”가 아니라
“제약 위험 거리 증가”를 입증한다.

## 속도 pulse 결과

| 구조 | 마진 | wrist-3 peak [rad/s] | excess area [rad/s·step] | 위반 step | tracking RMSE [mm] |
|---|---|---:|---:|---:|---:|
| Post-DOB | fixed 5% | 3.1045 ± 0.1228 | 0.0298 ± 0.0413 | 0.6 ± 0.9 | 1.823 ± 0.279 |
| Post-DOB | directional | 2.8630 ± 0.0698 | 0 | 0 | 2.033 ± 0.379 |
| Model-only | fixed 5% | 4.5040 ± 0.3164 | 7.2164 ± 1.8391 | 7.8 ± 0.4 | 12.709 ± 7.416 |
| Model-only | directional | 4.3836 ± 0.3383 | 6.4256 ± 1.8733 | 7.8 ± 0.4 | 17.723 ± 13.347 |

Paired directional 효과:

- Post-DOB: peak -0.2416 rad/s, excess area -0.0298, 위반 step 0.6→0,
  RMSE +0.210 mm
- Model-only: peak -0.1204 rad/s, excess area -0.7908이지만 위반 step은 그대로 7.8,
  RMSE +5.014 mm

같은 directional margin끼리 비교하면 Model-only의 peak는 Post-DOB보다 평균
1.5207 rad/s 높고, excess area는 6.4256, 위반은 7.8 step 더 많았다.

## 해석

seed 47의 대표 궤적에서 Post-DOB peak는 step 3, Model-only peak는 step 8에
나타났다. Directional velocity upper margin은 두 구조 모두 step 0의 0.314 rad/s에서
step 1의 0.545 rad/s, step 2 이후 0.628 rad/s로 빠르게 증가했다. 따라서 이번
Model-only 실패는 “마진이 갱신되지 않아서”만은 아니다.

현재 Model-only는 추정 외란을 rollout plant에만 넣고 actuator 출력에는 직접 더하지
않는다. 이 조건에서 짧은 MPPI horizon과 soft cost가 pulse 동안 필요한 제동 토크를
충분히 선택하지 못했고, peak 전까지 속도가 계속 증가했다. 반면 Post-DOB는 추정값을
출력 보상에도 사용해 실제 제동이 더 빨랐다.

연산시간은 구조 차이가 거의 없었다. Directional 기준 mean solve time은 위치에서
Post-DOB 14.282 ms, Model-only 14.284 ms이고 속도에서 각각 14.320 ms, 14.310 ms였다.
Directional margin 자체는 fixed보다 약 0.15~0.24 ms를 추가했다.

## 판단 및 다음 단계

1. KRoC의 주 비교 구조는 현재로서는 Post-DOB + directional margin을 유지한다.
2. 위치 결과는 위험 거리 증가, 속도 결과는 물리 위반 제거로 구분해 주장한다.
3. Model-only를 다시 후보로 삼으려면 horizon/샘플 수, terminal velocity cost,
   disturbance-aware nominal shift를 별도 튜닝해야 한다. 이는 마진 A/B와 분리된
   controller architecture 개선 실험이어야 한다.
4. Headroom margin은 아직 검증되지 않았으므로 이번 결론에 포함하지 않는다.

## 재현 파일

- benchmark: `kroc/benchmark_current_margin_architectures.py`
- analyzer: `kroc/analyze_current_margin_architectures.py`
- raw/metrics: `kroc/results/current_margin_architectures_20261109_020446_579912/`
