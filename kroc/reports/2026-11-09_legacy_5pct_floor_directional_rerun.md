# 기존 5% soft constraint 기반 동적 마진 재실험

## 변경한 비교 기준

직전 zero-floor/softplus 비교를 다음 native MPPI 기준으로 교체해 별도로 실행했다.

- basic: 기존 5% full-span soft boundary + ReLU squared
- dynamic: 기존 5%를 최소 마진으로 유지하고 정지거리 항을 더해 5–10%로 변화
- conservative static: 항상 10% full-span 축소
- headroom: OFF
- 모든 비교군: ReLU squared, 기존 5% soft-zone 폭으로 정규화

센서 노이즈, DOB 필터, 외란 시나리오, paired seeds는 직전 실험과 동일하다.

## 동적 경계 및 비용

기존 soft-zone 폭을 `m0`라 하면 target margin은 다음과 같다.

```text
m_dq,target^+ = m_dq,0 + ReLU(a_tilde) Tr
m_dq,target^- = m_dq,0 + ReLU(-a_tilde) Tr

m_q,target^+ = m_q,0^+ + ReLU(dq) Tr
                 + 0.5 ReLU(a_tilde) Tr^2
m_q,target^- = m_q,0^- + ReLU(-dq) Tr
                 + 0.5 ReLU(-a_tilde) Tr^2
```

target은 5–10% full span으로 clip한 뒤 증가 beta=1.0, 감소 beta=0.1로
시간 필터링한다. 최종 soft boundary는 `q_max-m_q^+`, `q_min+m_q^-`,
`dq_max-m_dq^+`, `-dq_max+m_dq^-`이다.

비용의 분모는 동적 마진 크기로 바꾸지 않고 기존 5% soft-zone 폭 `m0`를
유지한다. 따라서 경계가 이동할수록 동일한 상태에서 더 큰 비용을 부과한다.

## 조건

- GPU: GTX 1650, CUDA only
- seeds: 17, 27, 37, 47, 57
- 2 scenarios × 2 architectures × 3 constraints × 5 seeds = 60 runs
- q noise 0.001 rad, dq noise 0.01 rad/s
- acceleration LPF 10 Hz, exact-ZOH DOB 8 Hz
- joint-wise P99 acceleration deadband
- Tr=0.1 s, state hard clamp OFF

## 위치 외란 결과

| 구조 | 제약 | RMSE [mm] | 위치 위반 [step] | 속도 위반 [step] | 최소 위치 여유 [rad] |
|---|---|---:|---:|---:|---:|
| Post-DOB | native 5% | 3.190 ± 0.573 | 0 | 0 | 0.2505 |
| Post-DOB | dynamic 5–10% | 3.141 ± 1.221 | 0 | 1.6 ± 0.9 | 0.6994 |
| Post-DOB | static 10% | 5.643 ± 2.686 | 0 | 0 | 0.7166 |
| Model-only | native 5% | 104.422 ± 37.475 | 60.8 ± 9.0 | 19.8 ± 2.6 | -2.5514 |
| Model-only | dynamic 5–10% | 113.686 ± 37.887 | 52.4 ± 11.2 | 18.0 ± 2.5 | -2.0647 |
| Model-only | static 10% | 130.604 ± 61.672 | 46.4 ± 12.6 | 18.8 ± 2.9 | -1.6123 |

Post-DOB에서는 기존 5%만으로도 위치 위반이 이미 0이었다. 동적 방식은 최소 위치
여유를 0.2505 rad에서 0.6994 rad로 늘렸지만, 더 강한 회피 동작 때문에 평균
1.6 step의 속도 위반이 새로 발생했다. 정적 10%는 비슷한 위치 여유를 만들었지만
RMSE가 5.643 mm로 증가했다.

## 속도 외란 결과

| 구조 | 제약 | RMSE [mm] | 속도 위반 [step] | 최대 속도 초과 [rad/s] |
|---|---|---:|---:|---:|
| Post-DOB | native 5% | 1.881 ± 0.284 | 0.8 ± 1.3 | 0.0331 ± 0.0490 |
| Post-DOB | dynamic 5–10% | 1.803 ± 0.162 | 0 | 0 |
| Post-DOB | static 10% | 1.831 ± 0.211 | 0 | 0 |
| Model-only | native 5% | 12.282 ± 7.194 | 8.0 ± 0.7 | 1.4639 ± 0.4075 |
| Model-only | dynamic 5–10% | 14.651 ± 5.821 | 7.8 ± 0.8 | 1.4781 ± 0.3842 |
| Model-only | static 10% | 17.419 ± 7.824 | 7.2 ± 0.4 | 1.2393 ± 0.3105 |

Post-DOB에서는 동적 5–10%가 native 5%에 남아 있던 작은 속도 위반을 제거했고,
정적 10%보다 약간 낮은 RMSE를 보였다. Model-only에서는 어느 마진도 외란 보상
부족을 해결하지 못했다.

## 판단

- 기존 soft constraint에 동적 마진을 더한다는 설계 해석은 구현과 일치한다.
- Post-DOB 속도 시나리오에서는 동적 5–10%의 효과가 확인됐다.
- Post-DOB 위치 시나리오에서는 native 5%가 이미 물리 위반을 막았으므로 동적
  마진의 주효과는 추가 안전거리 확보이며, 속도 위반 trade-off가 나타났다.
- 따라서 위치·속도 마진 gain을 동일하게 1로 두는 현재 설정은 재조정할 여지가 있다.

## 재현 파일

- benchmark: `kroc/benchmark_simple_directional_architectures.py`
- result: `kroc/results/legacy_floor_directional_architectures_20261109_multiseed/`
