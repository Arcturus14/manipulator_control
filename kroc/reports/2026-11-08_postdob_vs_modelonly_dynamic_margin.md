# Post-DOB와 model-only의 동일 동적 상태 마진 비교

## 목적

동일한 headroom 기반 동적 상태 마진을 사용하면서 DOB 보상의 위치만 바꿔
Post-DOB와 model-only를 비교했다. 이 실험은 동적 마진의 유무 비교가 아니라,
동적 마진이 적용된 두 DOB 구조의 비교다.

## 구현 변경

기존 `KrocTracker`는 margin용 외란으로 model-only rollout 변수만 읽었기 때문에
Post-DOB에서는 동적 마진에 들어가는 외란이 0이었다. margin용 외란을 DOB의
`d_hat`에서 직접 읽도록 변경했다. rollout에 외란을 넣는지는 기존 DOB 구조에
따라 결정되며, margin 생성기는 두 구조에서 같은 추정값을 사용한다.

두 구조 모두 torque headroom 기준은 확정 설계와 동일하게 unclamped
`U_center = U_ff + R`로 두고 `d_hat`을 다시 더하지 않았다. 따라서 이번 실험은
margin 규칙까지 동일한 통제 비교다. 다만 Post-DOB는 MPPI 출력 뒤에 `d_hat`을
더하므로, 이 기준이 실제 최종 actuator headroom을 정확히 나타내지는 않는다.

## 실험 조건

- GPU: NVIDIA GeForce GTX 1650
- trajectory: circle, 250 steps, `dt=0.02 s`
- seeds: 7, 17, 27
- MPPI: `H=10`, `K=136`, `temperature=1`, temporal noise `rho=0.8`
- state-limit weights: position/velocity 각각 50,000
- margin: headroom, 최소 전체 범위 2.5%/side, 최대 10%/side,
  반응시간 0.1 s
- nominal: 외란 없음, 원래 torque limit
- stress: 1.24 s부터 4배 step 외란, 원래 torque limit
- limited: 1.24 s부터 2배 step 외란, torque limit 40%

총 18회 CUDA 실행을 완료했다. 저장된 raw state에는 clamp가 없으며 audit에서
18/18 run의 CUDA 실행, `d_hat` 전달, headroom/gain 재계산, margin 구간,
raw 상태 위반 집계를 검증했다.

## 평균 결과

| Scenario | 구조 | RMSE [mm] | 위치 위반 step | 속도 위반 step | 최대 속도 [rad/s] | 포화 step | solve mean [ms] |
|---|---|---:|---:|---:|---:|---:|---:|
| nominal | Post-DOB | 3.592 | 0 | 0 | 1.970 | 0.0 | 17.381 |
| nominal | model-only | 3.748 | 0 | 0 | 1.685 | 0.0 | 17.371 |
| stress | Post-DOB | 4.250 | 0 | 0 | 2.206 | 0.0 | 17.456 |
| stress | model-only | 701.033 | 88.0 | 147.7 | 11.954 | 0.0 | 17.449 |
| limited | Post-DOB | 5.145 | 0 | 0 | 1.945 | 60.0 | 17.414 |
| limited | model-only | 20.372 | 0 | 0 | 2.298 | 71.7 | 17.444 |

## 해석

nominal에서는 두 구조의 차이가 작다. 강한 외란에서는 Post-DOB가 세 seed 모두
낮은 추종 오차와 상태 제약 만족을 유지한 반면, model-only는 세 seed 모두
폐루프 상태가 크게 발산했다. limited에서도 두 구조 모두 상태 위반은 없었지만
Post-DOB의 평균 RMSE가 약 74.7% 낮고 포화 step도 평균 11.7회 적었다.

solve 시간 차이는 0.03 ms 이내라 구조 선택에 따른 유의미한 계산시간 차이는
관찰되지 않았다. 두 구조 모두 margin 계산을 포함한 평균 solve 시간이 약
17.4 ms다.

stress에서 model-only의 실제 actuator saturation이 0인데도 발산했다. 따라서
실패를 actuator limit 부족만으로 설명할 수 없으며, 현재 residual warm-start와
rollout 외란 모델의 결합 및 DOB 추정 지연을 별도로 진단해야 한다.

## 주장 범위와 추가 결정

이 결과만으로 “동적 마진 때문에 Post-DOB가 좋아졌다”고 주장할 수 없다. 양쪽에
같은 동적 마진을 적용했기 때문에 확인된 것은 현재 설정에서 Post-DOB 구조가
model-only 구조보다 강건했다는 사실이다. 동적 마진의 순효과는 각 구조 안에서
fixed margin과 다시 비교해야 한다.

Post-DOB 전용 headroom을 향후 어떻게 정의할지도 결정해야 한다. 현재 통제 비교는
확정식 `U_ff + R`을 유지했지만, 실제 최종 명령 기준의 별도 후보는
`U_ff + R + d_hat`이다. 이는 model-only에 적용하면 중복 가능성이 있으므로,
공통 정의를 바꾸기보다 Post-DOB 전용 ablation으로 분리하는 것이 적절하다.

## 산출물

- 실행 코드: `kroc/benchmark_postdob_vs_modelonly.py`
- 검증 코드: `kroc/audit_postdob_vs_modelonly.py`
- raw 결과 및 그림: `kroc/results/postdob_vs_modelonly_20261108_030119_468672/`
- 자동 요약: 위 결과 폴더의 `REPORT.md`, `aggregate.json`, `audit.json`
