# MPPI 비용함수 구성

이 문서는 현재 저장소의 UR5 MPPI 구현에서 실제로 계산되는 비용을 설명한다.
대상은 `controllers/mppi_dob_controller.py`, `costs/tracking_cost.py`,
`costs/obstacle_cost.py`와 Conda `manipulator` 환경의 `pytorch-mppi 0.9.1`이다.

## 1. 기호와 sampling

- $K$: control trajectory sample 수. 일반 조건 기본값은 136이다.
- $T$: prediction horizon. 일반 조건 기본값은 10이다.
- $n_u=6$: UR5 관절 torque 입력 차원이다.
- $U_t\in\mathbb{R}^6$: horizon 단계 $t$의 nominal input이다.
- $\epsilon_{k,t}\in\mathbb{R}^6$: sample $k$, 단계 $t$의 Gaussian noise이다.
- \(u_{k,t}\): torque limit을 적용한 sampled actuator command이다.
- \(x_{k,t}=[q_{k,t},\dot q_{k,t}]\): rollout state이다.
- \(p(q)\): forward kinematics로 계산한 end-effector 위치이다.

각 MPPI solve는 다음 shape의 새로운 noise를 생성한다.

\[
\epsilon\in\mathbb{R}^{K\times T\times6},\qquad
\epsilon_{k,t}\sim\mathcal N(0,\Sigma).
\]

따라서 모든 horizon 단계에 동일한 noise를 반복해서 사용하지 않는다. Sample,
horizon 단계, 관절마다 별도의 표준정규 난수를 만든다. 현재 covariance는 diagonal이므로
관절 간 noise도 독립적이다.

\[
\Sigma=\operatorname{diag}
(13.6,23.8,13.6,2.72,2.72,0.85).
\]

위 값은 variance이다. 대응하는 표준편차는 약

\[
(3.69,4.88,3.69,1.65,1.65,0.92)\ \text{N·m}
\]

이다. Sampled command는 다음과 같다.

\[
u_{k,t}=\operatorname{clamp}(U_t+\epsilon_{k,t},u_{\min},u_{\max}).
\]

Clamp 때문에 실제로 비용 계산에 사용되는 noise는 원래 Gaussian noise가 아니라
\(u_{k,t}-U_t\)로 다시 계산된다.

## 2. 한 sample trajectory의 사용자 정의 비용

MPPI library는 각 단계에서 먼저 dynamics를 한 번 진행하고, 그 결과 state에 running
cost를 계산한다. 한 sample $k$의 사용자 정의 비용은 다음과 같다.

\[
J_k^{\mathrm{user}}
=\sum_{t=0}^{T-1}\ell(x_{k,t+1},u_{k,t})
+\phi(x_{k,T},u_{k,0:T-1}).
\]

현재 reference indexing도 구현 그대로 사용한다. Running 단계 $t$에서는 한 단계
진행된 state $x_{t+1}$를 `waypoints[s+t]`와 비교한다. 마지막 $x_T$는 running
cost에서 `waypoints[s+T-1]`와 비교되고, terminal cost에서 다시
`waypoints[s+T]`와 비교된다.

### 2.1 Running tracking cost

\[
J_{\mathrm{pos},k}
=w_p\sum_{t=0}^{T-1}
\left\|p(q_{k,t+1})-p^{\mathrm{ref}}_{s+t}\right\|_2^2.
\]

- $s$는 현재 episode step이다.
- $w_p=1{,}200{,}000$이다.
- Reference는 Cartesian position만 포함한다. End-effector orientation error는 없다.

### 2.2 Actuator torque cost

\[
J_{\mathrm{act},k}
=w_u\sum_{t=0}^{T-1}\|u_{k,t}\|_2^2,
\qquad w_u=0.0005.
\]

이 항은 외란을 뺀 effective torque가 아니라 clamp된 actuator command를 평가한다.
DOB 보상 모드별 차이는 6절에서 설명한다.

### 2.3 Joint velocity cost

\[
J_{\mathrm{vel},k}
=w_v\sum_{t=0}^{T-1}\|\dot q_{k,t+1}\|_2^2,
\qquad w_v=5.0.
\]

이 항은 속도를 작게 만드는 quadratic regularizer이다. Joint velocity limit을 직접
표현하는 hard constraint 또는 limit violation penalty는 아니다.

### 2.4 이전 command와 첫 predicted command의 경계 smoothness

\[
J_{\mathrm{boundary},k}
=w_s\|u_{k,0}-u_{\mathrm{prev}}\|_2^2.
\]

- 장애물 없는 조건은 $w_s=0.25$이다.
- 장애물 조건은 $w_s=0.025$이다.
- 현재 `u_prev`는 직전 MPPI optimizer 출력이다.
- Post-DOB 모드에서는 직전 optimizer 출력 뒤에 더한 \(\hat d\)가 `u_prev`에
  포함되지 않는다. 즉 이 항은 직전 최종 applied actuator torque와의 차이를 정확히
  계산하지 않는다.

### 2.5 Horizon 내부 torque smoothness

\[
J_{\Delta u,k}
=\frac{w_s}{T-1}\sum_{t=0}^{T-2}
\|u_{k,t+1}-u_{k,t}\|_2^2.
\]

구현 위치는 terminal callback 안이지만 terminal state 하나만 평가하는 항이 아니다.
Sampled input sequence의 horizon 내부 모든 torque 변화를 평가한다. $T-1$로 나누므로
horizon 길이가 달라졌을 때 transition 개수만으로 비용이 선형 증가하지 않는다.

### 2.6 Terminal tracking cost

\[
J_{\mathrm{terminal},k}
=w_T\left\|p(q_{k,T})-p^{\mathrm{ref}}_{s+T}\right\|_2^2,
\qquad w_T=3{,}600{,}000.
\]

최종 state에 대해 Cartesian position error를 다시 강하게 평가한다.

### 2.7 장애물 비용

장애물이 활성화되면 running cost와 terminal cost에 end-effector barrier가 추가된다.
Cube 중심을 $c$, half-size를 $h$, end-effector 위치를 $p$라 하면 cube 표면까지의
거리는 다음과 같이 계산한다.

\[
d(p,c)=\left\|\max(|p-c|-h,0)\right\|_2.
\]

Barrier penetration은

\[
z=\max(m+r-d,0)
\]

이고 비용은

\[
J_{\mathrm{obs}}=w_{\mathrm{obs}}\left(e^{k_{\mathrm{obs}}z}-1\right)
\]

이다. 여러 cube가 있으면 모두 더한다. Terminal obstacle cost는 같은 barrier의 4배다.

현재 주요 obstacle 설정은 다음과 같다.

| 조건 | weight | gain | margin | exponential zone | cube size |
|---|---:|---:|---:|---:|---:|
| Static | 5,000 | 500 | 0.002 m | 0.004 m | 0.010 m |
| Dynamic | 1,400 | 38 | 0.026 m | 0.012 m | 0.010 m |

현재 obstacle cost는 end-effector point만 검사한다. Robot link와 self-collision은
검사하지 않는다.

## 3. 사용자 정의 비용의 전체 식

장애물 없는 경우 한 sample의 비용을 합치면 다음과 같다.

\[
\begin{aligned}
J_k^{\mathrm{user}}
=&\;w_p\sum_{t=0}^{T-1}
\|p(q_{k,t+1})-p^{\mathrm{ref}}_{s+t}\|_2^2\\
&+w_u\sum_{t=0}^{T-1}\|u_{k,t}\|_2^2
+w_v\sum_{t=0}^{T-1}\|\dot q_{k,t+1}\|_2^2\\
&+w_s\|u_{k,0}-u_{\mathrm{prev}}\|_2^2\\
&+\frac{w_s}{T-1}\sum_{t=0}^{T-2}
\|u_{k,t+1}-u_{k,t}\|_2^2\\
&+w_T\|p(q_{k,T})-p^{\mathrm{ref}}_{s+T}\|_2^2.
\end{aligned}
\]

장애물이 활성화되면 각 running 단계의 barrier와 4배 terminal barrier를 더한다.

## 4. MPPI library가 추가하는 sampling 보정항

위 비용 외에 `pytorch-mppi`가 importance-sampling control correction 항을 추가한다.
현재 diagonal covariance와 `noise_abs_cost=False` 조건에서는

\[
J_{\mathrm{corr},k}
=\lambda\sum_{t=0}^{T-1}
U_t^\top\Sigma^{-1}\tilde\epsilon_{k,t}
\]

이다. 여기서

\[
\tilde\epsilon_{k,t}=u_{k,t}-U_t
\]

는 clamp 이후 다시 계산한 perturbation이다. 현재 temperature는
\(\lambda=0.24\)이다. MPPI가 weighting에 사용하는 최종 sample cost는

\[
S_k=J_k^{\mathrm{user}}+J_{\mathrm{corr},k}
\]

이다. 이 보정항은 `W_ACT` torque-size cost와 별개다.

## 5. Weight와 nominal sequence update

수치 안정성을 위해 전체 sample의 최소 비용

\[
\beta=\min_k S_k
\]

를 뺀 뒤 weight를 계산한다.

\[
\omega_k=
\frac{\exp(-(S_k-\beta)/\lambda)}
{\sum_i\exp(-(S_i-\beta)/\lambda)}.
\]

Weighted noise로 nominal sequence를 갱신한다.

\[
U_t^{\mathrm{candidate}}
=U_t+\sum_{k=1}^{K}\omega_k\tilde\epsilon_{k,t}.
\]

현재 기본 `mean_update_alpha=1.0`에서는 candidate를 그대로 사용한다. Alpha를
실험적으로 변경한 경우에는

\[
U_t^{\mathrm{new}}
=(1-\alpha)U_t+\alpha U_t^{\mathrm{candidate}}
\]

로 blending한다.

## 6. DOB 보상 모드에 따른 비용의 의미

| 모드 | Sampling center | Prediction effective torque | 실제 actuator command | 비용에서 평가하는 action |
|---|---|---|---|---|
| Post-DOB `output` | \(U^{ff}+R\) | \(u_{sample}\) | \(\operatorname{clamp}(u_{MPPI}+\hat d)\) | \(u_{sample}\) |
| Model-only `rollout` | \(U^{ff}+R\) | \(u_{sample}-\hat d\) | \(\operatorname{clamp}(u_{MPPI})\) | \(u_{sample}\) |
| Nominal-shift `rollout_nominal_shift` | \(U^{ff}+R+\hat d\) | \(u_{sample}-\hat d\) | \(\operatorname{clamp}(u_{MPPI})\) | \(u_{sample}\) |

Model-only와 nominal-shift에서는 비용의 torque 항이 rollout에 전달되는 effective
torque가 아니라 actuator command를 평가한다. 이는 actuator effort를 평가한다는
점에서 물리적으로 일관된다.

Post-DOB에서는 optimizer 뒤에 더하는 \(\hat d\)가 rollout과 action/smoothness cost에
포함되지 않는다. 따라서 Post-DOB가 평가한 torque cost와 실제 actuator command의
cost가 다를 수 있으며, optimizer도 후보상 후의 saturation을 직접 예측하지 않는다.

모든 aware rollout에서 \(\hat d\)는 한 solve가 시작될 때 snapshot하고 horizon 전체에
같은 값으로 적용한다. Noise는 horizon 단계마다 독립이므로 이 동작과 구분해야 한다.

## 7. 현재 포함되지 않은 항과 제약

현재 MPPI tracking cost에는 다음 항이 없다.

- Joint position limit penalty
- Joint velocity limit violation penalty
- Cartesian workspace limit penalty
- Robot link와 self-collision penalty
- End-effector orientation tracking cost
- DOB 추정 불확실성 또는 \(\hat d\) 변화 예측 cost

Actuator torque는 sample과 최종 command에 clamp되지만, 외란을 차감한 effective torque는
actuator 출력이 아니므로 다시 clamp하지 않는다. Joint velocity quadratic cost는
존재하지만 velocity limit constraint를 대신하지 않는다.

## 8. 구현 위치

| 내용 | 파일 |
|---|---|
| MPPI 구성, DOB rollout, nominal sequence | `controllers/mppi_dob_controller.py` |
| Tracking, velocity, torque, smoothness | `costs/tracking_cost.py` |
| Static/dynamic obstacle barrier | `costs/obstacle_cost.py` |
| 가중치와 covariance 기본값 | `config.py` |
| Sampling, correction cost, weighting | 설치된 `pytorch_mppi` package의 `mppi.py` |

이 문서는 MPPI 비용을 대상으로 한다. iLQR과 SQP는 유사한 물리적 항을 사용하지만
optimizer별 가중치와 구현 방식이 다르므로 이 식과 수치를 그대로 적용하면 안 된다.
