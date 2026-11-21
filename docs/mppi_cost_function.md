# MPPI 비용함수 구성

이 문서는 저장소의 UR5 MPPI 구현에서 계산되는 비용을 설명한다.
**1–8절은 이전 legacy 구현의 기록**이며, 당시 대상은
`controllers/mppi_dob_controller.py`, `costs/tracking_cost.py`,
`costs/obstacle_cost.py`와 Conda `manipulator` 환경의 `pytorch-mppi 0.9.1`이었다.
그 절의 horizon별 독립 noise, sampled input clamp, sampling 보정항 및 수치를
현재 기본값에 그대로 적용하면 안 된다.

**현재 기본 `final_n0`와 KROC 최종 N0/N1 비교는 9절을 따른다.**
9절은 2026-11-17 기준으로 최종 연구에 사용한 예상 clipping 손실 비용과 공통 설정을 정리한다.

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

이 항은 속도를 작게 만드는 quadratic regularizer이며, 아래의 limit 접근 penalty와
별개의 항이다.

### 2.4 관절 위치·속도 제한 접근 비용

공식 UR5/CB3 description의 관절 위치 한계는

\[
q_{\max}=(2\pi,2\pi,\pi,2\pi,2\pi,2\pi),\qquad q_{\min}=-q_{\max}
\]

이고, 전 관절 속도 한계는

\[
\dot q_{\max}=\pi\ \mathrm{rad/s}
\]

이다. 각 한계의 $r=0.9$ 지점부터 다음 정규화된 soft penalty를 부과한다.
수치는 Universal Robots의
[`config/ur5/joint_limits.yaml`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/rolling/config/ur5/joint_limits.yaml)을 따른다.
이 파일에서 elbow의 $\pm\pi$는 물리적 회전 가능 범위 전체가 아니라 planning 문제를
피하기 위해 명시적으로 축소한 공식 description 한계다.

\[
J_{q,\lim}=w_{q,\lim}\sum_j\left[
\left(\frac{\max(rq_{j,\min}-q_j,0)}{(1-r)|q_{j,\min}|}\right)^2+
\left(\frac{\max(q_j-rq_{j,\max},0)}{(1-r)q_{j,\max}}\right)^2
\right]
\]

\[
J_{\dot q,\lim}=w_{\dot q,\lim}\sum_j
\left(\frac{\max(|\dot q_j|-r\dot q_{j,\max},0)}
{(1-r)\dot q_{j,\max}}\right)^2.
\]

기본값은 $w_{q,\lim}=w_{\dot q,\lim}=50{,}000$이다. 두 항 모두 running과
terminal state에 적용한다. 상태 전이는 더 이상 $q$ 또는 $\dot q$를 clamp하지 않으며,
실제 한계 초과는 결과의 violation 필드에 기록한다.

### 2.5 이전 command와 첫 predicted command의 경계 smoothness

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

### 2.6 Horizon 내부 torque smoothness

\[
J_{\Delta u,k}
=\frac{w_s}{T-1}\sum_{t=0}^{T-2}
\|u_{k,t+1}-u_{k,t}\|_2^2.
\]

구현 위치는 terminal callback 안이지만 terminal state 하나만 평가하는 항이 아니다.
Sampled input sequence의 horizon 내부 모든 torque 변화를 평가한다. $T-1$로 나누므로
horizon 길이가 달라졌을 때 transition 개수만으로 비용이 선형 증가하지 않는다.

### 2.7 Terminal tracking cost

\[
J_{\mathrm{terminal},k}
=w_T\left\|p(q_{k,T})-p^{\mathrm{ref}}_{s+T}\right\|_2^2,
\qquad w_T=3{,}600{,}000.
\]

최종 state에 대해 Cartesian position error를 다시 강하게 평가한다.

### 2.8 장애물 비용

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
&+\sum_{t=0}^{T-1}(J_{q,\lim,t+1}+J_{\dot q,\lim,t+1})\\
&+w_s\|u_{k,0}-u_{\mathrm{prev}}\|_2^2\\
&+\frac{w_s}{T-1}\sum_{t=0}^{T-2}
\|u_{k,t+1}-u_{k,t}\|_2^2\\
&+w_T\|p(q_{k,T})-p^{\mathrm{ref}}_{s+T}\|_2^2\\
&+J_{q,\lim,T}+J_{\dot q,\lim,T}.
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

- Cartesian workspace limit penalty
- Robot link와 self-collision penalty
- End-effector orientation tracking cost
- DOB 추정 불확실성 또는 \(\hat d\) 변화 예측 cost

Actuator torque는 sample과 최종 command에 clamp되지만, 외란을 차감한 effective torque는
actuator 출력이 아니므로 다시 clamp하지 않는다. 관절 상태는 원래 동역학으로 적분하며
clamp하지 않는다. 위치·속도 제한은 soft penalty이므로 hard feasibility를 수학적으로
보장하지 않으며 실제 초과 여부를 별도로 기록한다.

## 8. 구현 위치

| 내용 | 파일 |
|---|---|
| MPPI 구성, DOB rollout, nominal sequence | `controllers/mppi_dob_controller.py` |
| Tracking, state limits, velocity, torque, smoothness | `costs/tracking_cost.py` |
| Static/dynamic obstacle barrier | `costs/obstacle_cost.py` |
| 가중치와 covariance 기본값 | `config.py` |
| Sampling, correction cost, weighting | 설치된 `pytorch_mppi` package의 `mppi.py` |

이 문서는 MPPI 비용을 대상으로 한다. iLQR과 SQP는 유사한 물리적 항을 사용하지만
optimizer별 가중치와 구현 방식이 다르므로 이 식과 수치를 그대로 적용하면 안 된다.

## 9. 최종 연구 N0/N1: 보상 포함 요청 입력의 예상 clipping 손실 비용

### 9.1 명목 입력, 보상량, 실제 입력의 구분

아래에서는 현재 제어 시점을 $s$, rollout sample을 $k$, horizon 단계를 $t$로 둔다.
$v_{k,t}$는 MPPI가 최적화하는 **명목 유효 토크**다. 1절의 clamp된 sampled
actuator command와 구분한다.

N0/N1 모두 명목 sample과 optimizer 출력 $v$를 물리 토크 한계로 clamp하지 않는다.
Rollout은 다음 명목 모델을 사용한다.

\[
M(q)\ddot q+h(q,\dot q)=v_{k,t}.
\]

$h$는 모델의 bias와 damping을 포함한 항이다. Rollout 상태 전이에
$\hat d$ 또는 actuator saturation을 삽입하지 않는다.
한 solve가 시작될 때의 causal DOB 보상 snapshot $\hat d_s$를 horizon 전체에
고정하고, 각 sample의 보상 후 요청 입력을 다음처럼 평가한다.

\[
a^{\mathrm{req}}_{k,t}=v_{k,t}+\hat d_s.
\]

실제 제어 주기에 적용하는 입력과 plant의 외란 부호는 다음과 같다.

\[
u_{\mathrm{app},s}
=\operatorname{clip}(v_s^*+\hat d_s,u_{\min},u_{\max}),
\qquad
M(q)\ddot q+h(q,\dot q)=u_{\mathrm{app},s}-d_s.
\]

물리 actuator의 최종 clamp는 N0/N1 모두 유지한다.

### 9.2 N1에 추가한 running cost

보상 후 요청 입력 중 물리 clamp 때문에 사라질 부분을

\[
\delta u^{\mathrm{clip}}_{k,t}
=a^{\mathrm{req}}_{k,t}
-\operatorname{clip}
  (a^{\mathrm{req}}_{k,t},u_{\min},u_{\max})
\]

로 정의하고, N1은 각 running 단계에 다음 비용을 추가한다.

\[
\boxed{
\ell_{\mathrm{clip},k,t}
=w_{\mathrm{clip}}
\sum_{j=1}^{6}
\left(\delta u^{\mathrm{clip}}_{k,t,j}\right)^2
}
\]

대칭 토크 한계 $[-\tau_j^{\max},\tau_j^{\max}]$에서는 다음과 같다.

\[
\ell_{\mathrm{clip},k,t}
=w_{\mathrm{clip}}\sum_{j=1}^{6}
\left[\max\left(
|v_{k,t,j}+\hat d_{s,j}|-\tau_j^{\max},0
\right)\right]^2.
\]

- 최종 실험 및 GUI KROC N1 설정은 **$w_{\mathrm{clip}}=1$**이다.
  연구 클래스 생성자의 기본 인자와 달리, 실행 설정에서 이 값을 명시한다.
- 물리 한계 안의 요청 입력에는 추가 비용이 0이다. 85% 또는 95%부터
  시작하는 headroom 비용은 최종 N1에 사용하지 않는다.
- 이 항은 절대 토크 초과량의 **Nm² 기준**이며, 관절별 torque limit으로
  정규화하지 않는다. 이전 정규화 headroom 비용의 가중치와 수치만 비교하면 안 된다.
- 비용 계산의 `clip`은 sample이나 optimizer 출력을 투영하는 연산이 아니다.
- 미래 horizon 단계의 요청 입력에도 비용을 계산하므로, 현재 실제 입력이
  포화되기 전에도 미래 초과 요청을 덜 선호하도록 유도할 수 있다.
  입력 포화 방지나 정확한 추종을 보장하는 항은 아니다.

### 9.3 N0/N1 전체 목적함수에서의 위치

기존 추종·입력·상태·충돌·smoothness 비용을 $J_k^{\mathrm{common}}$이라 하면

\[
J_k^{\mathrm{N0}}=J_k^{\mathrm{common}},
\qquad
\boxed{
J_k^{\mathrm{N1}}
=J_k^{\mathrm{common}}
+\sum_{t=0}^{T-1}\ell_{\mathrm{clip},k,t}
}.
\]

추가 clipping 손실 비용은 running callback에만 들어가며, 별도의 terminal
clipping 비용은 없다. 구현에서는 단계별 비용을 그대로 합산하고 $\Delta t$를
곱하지 않는다. 공통 비용의 주요 항은 다음과 같다.

| 항목 | 최종 연구의 평가 대상·설정 |
|---|---|
| Running / terminal EE 위치 추종 | 가중치 1,200,000 / 3,600,000; orientation 항 없음 |
| 입력 크기 | $0.0005\sum_t\|v_{k,t}\|^2$; 보상 후 요청 입력이 아닌 명목 입력 |
| 관절 속도 | $5\sum_t\|\dot q_{k,t+1}\|^2$ |
| 첫 입력 경계 smoothness | $0.25\|v_{k,0}-v_{\mathrm{prev}}\|^2$ |
| Horizon 내부 smoothness | $\frac{0.25}{T-1}\sum_{t=0}^{T-2}\|v_{k,t+1}-v_{k,t}\|^2$ |
| 관절 위치·속도 제약 비용 | 각각 50,000; full-span 기준 고정 마진 2.5% |
| Link / self / floor 충돌 기하 비용 | weight 50,000, clearance 0.005 m |
| N1 예상 clipping 손실 | $\sum_t\ell_{\mathrm{clip},k,t}$; N0에서는 추가하지 않음 |

Effort와 smoothness 항은 N0/N1 모두 **clamp하지 않은 명목 $v$ 좌표**에 유지한다.
따라서 두 방법의 차이는 추가 clipping 손실 비용이다. 실제 인가 토크의 변화율이나
DOB 보상량의 변화율을 이 smoothness 항이 직접 벌하는 것은 아니다.

위 고정 상태 마진과 충돌 기하 비용은 2.4절의 legacy 90% onset 및 2.8절의
EE-only obstacle barrier와 구분해야 한다. 현재 기본 경로는 link/self 충돌 비용도 포함하므로
7절의 legacy 미포함 항 목록 역시 현재 경로에 그대로 적용되지 않는다.

### 9.4 최종 sampling과 weighting

최종 연구는 sample마다 하나의 6차원 Gaussian perturbation을 뽑아 horizon에 반복한다.

\[
\epsilon_k\sim\mathcal N(0,\Sigma),\qquad
\epsilon_{k,t}=\epsilon_k,\qquad
v_{k,t}=\bar v_t+\epsilon_k.
\]

\[
\Sigma=\operatorname{diag}(9,9,9,0.3136,0.3136,0.3136)\ \mathrm{Nm}^2,
\quad
\sigma=(3,3,3,0.56,0.56,0.56)\ \mathrm{Nm}.
\]

이는 $\sigma_j=0.02\tau_j^{\max}$ 에 대응한다. Noise는 sample간·관절간에는
독립이지만 horizon내에서는 공유한다. 1절의 horizon별 독립 sampling과 다르다.

최종 연구의 sample weight는 **rollout task cost만**으로 계산한다.
4절의 library importance-sampling control correction은 추가하지 않는다.

\[
S_k=J_k^{\mathrm{N0}}\ \text{또는}\ J_k^{\mathrm{N1}},\qquad
\omega_k=
\frac{\exp[-(S_k-\min_i S_i)/64]}
{\sum_i\exp[-(S_i-\min_j S_j)/64]}.
\]

공통 설정은 $K=136$, $T=10$, $\Delta t=0.02$ s, temperature $\lambda=64$,
FF + residual warm start, mean update $\alpha=1$, DOB Q cutoff 3 Hz,
가속도 LPF 6 Hz이다. 기본 GUI `final_n0`는 이 N0 구조를 사용하며,
KROC 비교에서 N1을 선택한 경우에만 위 추가 비용을 활성화한다.

### 9.5 최적화 비용과 결과 지표의 차이

실험 결과의 clipping 누적 손실은

\[
J_{\mathrm{clip}}^{\mathrm{metric}}
=\sum_s
\left\|u_{\mathrm{app},s}-(v_s^*+\hat d_s)\right\|_2\Delta t
\]

이다. 이는 실제 실행 이력에서 산출하는 **Nm·s 단위 평가 지표**이며,
rollout중의 초과 토크 제곱을 합산하는 최적화 비용과 다르다.
또한 clipping 손실 감소가 항상 추종오차 감소로 이어지지는 않는다.

외란·DOB가 없는 경우를 수식상 생각하면 $\hat d_s=0$ 이므로 추가 비용은
$w_{\mathrm{clip}}\|v-\operatorname{clip}(v)\|^2$ 로 줄어든다.
급격한 궤적 전환에 대한 DOB OFF 조건의 효과는 이번 최종 연구의 주 조건으로
검증하지 않았다. 연구용 N1 클래스에는 DOB ON을 요구하는 실행 체크도 있으므로,
이 수식이 현행 N1을 그대로 DOB OFF 실험에 사용할 수 있다는 뜻은 아니다.

### 9.6 구현·설정·최종 보고서

| 내용 | 파일 |
|---|---|
| 추가 비용의 식과 running cost 가산 | `kroc/improve_n1_clipping_cost.py`: `clipping_loss_cost`, `ClippingLossN1._running_cost` |
| 명목 입력 clamp 해제와 공통 입력 비용 보정 | `kroc/input_headroom_study.py`: `InputHeadroomTracker` |
| 최종 비교의 고정 설정 | `kroc/validate_n1_final_fixed.py` |
| GUI 기본 N0 | `controllers/final_n0_controller.py`, `configs/gui_final_n0.yaml` |
| KROC 비교 실행 경로 | `kroc/view_input_headroom.py` |

최종 결과와 Figure 설명은
[최종 보고서](../kroc/final_report_20261116/report.md)를 참조한다.
