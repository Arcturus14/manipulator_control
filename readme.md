# UR5 Manipulator Control Simulation

UR5 6축 매니퓰레이터의 Cartesian 궤적 추종 성능을 시뮬레이션하고, **MPPI(Model Predictive Path Integral)** 와 **비선형 SQP(Sequential Quadratic Programming)** 제어기를 비교하는 프로젝트입니다. 외란, DOB(Disturbance Observer), 이동 장애물을 각각 켜고 끌 수 있으며 실행 결과를 수치 데이터, 정적 그래프, 3D 애니메이션으로 저장·표시합니다.

![최종 궤적 예시](docs/images/final_trajectory.png)

## 주요 기능

- UR5의 6자유도 비선형 강체 동역학 시뮬레이션
- GPU 가속 MPPI와 acados 기반 Full-SQP 제어
- 원(circle), 8자(infinity), 사각형(rectangle), 3D 파동(complex_3d_wave), 다중 스케일 3D(multiscale_3d) 말단 궤적 추종
- 정현파 토크 외란과 1차 Q-filter DOB 보상
- 기준 궤적 반대편에서 4.6배속으로 추격하는 10 mm 동적 정육면체 또는 기준 궤적에
  균등 배치되는 10 mm 정적 정육면체 장애물
- 단일 조건 실행 또는 다음 두 조건 비교
  - MPPI vs SQP
  - DOB OFF vs DOB ON
- 평균 오차, RMSE, 표준편차, 평균/최악 계산 시간 산출
- NPZ 결과 저장, PNG 그래프 생성 및 이전 그래프 자동 보관

## 모델과 제어기

시뮬레이션 플랜트와 두 제어기는 동일한 UR5 파라미터와 다음 동역학을 사용합니다.

```text
M(q)qdd + C(q, qd)qd + g(q) + Bqd = tau
```

- **MPPI**: `pytorch_mppi`를 이용해 토크 시퀀스를 샘플링하고, 말단 위치 추종·속도·입력·입력 변화·장애물 비용을 최소화합니다. rollout 동역학은 `torch.compile`로 컴파일되며 CUDA가 있으면 자동으로 GPU를 사용합니다.
- **SQP**: CasADi로 비선형 OCP를 구성하고 acados의 Full-SQP 및 HPIPM QP solver로 풉니다. SQP 플랜트 계산은 CPU `float64`를 사용합니다.
- **DOB**: 측정 가속도와 nominal inverse dynamics로 외란 토크를 추정하고 제어 입력에 보상값을 더합니다.

기본 시뮬레이션은 제어 주기 20 ms, 총 10초(500 step), 예측 horizon 10으로 설정되어 있습니다. 세부 물리 파라미터와 비용 가중치는 [`config.py`](config.py)에서 변경할 수 있습니다.

## 프로젝트 구조

```text
.
├── config.py                 # 모델, 시뮬레이션, 제어기 및 비용 파라미터
├── config.txt                # 실행 조건 설정
├── execute.sh                # config.txt의 mode에 맞는 진입점 실행
├── run.py                    # 단일 제어기 시뮬레이션
├── run_comparison.py         # 두 조건 실행 및 결과 비교
├── controllers/
│   ├── mppi_dob_controller.py
│   └── sqp.py
├── dynamics/                 # UR5 순/역동역학 및 plant integration
├── kinematics/               # DH 기반 FK, Jacobian, IK
├── costs/                    # 추종 및 장애물 비용
├── observers/                # DOB 구현
├── utils/                    # 설정 파서, 궤적, 외란 생성기
├── visualization/            # 그래프와 3D 애니메이션
├── c_generated_code_ur5_sqp/ # acados 자동 생성·컴파일 결과
└── data/
    ├── npz/                  # 조건별 원시 결과
    ├── png/                  # 이전 그래프의 timestamp별 보관본
    └── *.png                 # 가장 최근 실행의 그래프
```

## 실행 환경

현재 스크립트는 **Python 3.10**, PyTorch/CUDA 환경과 로컬 acados 소스 트리를 전제로 작성되어 있습니다.

### Python 패키지

가상환경을 만든 뒤 프로젝트 의존성을 설치합니다.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MPPI 실행에는 최소한 NumPy, SciPy, Matplotlib, PyTorch, `pytorch_mppi`가 필요합니다. `requirements.txt`는 개발 환경 전체를 고정한 목록이므로 설치 환경과 CUDA 버전에 맞는 PyTorch wheel이 필요한 경우 먼저 PyTorch를 별도로 설치하십시오.

### SQP 추가 의존성

SQP를 사용하려면 CasADi와 빌드가 완료된 acados가 추가로 필요합니다. 현재 코드에는 개발 머신의 다음 경로가 기본값으로 들어 있습니다.

```text
/home/arclab-04/acados
/home/arclab-04/miniconda3/envs/mppi/lib/python3.10/site-packages
```

다른 머신에서는 다음 항목을 로컬 환경에 맞게 수정해야 합니다.

- [`execute.sh`](execute.sh)의 `PYTHON_BIN`과 기본 `ACADOS_SOURCE_DIR`
- [`controllers/sqp.py`](controllers/sqp.py)의 `_CASADI_SITE`, `_ACADOS_PYTHON`, 기본 `ACADOS_SOURCE_DIR`
- 실행 전 `LD_LIBRARY_PATH=${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH}` 설정

`execute.sh`를 사용하지 않고 현재 가상환경의 Python으로 직접 실행할 수도 있습니다.

## 실행 설정

[`config.txt`](config.txt)의 모든 키는 필수이며 알 수 없는 키나 중복 키가 있으면 실행이 중단됩니다.

| 키 | 값 | 설명 |
|---|---|---|
| `mode` | `single`, `comparison` | 단일 실행 또는 두 조건 비교 |
| `comparison_type` | `controller`, `dob` | 제어기 비교 또는 DOB ON/OFF 비교 |
| `controller` | `mppi`, `sqp` | 단일 실행 제어기 또는 DOB 비교의 공통 제어기 |
| `use_dob` | `true`, `false` | 단일 실행 또는 제어기 비교에 공통 적용할 DOB 설정 |
| `use_disturbance` | `true`, `false` | 외란 적용 여부 |
| `T_SIM` | 양의 실수, `DT`의 정수배 | 기준 궤적 1회 주행 및 전체 시뮬레이션 시간 [s] |
| `USE_OBSTACLES` | `Dynamic`, `Static`, `False` | 이동 구 장애물, 균등 배치된 10 mm 정적 정육면체, 또는 장애물 미사용 |
| `STATIC_OBSTACLE_COUNT` | 1 이상의 정수 | 시작점을 제외하고 궤적 진행률에 균등 배치할 정적 장애물 수 |
| `trajectory` | `circle`, `infinity`, `rectangle`, `complex_3d_wave`, `multiscale_3d` | 기준 말단 궤적 |

Boolean 값에는 `true/false` 외에 `yes/no`, `y/n`, `1/0`, `on/off`도 사용할 수 있습니다.

`N_STEPS = T_SIM / DT`로 계산되므로 `DT`를 유지한 채 `T_SIM`을 늘리면 같은
기준 궤적을 더 많은 step에 걸쳐 천천히 주행합니다. 정적 장애물은 설정한 개수를
유지하면서 전체 궤적 진행률에 균등 배치되며 reference index 0에는 생성되지 않습니다.

정적 장애물 MPPI는 CUDA 완료 동기화 기준 20 ms 제어주기에 맞춘 별도 기본값을 사용합니다:
예측 horizon 9, sample 112, 장애물 표면 안전 여유 2 mm, soft zone 4 mm입니다.
SQP는 MPPI와 독립된 장벽 파라미터를 사용합니다. 정적 장애물에서는 국소 SQP가
장애물 정면에서 멈추는 대칭 해를 피하기 위해 원래 기준 궤적 주변에 부드러운 측방
유도 궤적을 내부적으로 생성합니다. 저장되는 오차는 계속 원래 기준 궤적에 대해
계산되며, 내부 유도 궤적은 NPZ의 `sqp_guidance_waypoints`에 함께 저장됩니다.
재검증에는 `benchmark_static_mppi.py --cases tuned_realtime`을 사용할 수 있습니다.

### 설정 예시

MPPI와 SQP를 동일한 외란 및 장애물 조건에서 비교하는 예시입니다.

```ini
mode = comparison
comparison_type = controller
controller = mppi
use_dob = true
use_disturbance = true
T_SIM = 10.0
USE_OBSTACLES = Dynamic
STATIC_OBSTACLE_COUNT = 4
trajectory = infinity
```

`comparison_type = controller`이면 `controller` 값은 비교 대상 결정에 사용되지 않으며 MPPI와 SQP를 모두 실행합니다. `comparison_type = dob`이면 지정한 `controller`에 대해 DOB OFF와 ON을 각각 실행하고, 이때 `use_dob` 값은 비교 대상 결정에 사용되지 않습니다.

## 실행 방법

### 설정 파일 기반 실행

```bash
chmod +x execute.sh
./execute.sh
```

`execute.sh`는 `config.txt`의 `mode`를 읽어 `run.py` 또는 `run_comparison.py`를 선택합니다.

현재 Python 환경을 직접 사용하려면 다음과 같이 실행합니다.

```bash
# config.txt의 mode가 single일 때
python run.py --config config.txt

# config.txt의 mode가 comparison일 때
python run_comparison.py --config config.txt
```

설정 파일 없이 `python run.py`를 실행하면 제어기, 외란, DOB, 장애물, 궤적을 터미널에서 대화형으로 선택할 수 있습니다.

MPPI 튜닝값은 여러 난수 조건에서 다음과 같이 paired benchmark할 수 있습니다.

```bash
python benchmark_mppi.py \
  --seeds 7 17 27 37 47 \
  --trajectory infinity \
  --output /tmp/mppi_multiseed_infinity.json
```

각 seed에서 기존 설정과 현재 튜닝 설정을 동일한 난수 조건으로 실행하고, 추종 오차의 평균·표준편차, 계산시간, 20 ms deadline 위반률과 토크 포화율을 출력합니다.

첫 MPPI 실행에서는 Torch/Inductor kernel 컴파일 때문에 초기 시간이 오래 걸릴 수 있습니다. 컴파일된 Inductor/Triton kernel은 `.torch_cache/`에 보관되어 이후 별도 프로세스에서도 재사용되며, 이 디렉터리는 Git에서 제외됩니다. 그래프나 실행 환경이 바뀌면 Torch가 새 cache key로 필요한 kernel만 다시 컴파일합니다. 커널 준비는 제어 루프 진입 전에 완료되며 성능 통계에는 모든 제어 step이 포함됩니다.

SQP의 acados 생성 코드와 공유 라이브러리는 `c_generated_code_ur5_sqp/`에 보관되며 역시 Git에서 제외됩니다. 실행 시 현재 OCP와 캐시의 호환성을 검사해 일치하면 바로 재사용하고, 모델·horizon·비용·제약 등이 달라졌으면 자동으로 다시 생성하고 컴파일합니다. 실행 종료 시 Matplotlib 창이 열리고 `plt.show()`에서 대기하므로 GUI가 없는 환경에서는 적절한 Matplotlib backend 설정이 필요합니다.

## 결과

각 실행은 다음 형식으로 결과를 저장합니다.

```text
data/npz/{controller}_dist_{on|off}_dob_{on|off}_obs_{dynamic|static|false}_{trajectory}_results.npz
```

NPZ에는 관절 상태(`q`), 말단 위치(`ee`), 입력 토크(`u`), 추종 오차(`err`, `err_mm`), 외란 추정/실제값(`d_hat`, `d_true`), DOB residual, step 계산 시간(`solve_t`) 및 실행 조건이 포함됩니다.

가장 최근 실행의 그래프는 다음 위치에 생성됩니다.

- `data/final_trajectory.png`: 기준 및 실제 3D 말단 궤적
- `data/timeseries_ee_error.png`: 시간별 추종 오차와 step 계산 시간
- `data/metrics_dashboard.png`: 추종 오차 및 계산 시간 비교

새 실행 전에 기존 최상위 PNG는 `data/png/<timestamp>/`로 자동 이동하므로 과거 결과가 보존됩니다. `c_generated_code_ur5_sqp/`는 SQP solver 생성 과정에서 갱신될 수 있습니다.

## 주요 설정값

| 항목 | 기본값 |
|---|---:|
| 시뮬레이션 시간 / 제어 주기 | 10 s / 0.02 s |
| 총 step | 500 |
| MPPI/SQP horizon | 10 |
| MPPI sample 수 | 136 |
| MPPI temperature `lambda` | 0.2 |
| 기준 궤적 중심 | `[0.40, 0.00, 0.35]` m |
| 기준 궤적 반경 | 0.13 m |
| 관절 속도 제한 | ±6 rad/s |
| DOB cutoff 계수 `alpha` | 40 |

토크 제한은 1~3번 관절 ±150 Nm, 4~6번 관절 ±28 Nm입니다. 물리·제어 파라미터를 변경할 때는 MPPI와 SQP가 모두 [`config.py`](config.py)의 값을 공유한다는 점을 고려하십시오.
