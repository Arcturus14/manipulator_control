# DGX Spark용 MPPI 충돌 최적화 실행 묶음

이 디렉터리 전체만 Spark로 옮기면 됩니다. 원본 저장소나 x86 가상환경을
추가로 복사할 필요가 없도록 필요한 소스·형상·입력 데이터를 포함했습니다.
**Spark 실기기 및 ARM64 Docker 빌드는 아직 검증하지 않았습니다.**

## 빠른 실행: Docker 권장

Spark에서 Docker와 NVIDIA Container Toolkit이 준비되어 있어야 합니다.
드라이버는 호스트 것을 사용하며 이 묶음은 드라이버를 설치·변경하지 않습니다.

```bash
tar -xzf spark_bundle_20261112.tar.gz
cd spark_bundle
bash docker_run.sh build
bash docker_run.sh smoke
bash docker_run.sh micro
bash docker_run.sh closedloop
```

전체 검증을 연속 실행하려면 다음 명령을 사용합니다.

```bash
bash docker_run.sh all --seeds 7
# 다중 seed 비교가 필요할 때만 실행
bash docker_run.sh closedloop --seeds 7 17 27
```

각 단계의 자세한 출력과 오류는 `results/<UTC시각>_<단계>/*.log`에 남습니다.
실패하면 즉시 중단하며 CPU fallback 결과를 만들지 않습니다.

## 포함 파일

- `Dockerfile`, `requirements.txt`, `install_dependencies.py`: 환경 구성.
- `docker_run.sh`, `run.py`: 실행 진입점.
- `preflight.py`: 파일 무결성, ARM64 선택 검사, 실제 CUDA 연산,
  Triton 컴파일·수치 검사, `torch.linalg.solve` 확인 및 버전 기록.
- `source/`: 필요한 Python 소스의 독립 스냅샷. 원본 저장소와 분리됩니다.
- `source/kroc/assets/ur5_cb3_collision/`: 공식 UR5 CB3 충돌 메시와 파라미터.
- `source/kroc/results/`: 기존과 동일한 작은 입력 NPZ 1개와 자세 후보 JSON만 포함.
- `reference/`: GTX 1650의 microbenchmark·폐루프·단일 solve 검증 JSON.
- `bundle_manifest.json`: 포함 파일 목록과 SHA-256 해시.

라이브러리 **바이너리 자체는 동봉하지 않았습니다**. x86 라이브러리를 재사용하지
않고 Spark에서 ARM64 버전을 설치합니다. 따라서 최초 빌드에는 인터넷이 필요합니다.
전체 실험 데이터, GUI, SQP/acados 환경, 가상환경, Torch/Triton 캐시는 포함하지 않습니다.

## 환경 선택과 버전 주의

기본 이미지는 NVIDIA Spark playbook에서 사용하는
`nvcr.io/nvidia/pytorch:25.11-py3`입니다. ARM64 플랫폼으로 빌드·실행합니다.
이 이미지의 CUDA PyTorch/Triton은 유지하며 보조 패키지만 설치합니다.
설치 스크립트는 기존 Torch/Triton 버전을 constraints로 보호합니다.
NVIDIA 기본 pip constraints와 충돌하면 이를 무시하지 않고 설치 오류로 중단합니다.

추가 라이브러리는 PyTorch MPPI, python-fcl, trimesh, NumPy, SciPy,
Matplotlib, PyYAML입니다. `python-fcl` ARM wheel이 없으면 빌드할 수 있도록
FCL·OctoMap·CCD 개발 라이브러리와 C++ 빌드 도구를 Dockerfile에 넣었습니다.

기본 이미지의 PyTorch는 2.10.0a0 계열이고 GTX 기준 측정은 2.13.0+cu130입니다.
따라서 기본 이미지와의 성능 비교에는 **하드웨어와 프레임워크 버전 차이가 함께**
포함됩니다. GPU만의 차이를 분리하려면 양쪽의 PyTorch/CUDA/Triton 버전을 맞춰
GTX 기준도 다시 측정해야 합니다. 임의로 `pip install torch`를 실행하지 마세요.

다른 공식 ARM64 이미지를 쓰려면 호스트 드라이버 호환성을 먼저 확인하고 지정합니다.

```bash
KROC_NGC_IMAGE=nvcr.io/nvidia/pytorch:26.10-py3 bash docker_run.sh build
```

컨테이너 이미지는 용량이 크므로 소스 압축파일 크기와는 별개로 다운로드 공간이
필요합니다. `pip check`가 실패하면 빌드 로그에서 해당 의존성을 먼저 확인하세요.

## 이미 호환되는 네이티브 환경이 있다면

ARM64·Blackwell 대응 CUDA PyTorch/Triton이 먼저 설치되어 있어야 합니다.
FCL wheel 대신 source build가 필요하면 OS에 `build-essential`, `cmake`,
`pkg-config`, `python3-dev`, `libfcl-dev`, `liboctomap-dev`, `libccd-dev`가 필요합니다.

```bash
python3 install_dependencies.py
python3 run.py smoke --require-arm64
python3 run.py all --require-arm64
```

## 실행 단계

| 명령 | 내용 |
|---|---|
| `check` | 무결성·CUDA·Triton smoke test와 환경 JSON |
| `smoke` | check + 기존 17개 단위 테스트 |
| `micro` | 원본/융합 Torch/전용 Triton 충돌 평가 비교, 초기 컴파일 제외 |
| `validate` | 동일 CUDA 난수·상태·입력 중심의 MPPI 단일 solve 비교 |
| `closedloop` | 기존 / Triton / Triton+전체 컴파일의 8초 폐루프 비교 |
| `all` | 위 검증과 벤치마크 전체 |

폐루프 조건은 기존과 같은 K=136, H=10, dt=20 ms, post-DOB, 센서 노이즈,
정적 2.5% / 동적 2.5~5% / 정적 5%, 원궤적 R=4 cm·주기 4초입니다.
3번 관절 외란은 1초부터 0.2초 동안 -135 Nm로 증가한 뒤 종료까지 유지합니다.
충돌 형상·비용 수식은 바꾸지 않았고 **headroom은 사용하지 않습니다**.

## 결과 확인

- `results/.../environment.json`: GPU, CUDA capability, 설치 버전, nvidia-smi.
- `results/.../micro/results.json`: 수치 차이, round별 평가 시간, 소스·입력 해시.
- `results/.../solve_validation/results.json`: 단일 solve의 토크·상태·비용 차이.
- `results/.../comparison/results.json`: 평균/p95/p99/max, 20 ms 초과율,
  추종 RMSE, 위치 위반 및 메시 충돌 검출 수.
- 각 비교군 하위 디렉터리: manifest, metrics, 실제 궤적 NPZ.

첫 전체 rollout 컴파일은 오래 걸릴 수 있습니다. GTX에서는 약 101초였으며
Spark의 소요 시간은 미확인입니다. `source/.torch_cache/`에 대상 GPU용 캐시가
새로 생성됩니다. GTX 캐시를 Spark로 가져오지 마세요.

초기 컴파일은 제어 루프 밖에서 수행하고, 시간 통계에서 최초 5개 solve도 제외합니다.
비교해야 하는 것은 **같은 조건의 warm solve**입니다. 전체 시뮬레이션 루프 시간에는
플랜트 RK4·진단도 들어가므로 MPPI solve 시간과 구분해야 합니다.
단일 seed 결과나 평균 20 ms 미만만으로 실시간 안전성을 주장할 수는 없습니다.

정확한 수식은 같아도 GPU·라이브러리의 부동소수점 차이 때문에 장시간 궤적이
완전히 같지 않을 수 있습니다. 검증 실패 시 tolerance를 임의로 늘리지 말고
`environment.json`과 오류 로그를 함께 확인하세요.

## 공식 참고 자료

- [NVIDIA Spark PyTorch playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-pytorch-fine-tune/README.md)
- [NVIDIA PyTorch 25.11 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html)
- [NVIDIA NGC PyTorch 이미지](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)
- [UR5 충돌 형상 원본](https://github.com/ros-industrial/universal_robot/tree/noetic-devel/ur_description)

이 스냅샷은 생성 시점 버전으로 고정됩니다. 원본 코드를 변경해도 자동 동기화되지
않습니다. 파일 수정은 무결성 검사를 실패시키므로, 변경 후에는 새 묶음을 생성하세요.
