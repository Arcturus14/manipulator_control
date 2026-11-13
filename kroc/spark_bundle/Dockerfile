# NVIDIA's Spark playbook explicitly supports this ARM64 image.
# Override NGC_IMAGE only after checking target driver/CUDA compatibility.
ARG NGC_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${NGC_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config python3-dev libfcl-dev liboctomap-dev libccd-dev \
    && apt-get clean
COPY requirements.txt install_dependencies.py /opt/kroc-spark/
RUN python3 /opt/kroc-spark/install_dependencies.py
ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1
WORKDIR /bundle
CMD ["bash"]
