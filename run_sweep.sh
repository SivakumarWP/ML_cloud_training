#!/usr/bin/env bash
# run_sweep.sh (8x GPU, multi-job per GPU, Runpod-ready)
set -euo pipefail

MODEL="resnet18"
# If --jobs is not provided, we compute it as NUM_GPUS * PER_GPU
JOBS=""
PER_GPU=30              # default: 30 jobs per GPU (~2 GB/job fits on 80 GB A100)
ENABLE_MPS=0
FORCE=0
STOP_ON_ERROR=0
OUT_BASE="outputs"
PYBIN="${PYBIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;           # total concurrency (overrides --per-gpu)
    --per-gpu) PER_GPU="$2"; shift 2 ;;     # jobs per GPU (used only if --jobs not given)
    --enable-mps) ENABLE_MPS=1; shift ;;
    --force) FORCE=1; shift ;;
    --stop-on-error) STOP_ON_ERROR=1; shift ;;
    --out-dir) OUT_BASE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Detect local GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
  NUM_GPUS="${#GPU_IDS[@]}"
else
  GPU_IDS=(cpu)
  NUM_GPUS=1
fi

# If user did not pass --jobs, compute total concurrency as NUM_GPUS * PER_GPU
if [[ -z "${JOBS}" ]]; then
  if [[ "${GPU_IDS[0]}" != "cpu" ]]; then
    JOBS="$(( NUM_GPUS * PER_GPU ))"
  else
    JOBS="${PER_GPU}"
  fi
fi

# Optional: enable CUDA MPS (Multi-Process Service) to improve concurrent process throughput
# Ref: NVIDIA MPS docs suggest enabling when a single process underutilizes the GPU.
# This creates control/log dirs user-writable and starts the daemon if not running.
if [[ "${ENABLE_MPS}" -eq 1 && "${GPU_IDS[0]}" != "cpu" ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  if ! pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[MPS] Starting nvidia-cuda-mps-control daemon..."
    nvidia-cuda-mps-control -d
  else
    echo "[MPS] nvidia-cuda-mps-control already running."
  fi
fi

LRs=("1e-4" "5e-5" "1e-6" 1e-8")
WDs=("0.0" "1e-4" "1e-3" "1e-5")
DOs=("0.0" "0.2" "0.5")
LOSSES=("ce" "focal" "bce" "wbce")
FOCAL_GAMMAS=("1.0" "2.0")
WARMUPS=("0")
COSINE=("0" "1")

SUMMARY="${OUT_BASE}/sweep_results.csv"
mkdir -p "${OUT_BASE}"
echo "model,lr,weight_decay,dropout,loss,focal_gamma,warmup_steps,use_cosine,val_acc,test_acc,run_dir" > "${SUMMARY}"

# GPU-aware launcher; assigns GPU by slot index
run_one() {
  local slot="$1" lr="$2" wd="$3" do="$4" loss="$5" gamma="$6" wu="$7" cos="$8"

  # Pick GPU ID for this slot (modulo number of GPUs)
  local gpu_env=""
  if [[ "${GPU_IDS[0]}" != "cpu" ]]; then
    local gid="${GPU_IDS[$(( slot % NUM_GPUS ))]}"
    gpu_env="CUDA_VISIBLE_DEVICES=${gid}"
  fi

  local loss_tag="$loss"
  if [[ "$loss" == "focal" ]]; then loss_tag="${loss}_g${gamma}"; fi
  local cos_tag="cos${cos}"
  local run_dir="${OUT_BASE}/${MODEL}/lr_${lr}__wd_${wd}__do_${do}__loss_${loss_tag}__wu_${wu}__${cos_tag}"
  mkdir -p "${run_dir}"

  if [[ $FORCE -eq 0 && -f "${run_dir}/metrics.txt" ]]; then
    echo "[SKIP] ${run_dir} (metrics.txt exists)"
  else
    local cmd=( ${PYBIN} -u train.py
      --model "${MODEL}"
      --lr "${lr}"
      --weight-decay "${wd}"
      --dropout "${do}"
      --loss "${loss}"
      --warmup-steps "${wu}"
      --out-dir "${OUT_BASE}"
      --early-stop-patience 8
      --batch-size 16
      --epochs 40
      --img-size 224
    )
    if [[ "${loss}" == "focal" ]]; then cmd+=( --focal-gamma "${gamma}" ); fi
    if [[ "${cos}" == "1" ]]; then cmd+=( --use-cosine ); fi

    echo "[RUN][slot ${slot}] ${run_dir} on ${gpu_env:-cpu}"
    ( set -o pipefail; eval "${gpu_env} ${cmd[*]}" > "${run_dir}/train.log" 2>&1 ) || {
      echo "[ERROR] Training failed for ${run_dir}"
      return 99
    }
  fi

  local val_acc=""; local test_acc=""
  if [[ -f "${run_dir}/metrics.txt" ]]; then
    val_acc=$(grep -E "^best_val_acc=" "${run_dir}/metrics.txt" | head -n1 | cut -d'=' -f2 || true)
  else
    val_acc=$(grep -E "val [0-9.]+/([0-9.]+)" -o "${run_dir}/train.log" | tail -n1 | sed -E 's/.*\/([0-9.]+)/\1/' || true)
  fi
  echo "${MODEL},${lr},${wd},${do},${loss},${gamma},${wu},${cos},${val_acc},${test_acc},${run_dir}" >> "${SUMMARY}"
}

export -f run_one
export MODEL OUT_BASE SUMMARY FORCE PYBIN GPU_IDS NUM_GPUS

CMDS_FILE="$(mktemp)"
trap 'rm -f "${CMDS_FILE}"' EXIT

slot=0
for lr in "${LRs[@]}"; do
  for wd in "${WDs[@]}"; do
    for do in "${DOs[@]}"; do
      for loss in "${LOSSES[@]}"; do
        if [[ "${loss}" == "focal" ]]; then
          for g in "${FOCAL_GAMMAS[@]}"; do
            for wu in "${WARMUPS[@]}"; do
              for cos in "${COSINE[@]}"; do
                echo "run_one ${slot} ${lr} ${wd} ${do} ${loss} ${g} ${wu} ${cos}" >> "${CMDS_FILE}"
                slot=$((slot+1))
              done
            done
          done
        else
          g="2.0"
          for wu in "${WARMUPS[@]}"; do
            for cos in "${COSINE[@]}"; do
              echo "run_one ${slot} ${lr} ${wd} ${do} ${loss} ${g} ${wu} ${cos}" >> "${CMDS_FILE}"
              slot=$((slot+1))
            done
          done
        fi
      done
    done
  done
done

echo "[INFO] GPUs detected: ${NUM_GPUS} (${GPU_IDS[*]})"
echo "[INFO] Per-GPU concurrency target: ${PER_GPU}"
echo "[INFO] Total parallel jobs: ${JOBS}"

if command -v parallel >/dev/null 2>&1; then
  echo "[INFO] GNU parallel -j ${JOBS} (GPU pinned per slot)."
  parallel -j "${JOBS}" --halt now,fail=1 < "${CMDS_FILE}" || { [[ "${STOP_ON_ERROR}" -eq 1 ]] && exit 1; }
else
  echo "[WARN] GNU parallel not found; running sequentially."
  while IFS= read -r line; do
    bash -c "${line}" || { [[ "${STOP_ON_ERROR}" -eq 1 ]] && exit 1; }
  done < "${CMDS_FILE}"
fi

echo
echo "Top 5 by val_acc:"
awk -F, 'NR>1 && $9 != "" {print $0}' "${SUMMARY}" | sort -t, -k9,9gr | head -n 5
