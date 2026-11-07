set -u
export PATH="$HOME/.local/bin:$PATH"
export NEMO_TELEMETRY_ENABLED=false
export VLLM_NO_USAGE_STATS=1
export HF_HUB_DISABLE_TELEMETRY=1
B="$HOME/berka-saif"
N="$B/nemo"
V="$B/nemo-venv"
C="$B/Safe-Synthesizer/berka"
W="$B/synthetic-data-bootcamp/implementations/berka_comparison/work"
STEP="${1-}"
TAG="${2-}"
case "$TAG" in
  smoke) CFG="$C/berka-smollm3-smoke.yaml"; EXP="$N/smoke/export_check" ;;
  run1|run2|run3) CFG="$C/berka-smollm3-nodp-noeval.yaml"; EXP="$W/nemo_$TAG/export" ;;
  dpsmoke) CFG="$C/berka-smollm3-dp-smoke.yaml"; EXP="$N/dpsmoke/export_check" ;;
  dprun1|dprun2|dprun3) CFG="$C/berka-smollm3-dp-noeval.yaml"; EXP="$W/nemo_$TAG/export" ;;
  *) echo "give the step, then smoke, run1, run2, run3, dpsmoke, dprun1, dprun2 or dprun3"; exit 1 ;;
esac
say() { echo "$1 nemo_$STEP $TAG nemo-venv $(TZ=America/Toronto date +%H:%M:%S)${2-}"; }
say START
case "$STEP" in
  train)
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    if [ "$U" -gt 2048 ]; then say FAILED ", the GPU already holds $U MiB"; exit 1; fi
    if [ -e "$N/$TAG" ]; then say FAILED ", $N/$TAG exists"; exit 1; fi
    cd "$N" && "$V/bin/safe-synthesizer" run --config "$CFG" --data-source "$N/berka_train_joined.csv" --run-path "$N/$TAG" --output-file "$N/$TAG/synthetic.csv" > "$N/nemo_train_$TAG.log" 2>&1
    ;;
  split)
    set -- "$B"/nemo-wheel/nemo_safe_synthesizer-*.whl
    if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then say FAILED ", expected one wheel in $B/nemo-wheel"; exit 1; fi
    "$V/bin/python" "$C/split_back_berka.py" "$N/$TAG/synthetic.csv" "$EXP" > "$N/nemo_split_$TAG.log" 2>&1 && "$V/bin/python" "$C/nemo_cost.py" "$N/$TAG" "$EXP" "$TAG" "$CFG" "$1" >> "$N/nemo_split_$TAG.log" 2>&1
    ;;
  check)
    python3 "$C/nemo_check.py" "$W/comparison_results_pgu_orig.json" "$W/comparison_results_nemo_$TAG.json" > "$N/nemo_check_$TAG.txt" 2>&1
    ;;
  *) say FAILED ", the step is train, split or check"; exit 1 ;;
esac
R=$?
if [ "$R" -eq 0 ]; then say END; else say FAILED ", see the step's log in $N"; exit 1; fi
