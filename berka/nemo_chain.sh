set -u
export NEMO_TELEMETRY_ENABLED=false
export VLLM_NO_USAGE_STATS=1
export HF_HUB_DISABLE_TELEMETRY=1
B="$HOME/berka-saif"
S="$B/Safe-Synthesizer/berka/nemo_step.sh"
case "${1-}" in
  1|2|3) T="run$1" ;;
  dp1|dp2|dp3) T="dprun${1#dp}" ;;
  *) echo "give the run, 1, 2, 3, dp1, dp2 or dp3"; exit 1 ;;
esac
echo "CHAIN START $T $(TZ=America/Toronto date +%H:%M:%S)"
if bash "$S" train "$T" && bash "$S" split "$T" && CLAVA_RUN=clavaddpm RTF_RUN=realtabformer NEMO_RUN="nemo_$T" RESULTS_NAME="comparison_results_nemo_$T.json" RUN_TAG="score_nemo_$T" bash "$B/run_nemo.sh" 04_compare=berka-ref && bash "$S" check "$T"; then
  echo "CHAIN DONE $T $(TZ=America/Toronto date +%H:%M:%S)"
else
  echo "CHAIN STOPPED $T $(TZ=America/Toronto date +%H:%M:%S)"
  exit 1
fi
