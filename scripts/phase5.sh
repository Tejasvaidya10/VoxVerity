#!/usr/bin/env bash
# Phase 5 driver: manifests -> dev slice -> baseline (preprocess, extract,
# score) -> reproduction GATE -> measurements (background) -> pad-repeat and
# vad-all variants -> analysis. Every stage is resumable: a marker file per
# finished stage, and the underlying steps refuse to mix configurations.
#
# Full run (detached):
#   nohup caffeinate -i ./scripts/phase5.sh >> data/logs/phase5.log 2>&1 &
# Dry run on the first N real clips of each manifest (no gate, separate dirs):
#   LIMIT=200 nohup ./scripts/phase5.sh >> data/logs/phase5_dryrun.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
CFG=configs/default.yaml
MODEL=facebook/wav2vec2-xls-r-300m
CKPT=checkpoints/detector_aug.pt
LIMIT=${LIMIT:-}

if [[ -n $LIMIT ]]; then
  ROOT=data/phase5_dryrun; M=$ROOT/manifests; P=$ROOT/processed; F=$ROOT/features
  R=reports/phase5_dryrun; S=$ROOT/state; DEV_N=$(( LIMIT < 7000 ? LIMIT : 7000 )); NBOOT=100
else
  M=data/manifests; P=data/processed; F=data/features
  R=reports/phase5; S=data/phase5_state; DEV_N=7000; NBOOT=1000
fi
mkdir -p "$M" "$P" "$F" "$R" "$S" data/logs data/manifests

log() { echo "[$(date '+%F %T')] $*"; }

stage() {  # stage <name> <command...>
  local name=$1; shift
  if [[ -f $S/$name.done ]]; then log "STAGE $name SKIP (done)"; return 0; fi
  log "STAGE $name START"
  if "$@"; then touch "$S/$name.done"; log "STAGE $name DONE"
  else log "STAGE $name FAILED"; exit 1; fi
}

build_manifests() {
  $PY -m src.data.manifest --dataset asvspoof5 --raw-dir data/raw \
      --protocol data/raw/asvspoof5/ASVspoof5.train.tsv --audio-subdir flac_T \
      --out-name asvspoof5_train_part.csv &&
  $PY -m src.data.manifest --dataset asvspoof5 --raw-dir data/raw \
      --protocol data/raw/asvspoof5/ASVspoof5.dev.track_1.tsv --audio-subdir flac_D \
      --out-name asvspoof5_dev_part.csv &&
  $PY -m src.data.manifest --dataset in_the_wild --raw-dir data/raw
}

limit_manifests() {  # dry run only: first LIMIT clips of each raw manifest
  for m in asvspoof5_train_part asvspoof5_dev_part in_the_wild; do
    head -n $(( LIMIT + 1 )) "data/manifests/$m.csv" > "$M/$m.csv" || return 1
  done
}

preprocess() {  # preprocess <raw manifest stem> <out name> [extra flags...]
  local src=$1 name=$2; shift 2
  $PY -m src.preprocessing.preprocess --manifest "$M/$src.csv" --config $CFG \
      --out-audio-dir "$P/$name" --out-manifest "$M/${name}_chunks.csv" "$@"
}

extract() {  # extract <name>
  $PY -m src.features.extract_features --manifest "$M/${1}_chunks.csv" --out-dir "$F/$1" \
      --model-name $MODEL --device mps --batch-size 16
}

link() {  # link <variant name> <baseline name>
  $PY scripts/link_identical_embeddings.py --variant-manifest "$M/${1}_chunks.csv" \
      --baseline-manifest "$M/${2}_chunks.csv" --baseline-features "$F/$2" \
      --variant-features "$F/$1" --model-name $MODEL
}

score() {  # score <variant label> <dev name> <itw name>
  $PY -m src.eval.harness --checkpoint $CKPT --skip-rationales \
      --manifests "$M/${2}_chunks.csv" "$M/${3}_chunks.csv" \
      --features-dirs "$F/$2" "$F/$3" \
      --scores-out "$R/scores/$1" --out-report "$R/report_$1.json"
}

measure() {
  $PY scripts/characterize.py --manifest "$M/in_the_wild.csv" --dataset in_the_wild --out-dir "$R/measurements" &&
  $PY scripts/characterize.py --manifest "$M/asvspoof5_train_part.csv" --dataset asvspoof5_train --out-dir "$R/measurements" &&
  $PY scripts/characterize.py --manifest "$M/asvspoof5_dev_part.csv" --dataset asvspoof5_dev --out-dir "$R/measurements"
}

baseline_preprocess() { preprocess in_the_wild in_the_wild && preprocess asvspoof5_dev_slice asvspoof5_dev_slice; }
baseline_extract() { extract asvspoof5_dev_slice && extract in_the_wild; }
variant_preprocess() {  # variant_preprocess <variant> <flags...>
  local v=$1; shift
  preprocess in_the_wild "in_the_wild__$v" "$@" && preprocess asvspoof5_dev_slice "asvspoof5_dev_slice__$v" "$@"
}
variant_link() { link "in_the_wild__$1" in_the_wild && link "asvspoof5_dev_slice__$1" asvspoof5_dev_slice; }
variant_extract() { extract "asvspoof5_dev_slice__$1" && extract "in_the_wild__$1"; }

# The manifest builder silently skips files that aren't on disk, so refuse to
# start until the download script has finished extracting both datasets.
for marker in data/raw/asvspoof5/.extracted data/raw/in_the_wild/.extracted; do
  if [[ ! -f $marker ]]; then log "ABORT: $marker missing (download/extraction not finished)"; exit 1; fi
done

log "PHASE5 START (LIMIT=${LIMIT:-none}, reports -> $R)"
stage manifests build_manifests
if [[ -n $LIMIT ]]; then stage limit limit_manifests; fi
stage dev_slice $PY scripts/sample_clips.py --manifest "$M/asvspoof5_dev_part.csv" --n $DEV_N --seed 42 \
      --out "$M/asvspoof5_dev_slice.csv"
stage baseline_preprocess baseline_preprocess

# measurements need only raw audio: run them alongside the GPU work
( stage measurements measure ) & MEASURE_PID=$!

stage baseline_extract baseline_extract
stage baseline_score score baseline asvspoof5_dev_slice in_the_wild

if [[ -z $LIMIT ]]; then
  stage gate $PY scripts/check_gate.py --scores "$R/scores/baseline/in_the_wild_chunks_scores.csv" \
        --reference reports/eval_report_aug.json
else
  log "GATE skipped in dry run"
fi

for v in pad-repeat vad-all; do
  if [[ $v == pad-repeat ]]; then flag=(--pad-mode repeat); else flag=(--vad-mode all); fi
  stage "${v}_preprocess" variant_preprocess "$v" "${flag[@]}"
  if [[ $v == pad-repeat ]]; then stage "${v}_link" variant_link "$v"; fi
  stage "${v}_extract" variant_extract "$v"
  stage "${v}_score" score "$v" "asvspoof5_dev_slice__$v" "in_the_wild__$v"
done

log "waiting for measurements (pid $MEASURE_PID)"
if ! wait $MEASURE_PID; then log "STAGE measurements FAILED"; exit 1; fi

stage analysis $PY scripts/phase5_analysis.py --scores-root "$R/scores" --measurements "$R/measurements" \
      --raw-manifest "$M/in_the_wild.csv" --out "$R" --n-boot $NBOOT
log "PHASE5 COMPLETE"
