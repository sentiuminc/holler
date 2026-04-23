#!/bin/bash
# Orchestrator — chained after training is kicked off remotely.
# Waits for epoch-1 checkpoint on instance, downloads, runs inference,
# downloads samples, transcribes locally, then stops the instance.
set -e

SSH_KEY=/Users/nagy/.ssh/runpod
SSH_USER=root
SSH_HOST=ssh7.vast.ai
SSH_PORT=17474
INSTANCE_ID=35377474
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT"
RUN_REMOTE() { ssh $SSH_OPTS "$SSH_USER@$SSH_HOST" "$@"; }

LOCAL_OUT=~/Downloads/ivi-v7-katie-joe
mkdir -p "$LOCAL_OUT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- Stage 1: wait for epoch-1 checkpoint to appear ----
log "waiting for epoch-1 checkpoint on instance..."
EPOCH1_DIR=/workspace/output/checkpoint-epoch-1
while true; do
  # Check if training died
  if ! RUN_REMOTE "kill -0 \$(cat /workspace/train.pid 2>/dev/null) 2>/dev/null"; then
    # process died — check if checkpoint exists (success) or not (failure)
    if RUN_REMOTE "[ -f $EPOCH1_DIR/model.safetensors ]"; then
      log "training complete, epoch-1 ready"
      break
    else
      log "ERROR: training process died without saving epoch-1"
      RUN_REMOTE "tail -40 /workspace/train.log" 2>&1 | sed 's/^/[REMOTE] /'
      exit 1
    fi
  fi
  # Still running — peek at log
  last=$(RUN_REMOTE "tail -1 /workspace/train.log 2>/dev/null | tr -d '\r'" 2>/dev/null || echo "")
  if [ -n "$last" ]; then
    log "training: $last"
  fi
  sleep 30
done

# ---- Stage 2: run inference on instance (10 clips per voice) ----
log "running PyTorch inference for Katie + Joe (10 clips each)..."
RUN_REMOTE "cd /workspace && python3 remote_inference_multivoice.py \
    --checkpoint $EPOCH1_DIR \
    --output_dir /workspace/inference_samples \
    --voices katie joe 2>&1 | tee /workspace/inference.log | tail -40"

# ---- Stage 3: download checkpoint + samples ----
log "downloading epoch-1 checkpoint to $LOCAL_OUT/checkpoint..."
rsync -az --info=stats1 -e "ssh $SSH_OPTS" \
  "$SSH_USER@$SSH_HOST:$EPOCH1_DIR/" "$LOCAL_OUT/checkpoint/" 2>&1 | tail -5

log "downloading inference samples..."
rsync -az --info=stats1 -e "ssh $SSH_OPTS" \
  "$SSH_USER@$SSH_HOST:/workspace/inference_samples/" "$LOCAL_OUT/samples/" 2>&1 | tail -5

log "downloading training + inference logs..."
scp $SSH_OPTS "$SSH_USER@$SSH_HOST:/workspace/train.log" "$LOCAL_OUT/train.log" 2>&1 | tail -2
scp $SSH_OPTS "$SSH_USER@$SSH_HOST:/workspace/inference.log" "$LOCAL_OUT/inference.log" 2>&1 | tail -2

# ---- Stage 4: transcribe locally via whisper ----
log "transcribing samples locally with whisper..."
source /Users/nagy/Desktop/Files/AI/ivi/audio-test/.venv-tts-bench/bin/activate
python3 /Users/nagy/Desktop/Files/AI/ivi/audio-test/qwen3-tts-finetune/transcribe_and_verify.py \
  --manifest "$LOCAL_OUT/samples/manifest.jsonl" 2>&1 | tee "$LOCAL_OUT/transcription.log"

# ---- Stage 5: stop the instance (not destroy) ----
log "stopping Vast instance $INSTANCE_ID (preserved for future training)..."
vastai stop instance $INSTANCE_ID 2>&1 | tail -3

log "DONE. Artifacts in $LOCAL_OUT/"
ls -la "$LOCAL_OUT/"
