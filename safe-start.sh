#!/usr/bin/env bash
# Start GLM without risking a unified-memory freeze on the head.
# A freeze ends in a watchdog reboot, after which the GB10 clock can stay locked
# at ~600 MHz until a physical power cycle, so a failed start is the cheaper outcome.
set -uo pipefail
DIR="$HOME/glm53-flash"
LOG="$DIR/safe-start.log"
MIN_AVAIL_KB=${MIN_AVAIL_KB:-2097152}   # kill GLM below 2 GiB MemAvailable

echo "$(date -Is) safe-start: stopping infinity_embed, dropping page cache" | tee -a "$LOG"
docker stop -t 30 infinity_embed >/dev/null 2>&1 || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null

"$DIR/glm" start | tee -a "$LOG"
lowest=999999999
while :; do
  avail=$(awk "/MemAvailable/ {print \$2}" /proc/meminfo)
  (( avail < lowest )) && lowest=$avail
  if (( avail < 15728640 )) && [[ -z "${dumped:-}" ]]; then
    dumped=1
    wpid=$(pgrep -f "VLLM::Worker_TP0" | head -1)
    [[ -n "$wpid" ]] && (sudo -n "$HOME/.local/bin/py-spy" dump --pid "$wpid" > "$DIR/spike-stack.txt" 2>&1; sudo -n "$HOME/.local/bin/py-spy" dump --pid "$wpid" >> "$DIR/spike-stack.txt" 2>&1) &
  fi
  if (( avail < MIN_AVAIL_KB )); then
    echo "$(date -Is) safe-start: MemAvailable ${avail} kB < ${MIN_AVAIL_KB} kB, killing GLM" | tee -a "$LOG"
    cat /proc/meminfo >> "$LOG"
    ps -eo rss,pid,comm,args --sort=-rss | head -25 | cut -c1-160 >> "$LOG"
    docker kill glm53-flash >/dev/null 2>&1
    "$DIR/glm" stop >/dev/null 2>&1
    exit 1
  fi
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "$(date -Is) safe-start: GLM ready, lowest MemAvailable $((lowest / 1024)) MiB; starting infinity_embed" | tee -a "$LOG"
    docker start infinity_embed >/dev/null 2>&1
    break
  fi
  [[ "$(docker inspect -f "{{.State.Status}}" glm53-flash 2>/dev/null)" == running ]] || { echo "$(date -Is) safe-start: GLM container exited" | tee -a "$LOG"; exit 1; }
  echo "$(date +%T.%N | cut -c1-12) $avail $(awk '/^(Mlocked|Unevictable|Shmem|AnonPages|Mapped|Cached|MemFree):/ {printf "%s=%d ", $1, $2/1024}' /proc/meminfo)" >> "$DIR/safe-start.mem"; sleep 0.2
done
