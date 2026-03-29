#!/usr/bin/env bash
set -euo pipefail

SCENARIO_INDEX="${SCENARIO_INDEX:-1}"
MAX_SECONDS="${MAX_SECONDS:-120}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:1}"
OUT_DIR="${OUT_DIR:-$HOME/av_runs}"
STATE_PATH="$HOME/openpilot/tools/sim/scenario_bridge_state.json"
SESSION_NAME="${SESSION_NAME:-op-scenarionet}"

mkdir -p "$OUT_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
video_path="$OUT_DIR/scenarionet_${SCENARIO_INDEX}_${ts}.mp4"
trace_path="$OUT_DIR/scenarionet_${SCENARIO_INDEX}_${ts}.tsv"
final_path="$OUT_DIR/scenarionet_${SCENARIO_INDEX}_${ts}.final.json"

echo "starting scenario index=$SCENARIO_INDEX"
SCENARIO_INDEX="$SCENARIO_INDEX" "$HOME/restart_scenarionet.sh" >/tmp/restart_scenarionet.log 2>&1

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session $SESSION_NAME did not start"
  exit 1
fi

ffmpeg -y -video_size 1920x1080 -framerate 20 -f x11grab -i "$DISPLAY_VALUE" \
  -t "$MAX_SECONDS" -pix_fmt yuv420p "$video_path" >/tmp/scenarionet_ffmpeg.log 2>&1 &
ffmpeg_pid=$!

sleep 8
tmux send-keys -t "$SESSION_NAME":0.1 2
sleep 1
tmux send-keys -t "$SESSION_NAME":0.1 1
tmux send-keys -t "$SESSION_NAME":0.1 1
tmux send-keys -t "$SESSION_NAME":0.1 1

echo -e "unix_s\tscenario_id\tengaged\tdist_m\troute_completion\tdone\tout_of_road\tgoal_reached" > "$trace_path"

start="$(date +%s)"
while true; do
  if [[ -f "$STATE_PATH" ]]; then
    jq -r '[.updated_at_unix_s,.scenario_id,.engaged,.distance_to_goal_m,.route_completion,.done, (.done_info.metadrive_info.out_of_road // false), .goal_reached] | @tsv' "$STATE_PATH" >> "$trace_path" || true
    if jq -e '.done == true or .goal_reached == true' "$STATE_PATH" >/dev/null 2>&1; then
      cp "$STATE_PATH" "$final_path"
      break
    fi
  fi

  now="$(date +%s)"
  if (( now - start >= MAX_SECONDS )); then
    if [[ -f "$STATE_PATH" ]]; then
      cp "$STATE_PATH" "$final_path"
    fi
    break
  fi
  sleep 2
done

if kill -0 "$ffmpeg_pid" 2>/dev/null; then
  kill -INT "$ffmpeg_pid" || true
  wait "$ffmpeg_pid" || true
fi

echo "video=$video_path"
echo "trace=$trace_path"
echo "final=$final_path"
if [[ -f "$final_path" ]]; then
  cat "$final_path"
fi
