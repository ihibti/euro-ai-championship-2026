#!/bin/bash
R=repos/ihibti/euro-ai-championship-2026
G=$HOME/orca/workspaces/euro-ai-championship-2026/brill
last=$(date -u -v-2M +%Y-%m-%dT%H:%M:%SZ)
night_seen=1
heads=$(git -C $G ls-remote origin 'refs/heads/devin/*' 'refs/heads/kkrgp/*' 2>/dev/null | sort)
i=0
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  gh api "$R/issues/comments?since=$last&per_page=100" --jq '.[] | select(.body | startswith("Desk:") | not) | "comment #\(.issue_url | split("/") | last) \(.created_at): \(.body | gsub("\n"; " ") | .[0:200])"' 2>/dev/null || true
  gh api "$R/issues?since=$last&state=all&per_page=50" --jq ".[] | select(.created_at >= \"$last\") | select(.title | startswith(\"Desk\") | not) | \"new #\(.number) \(if .pull_request then \"PR\" else \"issue\" end): \(.title)\"" 2>/dev/null || true
  if [ $night_seen -eq 0 ] && [ -f "$HOME/.config/euroai/night.json" ]; then echo "night.json now exists"; night_seen=1; fi
  if [ $((i % 2)) -eq 0 ]; then
    new=$(git -C $G ls-remote origin 'refs/heads/devin/*' 'refs/heads/kkrgp/*' 2>/dev/null | sort)
    if [ -n "$new" ] && [ "$new" != "$heads" ]; then heads=$new; fi
  fi
  last=$now; i=$((i+1)); sleep 60
done
