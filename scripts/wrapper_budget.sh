#!/usr/bin/env bash
# Keep multi-turn wrapper cleanup inside Harbor's outer agent timeout.

set_swe_wrapper_budget_from_args() {
  # An explicit operator override always wins.
  if [ -n "${TRIAL_BUDGET_SEC:-}" ]; then
    return 0
  fi

  local timeout=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --agent-timeout)
        shift
        [ "$#" -gt 0 ] && timeout="$1"
        break
        ;;
      --agent-timeout=*)
        timeout="${1#--agent-timeout=}"
        break
        ;;
    esac
    shift
  done

  if [[ "$timeout" =~ ^[0-9]+$ ]]; then
    # Reserve one minute for wrapper unwind, final patch capture, and Harbor's
    # verifier transition. Without this, a cap-rescue turn can run into the
    # outer asyncio timeout and turn a valid partial solution into missing data.
    if [ "$timeout" -gt 60 ]; then
      export TRIAL_BUDGET_SEC="$((timeout - 60))"
    else
      export TRIAL_BUDGET_SEC="$timeout"
    fi
  fi
}
