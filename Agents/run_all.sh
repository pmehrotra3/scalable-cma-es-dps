#!/usr/bin/env bash
########################################
# run_all.sh
# Launches all training runs across all environments and all agents.
#
# Execution order:
#   1. run_cma  — all CMA-ES variants across all environments (completes fully first)
#   2. run_sb3  — all SB3 baselines across all environments (runs after CMA-ES)
#
# Within each function:
#   Phase 1: All non-Humanoid environments, MAX_PARALLEL=4
#   Phase 2: Humanoid-v5,                   MAX_PARALLEL=1
#
# Environment ordering within each phase: Ant-v5, HalfCheetah-v5, Walker2d-v5,
#   BipedalWalkerHardcore-v3 (Phase 1) and Humanoid-v5 (Phase 2).
#
# CMA-ES variant ordering: sep, simultaneous (block_size=4, 32),
#   sequential (block_size=4, 32), full CMA-ES (last — highest memory cost).
#
# Blockwise variants run with two block sizes (--block-size 4 and 32) for sensitivity analysis.
#
# CMA-ES variants live in their own subfolders and must be launched with cwd = the subfolder,
# so each such command is wrapped in a subshell: (cd <folder> && python3 <script> ...)
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com
########################################

declare -a PIDS=()
declare -a CMDS=()

COMPLETED=0
GRAND_TOTAL=0

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── helper: reap any finished child processes ────────────────────────────────
reap_finished() {
  local new_pids=()
  local new_cmds=()
  local i=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      new_cmds+=("${CMDS[$i]}")
    else
      wait "$pid"
      EXIT_CODE=$?
      CMD="${CMDS[$i]}"
      COMPLETED=$((COMPLETED + 1))
      if [ $EXIT_CODE -eq 0 ]; then
        log "✅ FINISHED ($COMPLETED/$GRAND_TOTAL) [PID $pid]: $CMD"
      else
        log "❌ FAILED   ($COMPLETED/$GRAND_TOTAL) [PID $pid] (exit $EXIT_CODE): $CMD"
      fi
    fi
    i=$((i + 1))
  done
  PIDS=("${new_pids[@]}")
  CMDS=("${new_cmds[@]}")
}

# ── helper: run a phase of commands with a parallelism limit ─────────────────
run_phase() {
  local PHASE_NAME="$1"
  local MAX_PARALLEL="$2"
  shift 2
  local COMMANDS=("$@")
  local TOTAL=${#COMMANDS[@]}
  local NEXT=0

  log "▶ Starting $PHASE_NAME — $TOTAL jobs, max $MAX_PARALLEL concurrent."

  while [ $NEXT -lt $TOTAL ] || [ ${#PIDS[@]} -gt 0 ]; do
    while [ ${#PIDS[@]} -lt $MAX_PARALLEL ] && [ $NEXT -lt $TOTAL ]; do
      CMD="${COMMANDS[$NEXT]}"
      NEXT=$((NEXT + 1))
      bash -c "$CMD" &
      PID=$!
      PIDS+=("$PID")
      CMDS+=("$CMD")
      log "🚀 STARTED  (job $NEXT/$TOTAL in $PHASE_NAME) [PID $PID]: $CMD"
    done
    sleep 5
    reap_finished
  done

  log "✔ $PHASE_NAME complete."
}

# =============================================================================
# CMA-ES experiments
# =============================================================================

run_cma() {
  log "════════════════════════════════════════"
  log "  CMA-ES EXPERIMENTS"
  log "════════════════════════════════════════"

  # ── Phase 1: non-Humanoid environments, MAX_PARALLEL=4 ───────────────────
  local CMA_PHASE1=(
    # ANT-v5
    "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5)"

    # HALFCHEETAH-v5
    "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5)"

    # WALKER2D-v5
    "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5)"

    # BIPEDALWALKERHARDCORE-v3
    "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5)"
  )

  # ── Phase 2: Humanoid-v5, MAX_PARALLEL=1 ─────────────────────────────────
  local CMA_PHASE2=(
    "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_simultaneous_blockwise_cma_direct_policy_search && python3 train_simultaneous_blockwise_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5 --block-size 4)"
    "(cd train_sequential_blockwise_cma_direct_policy_search && python3 train_sequential_blockwise_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5 --block-size 32)"
    "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5)"
  )

  GRAND_TOTAL=$(( GRAND_TOTAL + ${#CMA_PHASE1[@]} + ${#CMA_PHASE2[@]} ))
  run_phase "CMA Phase 1 (non-Humanoid)" 4 "${CMA_PHASE1[@]}"
  run_phase "CMA Phase 2 (Humanoid-v5)"  1 "${CMA_PHASE2[@]}"
}

# =============================================================================
# SB3 experiments
# =============================================================================

run_sb3() {
  log "════════════════════════════════════════"
  log "  SB3 EXPERIMENTS"
  log "════════════════════════════════════════"

  # ── Phase 1: non-Humanoid environments, MAX_PARALLEL=4 ───────────────────
  local SB3_PHASE1=(
    # ANT-v5
    "python3 train_a2c.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ppo.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_sac.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ddpg.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_td3.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_tqc.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_trpo.py Ant-v5 --total-timesteps 1000000 --num-runs 5"

    # HALFCHEETAH-v5
    "python3 train_a2c.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ppo.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_sac.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ddpg.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_td3.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_tqc.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_trpo.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"

    # WALKER2D-v5
    "python3 train_a2c.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ppo.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_sac.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ddpg.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_td3.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_tqc.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_trpo.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"

    # BIPEDALWALKERHARDCORE-v3
    "python3 train_a2c.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ppo.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
    "python3 train_sac.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ddpg.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
    "python3 train_td3.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
    "python3 train_tqc.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
    "python3 train_trpo.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  )

  # ── Phase 2: Humanoid-v5, MAX_PARALLEL=1 ─────────────────────────────────
  local SB3_PHASE2=(
    "python3 train_a2c.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ppo.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_sac.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_ddpg.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_td3.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_tqc.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
    "python3 train_trpo.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  )

  GRAND_TOTAL=$(( GRAND_TOTAL + ${#SB3_PHASE1[@]} + ${#SB3_PHASE2[@]} ))
  run_phase "SB3 Phase 1 (non-Humanoid)" 4 "${SB3_PHASE1[@]}"
  run_phase "SB3 Phase 2 (Humanoid-v5)"  1 "${SB3_PHASE2[@]}"
}

# =============================================================================
# Entry point — CMA-ES first, then SB3
# =============================================================================

log "Starting run_all.sh"
run_cma
run_sb3
log "🏁 All $GRAND_TOTAL jobs completed."