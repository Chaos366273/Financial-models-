from pathlib import Path

p = Path(__file__).resolve().parent / "stacked_ensemble.py"
text = p.read_text(encoding="utf-8")
marker = 'if __name__ == "__main__":'
i = text.find(marker)
if i < 0:
    raise SystemExit("main not found")

new_main = '''if __name__ == "__main__":
    print("Config snapshot:")
    print(f"  DATA_SOURCE              : {DATA_SOURCE}")
    print(f"  TICKERS                  : {TICKERS if TICKERS is not None else 'ALL'}")
    print(f"  PPO meta total_timesteps : {PPO_TOTAL_TIMESTEPS}")
    print(f"  PPO meta n_steps         : {PPO_N_STEPS}")
    print(f"  LSTM seq_len/epochs      : {SEQ_LEN} / {LSTM_EPOCHS}")
    print(f"  Auto-start trading PPO   : {AUTO_START_TRADING_PPO}")
    print(f"  Trading PPO (local copy) : {TRADING_PPO_DIR}")
    print(f"  Trading PPO timesteps    : {TRADING_PPO_START_TIMESTEPS} -> {TRADING_PPO_TARGET_TIMESTEPS}")
    if not sb3_available():
        print("  WARNING: pip install stable-baselines3 gymnasium for PPO meta training")
    print()

    if DATA_SOURCE == "bloomberg":
        print("Loading Bloomberg panel...")
        _configure_bloomberg()
    else:
        print("Loading Yahoo US panel...")
    print()

    result = run_stacked_ensemble(TICKERS)

    n = len(result["tickers"])
    print(f"Stacked ensemble (XGB + LSTM -> PPO meta) | Arch B | {n} tickers")
    print(f"Source: {DATA_SOURCE} | meta: {result['meta_method']}")
    print(f"Legs  : {', '.join(result['supervised_legs'])}")
    print(f"Tickers: {', '.join(result['tickers'][:12])}{'...' if n > 12 else ''}")
    print(
        f"Split : base train .. {result['base_train_end']} | "
        f"val {result['val_start']}..{result['val_end']} | "
        f"test {result['test_start']}..{result['test_end']} ({result['test_rows']} rows)"
    )
    if result.get("blend_stats"):
        bs = result["blend_stats"]
        print(
            f"PPO blend weight on XGB: mean={bs['mean_blend_weight_on_xgb']:.3f} "
            f"std={bs['std_blend_weight_on_xgb']:.3f}"
        )
    print("\\nOut-of-sample ENSEMBLE (PPO blend) test metrics:")
    print(f"  MSE  : {result['mse']:.6f}")
    print(f"  MAE  : {result['mae']:.6f}")
    print(f"  RMSE : {result['rmse']:.6f}")
    print(f"  R²   : {result['r2']:.4f}")

    if result.get("architecture_b"):
        ab = result["architecture_b"]
        print("\\nArchitecture B ranked book (for trading PPO timing):")
        print(f"  months={ab['months']} top_k={ab['top_k']} "
              f"avg_longs/month={ab['avg_long_candidates_per_month']:.1f}")

    print("\\nSaved:")
    for k, v in result["paths"].items():
        if v:
            print(f"  {k}: {v}")

    print("\\nLatest test month per ticker:")
    print(result["preview_display"].to_string(index=False))

    if AUTO_START_TRADING_PPO:
        code = launch_trading_ppo()
        if code != 0:
            raise SystemExit(code)
    else:
        print(
            "\\n[Stage 2] Skipped (AUTO_START_TRADING_PPO=False). "
            f'Run manually: python "{TRADING_PPO_SCRIPT}"'
        )
'''

# Fix double-escaped newlines from this writer helper
new_main = new_main.replace("\\n", "\n")

p.write_text(text[:i] + new_main, encoding="utf-8")
print("patched", p)
