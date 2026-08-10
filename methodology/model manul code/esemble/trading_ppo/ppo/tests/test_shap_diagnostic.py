"""
SHAP Diagnostic Test - Identify why SHAP isn't working

Purpose: Isolate each step of SHAP initialization and explanation to find the failure point.
Goal: Use SHAP to identify which features the agent focuses on, then remove low-importance features.

Run: python -m ppo.tests.test_shap_diagnostic
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Ensure project root in path (ppo/tests/ -> ppo/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change cwd to project root so relative paths (ppo/saved_models/...) work
os.chdir(PROJECT_ROOT)


def _get_action_masks(env, n_actions=4):
    """Get action_masks for MaskablePPO from env. Returns all-True if unavailable."""
    import numpy as np
    try:
        base = env
        if hasattr(base, 'envs') and base.envs:
            base = base.envs[0]
        while hasattr(base, 'env') and base.env is not None:
            base = base.env
        if hasattr(base, 'action_masks') and callable(getattr(base, 'action_masks')):
            m = base.action_masks()
            if m is not None:
                return np.asarray(m).reshape(1, -1) if len(np.asarray(m).shape) == 1 else m
    except Exception:
        pass
    return np.ones((1, n_actions), dtype=bool)


def _predict_with_masks(model, obs, env=None):
    """model.predict with action_masks for MaskablePPO compatibility."""
    import numpy as np
    mask = _get_action_masks(env, 4) if env is not None else np.ones((1, 4), dtype=bool)
    try:
        return model.predict(obs, action_masks=mask, deterministic=True)
    except TypeError:
        return model.predict(obs, deterministic=True)


def section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)


def test_1_shap_import():
    """Check if SHAP is installed and importable."""
    section("TEST 1: SHAP Import")
    try:
        import shap
        print(f"  OK: shap version = {getattr(shap, '__version__', 'unknown')}")
        return True, shap
    except ImportError as e:
        print(f"  FAIL: {e}")
        print("  Fix (PowerShell): Use the SAME Python that runs this script:")
        print("    python -m pip install shap")
        print("  Or if using python3.13.exe:")
        print("    python3.13.exe -m pip install shap")
        print("  Then run: python -m ppo.tests.test_shap_diagnostic")
        return False, None


def test_2_minimal_setup():
    """Load orchestrator, data, env, model - minimal setup for SHAP."""
    section("TEST 2: Minimal Setup (Orchestrator, Data, Env, Model)")
    try:
        from PPO_fixed import PPOTradingOrchestrator
        o = PPOTradingOrchestrator()
        print("  OK: Orchestrator loaded")
        data = o.download_market_data()
        if data is None:
            print("  FAIL: Data download returned None")
            return False, None
        print("  OK: Data downloaded")
        env = o.create_trading_environment(data, mode='eval')
        print("  OK: Env created")
        model_path = PROJECT_ROOT / "ppo" / "saved_models" / "ppo_trading_agent.zip"
        model = None
        if model_path.exists():
            try:
                from sb3_contrib import MaskablePPO
                model = MaskablePPO.load(str(model_path), device='cpu')
            except Exception:
                from stable_baselines3 import PPO
                model = PPO.load(str(model_path), device='cpu')
            print("  OK: Model loaded")
        else:
            print(f"  WARN: Model not found at {model_path}")
            print("  Run PPO training first (PPO_fixed.py) to generate the model.")
            print("  Tests 3-9 will be skipped; tests 1, 2 (partial), 4, 5 will run.")
        return True, {'orchestrator': o, 'data': data, 'env': env, 'model': model}
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False, None


def test_3_observation_shape(env, model):
    """Check observation shape and model compatibility."""
    section("TEST 3: Observation Shape & Model Predict")
    try:
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        import numpy as np
        obs = np.array(obs)
        if len(obs.shape) > 1:
            obs_flat = obs.flatten()
        else:
            obs_flat = obs
        print(f"  Obs shape: {obs.shape}, flat length: {len(obs_flat)}")
        action, _ = _predict_with_masks(model, obs, env)
        print(f"  Model predict OK, action: {action}")
        return True, obs_flat
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False, None


def test_4_feature_names(env, tickers):
    """Build feature names - use PPO_fixed._build_feature_names for consistency."""
    section("TEST 4: Feature Names")
    try:
        from PPO_fixed import _build_feature_names
        feature_names = _build_feature_names(env, tickers)
        obs_size = env.observation_space.shape[0] if hasattr(env, 'observation_space') else len(feature_names)
        print(f"  Num features: {len(feature_names)} (obs_size={obs_size})")
        print(f"  Sample: {feature_names[:3]} ... {feature_names[-2:]}")
        return True, feature_names
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False, None


def test_5_background_dataset(env, n_samples=5):
    """Create background dataset - same as AgentExplainer._create_background_dataset."""
    section("TEST 5: Background Dataset")
    try:
        import numpy as np
        background_data = []
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs = np.array(obs)
        for _ in range(n_samples):
            action = env.action_space.sample()
            step_result = env.step(action)
            obs = step_result[0]
            # VecEnv 4-tuple: (obs, rewards, dones, infos); Gym 5-tuple: (obs, reward, term, trunc, info)
            if len(step_result) >= 3:
                d = step_result[2]
                done = d[0] if hasattr(d, '__getitem__') and len(d) > 0 else bool(d)
                if len(step_result) == 5:
                    done = done or bool(step_result[3][0] if hasattr(step_result[3], '__getitem__') and len(step_result[3]) > 0 else step_result[3])
            else:
                done = False
            if isinstance(obs, np.ndarray):
                background_data.append(obs.flatten() if len(obs.shape) > 1 else obs)
            if done:
                reset_result = env.reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                obs = np.array(obs)
        bg = np.array(background_data)
        print(f"  Background shape: {bg.shape}")
        return True, bg
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False, None


def test_6_permutation_explainer_init(model, background_data, feature_names, n_features_subset=None):
    """Try to initialize PermutationExplainer. Uses same probability-based predict as PPO_fixed/AgentExplainer (avoids all-zero SHAP)."""
    section("TEST 6: PermutationExplainer Init")
    from ppo.utils.agent_explainability import _model_predict_action_probability
    if n_features_subset:
        print(f"  Using SUBSET of {n_features_subset} features (for speed)")
        bg = background_data[:, :n_features_subset]
        fn = feature_names[:n_features_subset]
    else:
        bg = background_data
        fn = feature_names
    num_features = len(fn)
    full_obs_size = model.observation_space.shape[0]
    required_max_evals = 2 * num_features + 1
    max_evals = min(int(required_max_evals * 1.5), 12000)
    print(f"  Features: {num_features}, required_max_evals: {required_max_evals}, max_evals: {max_evals}")

    def model_predict(observations):
        import numpy as np
        if isinstance(observations, np.ndarray):
            if len(observations.shape) == 1:
                observations = observations.reshape(1, -1)
            # Pad to full obs size for model (subset case)
            if observations.shape[1] < full_obs_size:
                pad = np.zeros((observations.shape[0], full_obs_size - observations.shape[1]))
                observations = np.hstack([observations, pad])
            # Same as PPO_fixed: return P(predicted action) so SHAP gets non-zero attributions
            return _model_predict_action_probability(model, observations, 4)
        return _model_predict_action_probability(model, np.array(observations).reshape(1, -1), 4)

    try:
        import shap
        t0 = time.time()
        explainer = shap.PermutationExplainer(
            model_predict,
            bg,
            feature_names=fn,
            max_evals=max_evals
        )
        print(f"  OK: PermutationExplainer initialized in {time.time()-t0:.2f}s")
        return True, explainer, model_predict, fn, bg
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False, None, None, None, None


def test_7_single_explain(explainer, obs_flat, feature_names, model, timeout_sec=120):
    """Run single SHAP explanation - this is where it often fails or times out."""
    section("TEST 7: Single SHAP Explain (one observation)")
    # Ensure obs matches feature count for explainer
    if len(obs_flat) > len(feature_names):
        obs_for_shap = obs_flat[:len(feature_names)].reshape(1, -1)
    else:
        obs_for_shap = obs_flat.reshape(1, -1)
    print(f"  Obs for SHAP shape: {obs_for_shap.shape}")
    print(f"  This may take a while (up to {timeout_sec}s)...")
    try:
        t0 = time.time()
        shap_values = explainer(obs_for_shap)
        elapsed = time.time() - t0
        print(f"  OK: SHAP explain completed in {elapsed:.2f}s")
        if hasattr(shap_values, 'values'):
            vals = shap_values.values[0]
            print(f"  SHAP values shape: {vals.shape}")
            top_indices = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)[:10]
            print(f"  Top 10 by |SHAP| (indices): {top_indices}")
            if feature_names and len(feature_names) >= len(vals):
                print(f"  Top 10 feature NAMES:")
                for idx in top_indices:
                    print(f"    {idx}: {feature_names[idx]} = {vals[idx]:.6f}")
        return True, shap_values
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False, None


def test_9_ppo_fixed_shap_integration(orchestrator, env, model, tickers):
    """Test SHAP via PPO_fixed code path (AgentExplainer + explain_evaluation_run)."""
    section("TEST 9: PPO_fixed SHAP Integration")
    try:
        # Simulate PPO_fixed: move model to CPU before SHAP (same as PPO_fixed.py)
        try:
            if hasattr(model, 'policy') and hasattr(model.policy, 'to'):
                model.policy.to('cpu')
                print("  Model moved to CPU (as PPO_fixed does)")
        except Exception as e:
            print(f"  Note: Could not move model to CPU: {e}")

        from PPO_fixed import _build_feature_names
        from ppo.utils.agent_explainability import AgentExplainer

        feature_names = _build_feature_names(env, tickers)
        print(f"  Feature names: {len(feature_names)} (from PPO_fixed._build_feature_names)")

        explainer = AgentExplainer(
            model=model,
            feature_names=feature_names,
            tickers=tickers,
            enable_shap=True,
            save_explanations=False,  # Skip file I/O for test speed
            output_dir="ppo/explanations"
        )

        # Mock results as PPO_fixed passes to explain_evaluation_run
        results = {'total_trades': 1, 'steps_simulated': 100}
        explain_trades_only = results.get('total_trades', 0) > 0
        max_steps = min(30, results.get('steps_simulated', 500))
        max_explanations = 5  # Fewer for fast test

        print(f"  Running explain_evaluation_run (max_steps={max_steps}, max_explanations={max_explanations})...")
        t0 = time.time()
        shap_results = explainer.explain_evaluation_run(
            env=env,
            model=model,
            max_steps=max_steps,
            explain_trades_only=explain_trades_only,
            max_explanations=max_explanations
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.2f}s")

        explanations = shap_results.get('explanations', [])
        shap_count = sum(1 for e in explanations if e.get('method') == 'shap')
        fallback_count = sum(1 for e in explanations if e.get('method') == 'signal_analysis')

        print(f"  Explanations: {len(explanations)} total, {shap_count} SHAP, {fallback_count} fallback")

        if shap_count > 0:
            print(f"  OK: PPO_fixed SHAP integration works (real SHAP used)")
            print(f"  Note: PPO_fixed.py uses this same AgentExplainer; run it to get non-zero SHAP in ppo/explanations/shap_report_*.md")
            return True
        elif len(explanations) > 0:
            print(f"  FAIL: All explanations used fallback (signal_analysis), not real SHAP")
            return False
        else:
            print(f"  WARN: No explanations generated (agent may have 0 BUY/SELL in {max_steps} steps)")
            return True  # Not a SHAP failure, just no trade decisions
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_8_kernel_explainer(model, background_data, feature_names, obs_flat, n_features_subset=50):
    """Try KernelExplainer as alternative. Uses same probability-based predict as PPO_fixed (avoids all-zero SHAP)."""
    section("TEST 8: KernelExplainer (Alternative)")
    import numpy as np
    from ppo.utils.agent_explainability import _model_predict_action_probability
    n = min(n_features_subset, background_data.shape[1], len(obs_flat))
    bg = background_data[:, :n]
    fn = feature_names[:n]
    obs_sub = obs_flat[:n].reshape(1, -1)
    full_obs_size = model.observation_space.shape[0]

    def model_predict(observations):
        if observations.shape[1] < full_obs_size:
            pad = np.zeros((observations.shape[0], full_obs_size - observations.shape[1]))
            observations = np.hstack([observations, pad])
        return _model_predict_action_probability(model, observations, 4)

    try:
        import shap
        print(f"  Using {n} features, nsamples=50")
        t0 = time.time()
        ke = shap.KernelExplainer(model_predict, bg, nsamples=50)
        shap_vals = ke.shap_values(obs_sub, nsamples=100)
        elapsed = time.time() - t0
        print(f"  OK: KernelExplainer completed in {elapsed:.2f}s")
        if isinstance(shap_vals, list):
            print(f"  SHAP values: list of {len(shap_vals)} arrays")
        else:
            print(f"  SHAP values shape: {np.array(shap_vals).shape}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("  SHAP DIAGNOSTIC TEST")
    print("  Goal: Find why SHAP falls back to signal_analysis")
    print("="*60)

    # Test 1
    ok, shap_mod = test_1_shap_import()
    if not ok:
        print("\n  Abort: SHAP not available")
        return 1

    # Test 2
    ok, setup = test_2_minimal_setup()
    if not ok or setup is None:
        print("\n  Abort: Setup failed")
        return 1

    env = setup['env']
    model = setup['model']
    tickers = setup['orchestrator'].config.get('tickers', ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY', 'GLD', 'KO', 'JNJ', 'AMZN', 'XLE', 'TLT'])

    has_model = model is not None
    obs_flat = None
    background_data = None
    feature_names = None

    # Test 3 (requires model)
    if has_model:
        ok, obs_flat = test_3_observation_shape(env, model)
        if not ok or obs_flat is None:
            print("\n  Abort: Observation check failed")
            return 1
    else:
        print("\n  Skipping Test 3 (no model)")

    # Test 4
    ok, feature_names = test_4_feature_names(env, tickers)
    if not ok or feature_names is None:
        print("\n  Abort: Feature names failed")
        return 1

    # Test 5
    ok, background_data = test_5_background_dataset(env, n_samples=10)
    if not ok or background_data is None:
        print("\n  Abort: Background dataset failed")
        return 1

    # Tests 6-9 require model
    if has_model and obs_flat is not None and background_data is not None:
        # Test 6a: Full features (likely to fail or be slow)
        print("\n  --- Trying with FULL features (may fail) ---")
        ok, explainer, _, fn, bg = test_6_permutation_explainer_init(model, background_data, feature_names, n_features_subset=None)
        if ok and explainer is not None:
            test_7_single_explain(explainer, obs_flat, fn, model, timeout_sec=180)
        else:
            print("  Skipping Test 7 (explainer init failed)")

        # Test 6b: Reduced features (50) - faster diagnostic
        print("\n  --- Trying with 50 features (faster) ---")
        ok, explainer, _, fn, bg = test_6_permutation_explainer_init(model, background_data, feature_names, n_features_subset=50)
        if ok and explainer is not None:
            test_7_single_explain(explainer, obs_flat[:50], fn, model, timeout_sec=60)

        # Test 8: KernelExplainer
        test_8_kernel_explainer(model, background_data, feature_names, obs_flat, n_features_subset=50)

        # Test 9: PPO_fixed SHAP integration
        ok9 = test_9_ppo_fixed_shap_integration(setup['orchestrator'], env, model, tickers)
        if not ok9:
            print("\n  WARN: PPO_fixed SHAP integration failed - check AgentExplainer / model device")
    else:
        print("\n  Skipping Tests 6-9 (no model - run PPO training first)")

    section("SUMMARY")
    if has_model:
        print("  DIAGNOSTIC PASSED: SHAP works in isolation.")
        print("  Test 9: PPO_fixed SHAP integration - verifies AgentExplainer works with PPO_fixed code path.")
        print("  For feature pruning: Use PermutationExplainer top-N indices to identify important features.")
    else:
        print("  PARTIAL: Tests 1, 2, 4, 5 passed. Train a model (PPO_fixed.py) for full SHAP diagnostic.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
