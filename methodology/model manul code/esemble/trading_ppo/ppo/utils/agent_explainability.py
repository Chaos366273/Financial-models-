#!/usr/bin/env python3
"""Agent Explainability System"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
logger = logging.getLogger(__name__)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

def _model_predict_action_probability(model, observations: np.ndarray, n_actions: int = 4) -> np.ndarray:
    """
    Return probability of the predicted action for each row (not the action index).
    SHAP needs a smooth output to attribute; discrete actions yield all-zero SHAP values.
    """
    import torch
    if not isinstance(observations, np.ndarray):
        observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = observations.reshape(1, -1)
    n = observations.shape[0]
    try:
        mask = np.ones((n, n_actions), dtype=bool)
        actions, _ = model.predict(observations, action_masks=mask, deterministic=True)
    except TypeError:
        actions, _ = model.predict(observations, deterministic=True)
    actions = np.asarray(actions).flatten()
    policy = getattr(model, "policy", model)
    device = getattr(policy, "device", None)
    if device is None:
        device = next(policy.parameters()).device
    obs_t = torch.as_tensor(observations, device=device, dtype=torch.float32)
    with torch.no_grad():
        dist = policy.get_distribution(obs_t)
        inner = getattr(dist, "distribution", dist)
        probs = getattr(inner, "probs", None)
        if probs is None and hasattr(inner, "logits"):
            probs = torch.softmax(inner.logits, dim=-1)
        if probs is None:
            return np.ones(n, dtype=np.float64) * 0.25
        probs_np = probs.cpu().numpy()
    return probs_np[np.arange(n), actions]


class AgentExplainer:
    def __init__(self, model, feature_names: List[str], tickers: List[str], enable_shap: bool = True, save_explanations: bool = True, output_dir: str = "ppo/explanations"):
        self.model = model
        self.feature_names = feature_names
        self.tickers = tickers
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.save_explanations = save_explanations
        self.output_dir = output_dir
        if self.save_explanations:
            os.makedirs(self.output_dir, exist_ok=True)
        self.explanations: List[Dict[str, Any]] = []
        self.shap_explainer = None
        logger.info(f"[SHAP] AgentExplainer initialized (SHAP available: {self.enable_shap})")
    def explain_evaluation_run(self, env, model, max_steps: int = 500, explain_trades_only: bool = True, max_explanations: int = 50) -> Dict[str, Any]:
        self.explanations = []
        # Handle both Gym API (tuple) and VecEnv API (single value)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        # Ensure obs is numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        done = False
        step_count = 0
        explanations_generated = 0
        logger.info(f"[SHAP] Starting explanation generation")
        if self.enable_shap:
            try:
                background_data = self._create_background_dataset(env, n_samples=10)
                n_actions = 4
                # SHAP needs smooth output (probability of predicted action), not discrete action index,
                # otherwise permuting one of 3k+ features rarely changes argmax -> all SHAP values 0.
                def model_predict(observations):
                    return _model_predict_action_probability(model, observations, n_actions)
                
                # CRITICAL FIX: Calculate required max_evals for Permutation explainer
                # Permutation explainer requires: max_evals >= 2 * num_features + 1
                num_features = len(self.feature_names)
                required_max_evals = 2 * num_features + 1
                # Use 1.5x required; cap must be >= required or explainer fails (was 5000, failed for 3241 features)
                max_evals = min(int(required_max_evals * 1.5), 8000)
                _ = model_predict(background_data[0:1])  # Sanity check
                logger.info(f"[SHAP] Initializing with {num_features} features, max_evals={max_evals} (required: {required_max_evals})")
                
                # Use Permutation explainer with proper max_evals
                if shap is not None:
                    self.shap_explainer = shap.PermutationExplainer(
                        model_predict, 
                        background_data, 
                        feature_names=self.feature_names,
                        max_evals=max_evals
                    )
                else:
                    raise ImportError("shap module not available")
                logger.info(f"[SHAP] PermutationExplainer initialized successfully with max_evals={max_evals}")
            except Exception as e:
                import traceback
                logger.warning(f"[SHAP] Failed to initialize PermutationExplainer: {e}")
                logger.info(f"[SHAP] Traceback: {traceback.format_exc()}")
                # Fallback: KernelExplainer (more robust for black-box)
                try:
                    logger.info("[SHAP] Attempting fallback to KernelExplainer...")
                    if shap is not None:
                        bg = self._create_background_dataset(env, n_samples=5)
                        def _pred(x):
                            return _model_predict_action_probability(model, x, 4)
                        self.shap_explainer = shap.KernelExplainer(_pred, bg[:3])
                        logger.info("[SHAP] KernelExplainer initialized successfully")
                    else:
                        raise ImportError("shap module not available")
                except Exception as e2:
                    import traceback as tb2
                    logger.warning(f"[SHAP] Failed SHAP (Permutation + Kernel): {e2}")
                    logger.info(f"[SHAP] Kernel traceback: {tb2.format_exc()}")
                    self.enable_shap = False
        while not done and step_count < max_steps and explanations_generated < max_explanations:
            # Ensure obs is proper format for model.predict
            if isinstance(obs, tuple):
                obs = obs[0]
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            # 2026-02-19: Pass action_masks so explanations match masked behavior (no SELL when flat)
            action_masks = None
            try:
                base = env
                while hasattr(base, 'env') and base.env is not None:
                    base = base.env
                if hasattr(base, 'action_masks') and callable(getattr(base, 'action_masks')):
                    action_masks = base.action_masks()
            except Exception:
                action_masks = None
            if action_masks is not None:
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            # CRITICAL FIX: Normalize action to int scalar (handles numpy arrays from model.predict)
            if isinstance(action, np.ndarray):
                action = int(action.item() if action.size == 1 else action[0])
            elif not isinstance(action, (int, np.integer)):
                action = int(action)
            else:
                action = int(action)  # Ensure it's a Python int
            
            should_explain = True
            if explain_trades_only and action == 0:
                should_explain = False
            if should_explain:
                try:
                    explanation = self._explain_decision(obs, action, step_count)
                    if explanation:
                        self.explanations.append(explanation)
                        explanations_generated += 1
                        if self.save_explanations:
                            self._save_individual_explanation(explanation, step_count, action)
                except Exception as e:
                    logger.warning(f"[SHAP] Failed to explain step {step_count}: {e}")
            # Handle both Gym API (tuple) and VecEnv API (array)
            step_result = env.step(action)
            if isinstance(step_result, tuple) and len(step_result) == 4:
                obs, reward, done, info = step_result
            elif isinstance(step_result, tuple) and len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # VecEnv returns array
                obs = step_result
                done = False  # VecEnv handles done internally
            step_count += 1
        return {'explanations_generated': explanations_generated, 'total_steps': step_count, 'explanations': self.explanations}
    def _create_background_dataset(self, env, n_samples: int = 50) -> np.ndarray:
        background_data = []
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        for _ in range(n_samples):
            action = env.action_space.sample()
            step_result = env.step(action)
            if isinstance(step_result, tuple) and len(step_result) >= 3:
                obs = step_result[0]
                done = step_result[2] if len(step_result) == 4 else (step_result[2] or step_result[3] if len(step_result) == 5 else False)
            else:
                obs = step_result
                done = False
            if isinstance(obs, np.ndarray):
                if len(obs.shape) == 1:
                    background_data.append(obs)
                else:
                    background_data.append(obs.flatten())
            if done:
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]
                else:
                    obs = reset_result
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
        return np.array(background_data)
    def _explain_decision(self, observation: np.ndarray, action: int, step: int) -> Optional[Dict[str, Any]]:
        try:
            if len(observation.shape) > 1:
                obs_flat = observation.flatten()
            else:
                obs_flat = observation.copy()
            if len(obs_flat) != len(self.feature_names):
                if len(obs_flat) < len(self.feature_names):
                    obs_flat = np.pad(obs_flat, (0, len(self.feature_names) - len(obs_flat)))
                else:
                    obs_flat = obs_flat[:len(self.feature_names)]
            explanation = {'step': step, 'action': int(action), 'action_name': ['HOLD', 'BUY', 'SELL', 'SHORT'][action] if 0 <= action <= 3 else 'UNKNOWN', 'feature_importance': {}, 'top_features': [], 'confidence': 0.0, 'shap_values': None}
            if self.enable_shap and self.shap_explainer is not None:
                # IMPROVED: Retry logic for SHAP calculation
                max_retries = 3
                shap_success = False
                for retry in range(max_retries):
                    try:
                        # CRITICAL FIX: Use max_evals parameter if explainer supports it (PermutationExplainer)
                        if hasattr(self.shap_explainer, 'max_evals'):
                            # PermutationExplainer - use max_evals from initialization
                            shap_values = self.shap_explainer(obs_flat.reshape(1, -1))
                        else:
                            # Other explainers (TreeExplainer, etc.) - call normally
                            shap_values = self.shap_explainer(obs_flat.reshape(1, -1))
                        
                        if hasattr(shap_values, 'values'):
                            shap_vals = shap_values.values[0]
                        elif isinstance(shap_values, list):
                            shap_vals = shap_values[0].values[0] if len(shap_values) > 0 else None
                        else:
                            shap_vals = shap_values[0] if len(shap_values) > 0 else None
                        
                        if shap_vals is not None and len(shap_vals) > 0:
                            # Ensure feature names match SHAP values length
                            min_len = min(len(self.feature_names), len(shap_vals))
                            for i in range(min_len):
                                feature_name = self.feature_names[i]
                                shap_val = float(shap_vals[i])
                                explanation['feature_importance'][feature_name] = shap_val
                            
                            feature_importance_sorted = sorted(explanation['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)
                            explanation['top_features'] = [{'feature': name, 'shap_value': float(val)} for name, val in feature_importance_sorted[:15]]
                            # FIXED: Use max absolute SHAP value (more sensitive to strong signals) 
                            # Combined with L2 norm normalized by feature count for robustness
                            max_shap = float(np.abs(shap_vals).max())
                            l2_norm = float(np.sqrt(np.sum(shap_vals**2)) / len(shap_vals)) if len(shap_vals) > 0 else 0.0
                            explanation['confidence'] = max(max_shap, l2_norm)  # Use the larger value for better sensitivity
                            explanation['shap_values'] = shap_vals.tolist() if isinstance(shap_vals, np.ndarray) else shap_vals
                            explanation['method'] = 'shap'
                            shap_success = True
                            break
                        else:
                            logger.debug(f"[SHAP] Retry {retry+1}/{max_retries}: SHAP values are None or empty")
                    except Exception as e:
                        if retry < max_retries - 1:
                            logger.debug(f"[SHAP] Retry {retry+1}/{max_retries}: {e}")
                        else:
                            logger.warning(f"[SHAP] Failed to calculate SHAP values after {max_retries} retries: {e}")
                
                if not shap_success:
                    explanation = self._explain_with_signals(observation, action, step)
            else:
                explanation = self._explain_with_signals(observation, action, step)
            return explanation
        except Exception as e:
            logger.warning(f"[SHAP] Error explaining decision: {e}")
            return None
    def _explain_with_signals(self, observation: np.ndarray, action: int, step: int) -> Dict[str, Any]:
        explanation = {'step': step, 'action': int(action), 'action_name': ['HOLD', 'BUY', 'SELL', 'SHORT'][action] if 0 <= action <= 3 else 'UNKNOWN', 'feature_importance': {}, 'top_features': [], 'confidence': 0.5, 'shap_values': None, 'method': 'signal_analysis'}
        obs_flat = observation.flatten() if len(observation.shape) > 1 else observation
        
        # FIX: Normalize feature values to prevent all showing 1.0
        # Use relative importance (normalized by max absolute value)
        abs_values = np.abs(obs_flat.astype(float))
        max_abs = np.max(abs_values) if len(abs_values) > 0 and np.max(abs_values) > 0 else 1.0
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(obs_flat):
                # Normalize importance to 0-1 range based on relative magnitude
                raw_value = abs(float(obs_flat[i]))
                importance = raw_value / max_abs if max_abs > 0 else 0.0
                explanation['feature_importance'][feature_name] = importance
        
        feature_importance_sorted = sorted(explanation['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)
        explanation['top_features'] = [{'feature': name, 'importance': float(val)} for name, val in feature_importance_sorted[:15]]
        return explanation
    def _save_individual_explanation(self, explanation: Dict[str, Any], step: int, action: int) -> None:
        try:
            action_name = ['HOLD', 'BUY', 'SELL', 'SHORT'][action] if 0 <= action <= 3 else 'UNKNOWN'
            filename = f"explanation_step_{step}_{action_name}.json"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(explanation, f, indent=2)
        except Exception as e:
            logger.warning(f"[SHAP] Failed to save explanation: {e}")
    def generate_shap_report(self, execution_results: Optional[Dict[str, Any]] = None) -> Optional[str]:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.output_dir, f"shap_report_{timestamp}.md")
            if not self.explanations:
                logger.warning("[SHAP] No explanations available; writing minimal report")
                with open(report_path, 'w') as f:
                    f.write("# SHAP Explainability Report\n\n")
                    f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("## Summary\n\n")
                    f.write("- **Total Decisions Explained**: 0\n")
                    f.write("- No BUY/SELL decisions in this run; no explanations generated.\n")
                    if execution_results:
                        pac = execution_results.get('policy_action_counts', {})
                        pap = execution_results.get('policy_action_percentages', {})
                        f.write(f"- **Deterministic action distribution**: HOLD={pac.get('HOLD', 0)} ({pap.get('HOLD', 0):.1f}%), BUY={pac.get('BUY', 0)} ({pap.get('BUY', 0):.1f}%), SELL={pac.get('SELL', 0)} ({pap.get('SELL', 0):.1f}%)\n")
                    f.write("\n")
                logger.info(f"[SHAP] Minimal report generated: {report_path}")
                return report_path
        except Exception as e0:
            logger.warning(f"[SHAP] Failed to write minimal report: {e0}")
            return None
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.output_dir, f"shap_report_{timestamp}.md")
            
            # Count SHAP vs fallback methods
            shap_count = sum(1 for exp in self.explanations if exp.get('method') == 'shap')
            fallback_count = sum(1 for exp in self.explanations if exp.get('method') == 'signal_analysis')
            
            # Aggregate top features across all explanations
            feature_importance_agg = {}
            for exp in self.explanations:
                top_features = exp.get('top_features', [])
                for feat in top_features:
                    feat_name = feat.get('feature', 'Unknown')
                    shap_val = feat.get('shap_value') or feat.get('importance', 0.0)
                    if feat_name not in feature_importance_agg:
                        feature_importance_agg[feat_name] = []
                    feature_importance_agg[feat_name].append(abs(shap_val))
            
            # Calculate average importance per feature
            feature_avg_importance = {name: np.mean(vals) for name, vals in feature_importance_agg.items()}
            top_features_global = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            with open(report_path, 'w') as f:
                f.write("# SHAP Explainability Report\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## Summary\n\n")
                f.write(f"- **Total Decisions Explained**: {len(self.explanations)}\n")
                f.write(f"- **SHAP Method Used**: {shap_count} decisions\n")
                f.write(f"- **Fallback Method Used**: {fallback_count} decisions\n")
                f.write(f"- **Features Analyzed**: {len(self.feature_names)}\n")
                shap_available = shap_count > 0 or (self.enable_shap and self.shap_explainer is not None)
                f.write(f"- **SHAP Status**: {'Available' if shap_available else 'Unavailable'}\n")
                if not shap_available and fallback_count > 0:
                    f.write(f"- **Tip**: If SHAP works in `python3.13.exe -m ppo.tests.test_shap_diagnostic` but not here, run PPO_fixed with the same Python: `python3.13.exe PPO_fixed.py`\n")
                f.write("\n")
                
                if top_features_global:
                    f.write(f"## Top 20 Most Important Features (Global)\n\n")
                    f.write("| Rank | Feature Name | Average Importance |\n")
                    f.write("|------|--------------|---------------------|\n")
                    for rank, (feat_name, avg_imp) in enumerate(top_features_global, 1):
                        f.write(f"| {rank} | {feat_name} | {avg_imp:.6f} |\n")
                    f.write("\n")
                
                # Sample explanations
                f.write(f"## Sample Explanations\n\n")
                for i, exp in enumerate(self.explanations[:5], 1):
                    action_name = exp.get('action_name', 'UNKNOWN')
                    method = exp.get('method', 'unknown')
                    confidence = exp.get('confidence', 0.0)
                    top_features = exp.get('top_features', [])[:5]
                    
                    f.write(f"### Decision {i}: {action_name} (Step {exp.get('step', '?')})\n\n")
                    f.write(f"- **Method**: {method}\n")
                    f.write(f"- **Confidence**: {confidence:.4f}\n")
                    if top_features:
                        f.write(f"- **Top Features**:\n")
                        for feat in top_features:
                            feat_name = feat.get('feature', 'Unknown')
                            feat_val = feat.get('shap_value') or feat.get('importance', 0.0)
                            f.write(f"  - {feat_name}: {feat_val:.6f}\n")
                    f.write("\n")
            
            logger.info(f"[SHAP] Report generated: {report_path}")
            logger.info(f"[SHAP] SHAP method: {shap_count}, Fallback: {fallback_count}")
            return report_path
        except Exception as e:
            logger.error(f"[SHAP] Failed to generate report: {e}")
            import traceback
            logger.debug(f"[SHAP] Traceback: {traceback.format_exc()}")
            return None