import json
import os
import time
from pathlib import Path
from ppo.utils.dynamic_config import get_dynamic_timesteps, update_dynamic_timesteps

class ScalingController:
    """
    Manages gradual timestep scaling based on error tracking
    """
    def __init__(self, config):
        self.config = config
        self.history_file = Path('ppo/saved_data/scaling_history.json')
        self.load_history()
    
    def load_history(self):
        """Load scaling history"""
        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)
        else:
            # CENTRALIZED: Initialize from dynamic_config instead of config
            current_timesteps = get_dynamic_timesteps()
            self.history = {
                'current_timesteps': current_timesteps,
                'runs': [],
                'clean_runs': 0
            }
    
    def save_history(self):
        """Save scaling history"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_next_timesteps(self, error_summary):
        """
        Determine next timestep count based on errors
        Returns: (timesteps, action, reason)
        """
        # CENTRALIZED: Read current from dynamic_config, not history
        current = get_dynamic_timesteps()
        
        # Check if we should reduce
        if error_summary['should_reduce']:
            new_timesteps = int(current * self.config['reduction_factor'])
            self.history['clean_runs'] = 0
            # CENTRALIZED: Update dynamic_config when reducing
            update_dynamic_timesteps(new_timesteps)
            return new_timesteps, 'REDUCE', 'Errors detected - reducing timesteps'
        
        # Check if we should increase
        self.history['clean_runs'] += 1
        if self.history['clean_runs'] >= self.config['success_runs_before_increase']:
            new_timesteps = current + self.config['increment_size']
            if new_timesteps <= self.config['target_timesteps']:
                self.history['clean_runs'] = 0
                # CENTRALIZED: Update dynamic_config when increasing
                update_dynamic_timesteps(new_timesteps)
                return new_timesteps, 'INCREASE', 'Clean runs - increasing timesteps'
            else:
                return current, 'MAINTAIN', 'Target reached - maintaining timesteps'
        
        # Maintain current
        return current, 'MAINTAIN', f'Need {self.config["success_runs_before_increase"] - self.history["clean_runs"]} more clean runs'
    
    def record_run(self, timesteps, error_summary, results):
        """Record run results"""
        self.history['runs'].append({
            'timesteps': timesteps,
            'errors': error_summary['total_errors'],
            'warnings': error_summary['warnings'],
            'return': results.get('total_return', 0),
            'sharpe': results.get('sharpe_ratio', 0),
            'timestamp': time.time()
        })
        # CENTRALIZED: Update dynamic_config when recording new timesteps
        update_dynamic_timesteps(timesteps)
        self.history['current_timesteps'] = timesteps
        self.save_history()
