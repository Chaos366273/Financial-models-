"""
Model manager for PPO Trading Agent
Handles model saving, loading, and versioning
"""

import os
import torch
import json
import shutil
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import pickle

from ..core.exceptions import ModelException

class ModelManager:
    """Model manager for handling model operations"""

    def __init__(self, base_dir: str = "models"):
        """
        Initialize model manager

        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)

        # Create base directory
        os.makedirs(base_dir, exist_ok=True)

        # Model metadata
        self.metadata_file = os.path.join(base_dir, "model_metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load model metadata from file"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'models': {},
                    'latest_model': None,
                    'best_model': None,
                    'model_count': 0
                }
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return {
                'models': {},
                'latest_model': None,
                'best_model': None,
                'model_count': 0
            }

    def _save_metadata(self):
        """Save model metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def save_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   config: Dict, metrics: Dict, model_name: Optional[str] = None) -> str:
        """
        Save model with metadata

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            config: Model configuration
            metrics: Training metrics
            model_name: Optional model name

        Returns:
            Path to saved model
        """
        try:
            # Generate model name if not provided
            if model_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"ppo_model_{timestamp}"

            # Create model directory
            model_dir = os.path.join(self.base_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save model checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

            checkpoint_path = os.path.join(model_dir, "checkpoint.pth")
            torch.save(checkpoint, checkpoint_path)

            # Save metadata
            model_info = {
                'name': model_name,
                'path': model_dir,
                'checkpoint_path': checkpoint_path,
                'config': config,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
            }

            self.metadata['models'][model_name] = model_info
            self.metadata['latest_model'] = model_name
            self.metadata['model_count'] += 1

            # Update best model if better
            if self._is_better_model(metrics):
                self.metadata['best_model'] = model_name

            self._save_metadata()

            self.logger.info(f"Model saved: {model_name}")
            return model_dir

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise ModelException(f"Failed to save model: {e}")

    def load_model(self, model_name: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict, Dict]:
        """
        Load model by name

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (model, optimizer, config, metrics)
        """
        try:
            if model_name not in self.metadata['models']:
                raise ModelException(f"Model {model_name} not found in metadata")

            model_info = self.metadata['models'][model_name]
            checkpoint_path = model_info['checkpoint_path']

            if not os.path.exists(checkpoint_path):
                raise ModelException(f"Checkpoint file not found: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Create model and optimizer (caller should initialize them)
            model_state_dict = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            config = checkpoint['config']
            metrics = checkpoint['metrics']

            self.logger.info(f"Model loaded: {model_name}")
            return model_state_dict, optimizer_state_dict, config, metrics

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelException(f"Failed to load model: {e}")

    def load_latest_model(self) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict, Dict]:
        """Load the latest model"""
        if self.metadata['latest_model'] is None:
            raise ModelException("No models available")
        return self.load_model(self.metadata['latest_model'])

    def load_best_model(self) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict, Dict]:
        """Load the best performing model"""
        if self.metadata['best_model'] is None:
            raise ModelException("No best model available")
        return self.load_model(self.metadata['best_model'])

    def _is_better_model(self, metrics: Dict) -> bool:
        """Check if current model is better than best model"""
        if self.metadata['best_model'] is None:
            return True

        best_model_metrics = self.metadata['models'][self.metadata['best_model']]['metrics']
        
        # Compare by reward (primary metric)
        current_reward = metrics.get('best_reward', float('-inf'))
        best_reward = best_model_metrics.get('best_reward', float('-inf'))
        
        return current_reward > best_reward

    def list_models(self) -> List[Dict]:
        """List all available models"""
        models = []
        for name, info in self.metadata['models'].items():
            models.append({
                'name': name,
                'timestamp': info['timestamp'],
                'metrics': info['metrics'],
                'size_mb': info['size_mb']
            })
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)

    def delete_model(self, model_name: str):
        """Delete a model"""
        try:
            if model_name not in self.metadata['models']:
                raise ModelException(f"Model {model_name} not found")

            # Remove model directory
            model_dir = self.metadata['models'][model_name]['path']
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            # Remove from metadata
            del self.metadata['models'][model_name]

            # Update latest/best model if needed
            if self.metadata['latest_model'] == model_name:
                self.metadata['latest_model'] = None
                if self.metadata['models']:
                    self.metadata['latest_model'] = list(self.metadata['models'].keys())[-1]

            if self.metadata['best_model'] == model_name:
                self.metadata['best_model'] = None
                # Find new best model
                best_reward = float('-inf')
                for name, info in self.metadata['models'].items():
                    reward = info['metrics'].get('best_reward', float('-inf'))
                    if reward > best_reward:
                        best_reward = reward
                        self.metadata['best_model'] = name

            self._save_metadata()
            self.logger.info(f"Model deleted: {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to delete model: {e}")
            raise ModelException(f"Failed to delete model: {e}")

    def cleanup_old_models(self, keep_count: int = 5):
        """Clean up old models, keeping only the most recent ones"""
        try:
            models = self.list_models()
            if len(models) <= keep_count:
                return

            # Keep best model and latest models
            models_to_keep = set()
            if self.metadata['best_model']:
                models_to_keep.add(self.metadata['best_model'])
            if self.metadata['latest_model']:
                models_to_keep.add(self.metadata['latest_model'])

            # Add most recent models
            for model in models[:keep_count]:
                models_to_keep.add(model['name'])

            # Delete old models
            for model in models[keep_count:]:
                if model['name'] not in models_to_keep:
                    self.delete_model(model['name'])

            self.logger.info(f"Cleaned up old models, keeping {len(models_to_keep)} models")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old models: {e}")

    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a model"""
        if model_name not in self.metadata['models']:
            raise ModelException(f"Model {model_name} not found")

        return self.metadata['models'][model_name]

    def export_model(self, model_name: str, export_path: str):
        """Export model to external location"""
        try:
            if model_name not in self.metadata['models']:
                raise ModelException(f"Model {model_name} not found")

            model_dir = self.metadata['models'][model_name]['path']
            if not os.path.exists(model_dir):
                raise ModelException(f"Model directory not found: {model_dir}")

            # Copy model directory
            shutil.copytree(model_dir, export_path)
            self.logger.info(f"Model exported to: {export_path}")

        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            raise ModelException(f"Failed to export model: {e}")

    def get_storage_info(self) -> Dict:
        """Get storage information"""
        total_size = 0
        model_count = len(self.metadata['models'])

        for model_info in self.metadata['models'].values():
            total_size += model_info.get('size_mb', 0)

        return {
            'total_models': model_count,
            'total_size_mb': total_size,
            'latest_model': self.metadata['latest_model'],
            'best_model': self.metadata['best_model']
        }
