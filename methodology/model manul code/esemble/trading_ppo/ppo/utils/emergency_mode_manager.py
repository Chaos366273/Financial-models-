"""
Emergency Mode Manager for PPO Trading System

This module provides emergency controls and safety mechanisms to prevent
system failures and ensure stable operation.
"""

import os
import logging
from typing import Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

class EmergencyModeManager:
    """Emergency mode manager for system safety"""
    
    def __init__(self):
        self.emergency_mode = False
        self.blocked_operations = []
        self.critical_errors = []
        self.model_cleanup_threshold = 5  # Max invalid models before cleanup
        
    def is_emergency_mode_active(self) -> bool:
        """Check if emergency mode is currently active"""
        return self.emergency_mode
    
    def can_train_model(self) -> Tuple[bool, str]:
        """Check if model training is allowed"""
        if self.emergency_mode:
            return False, "Emergency mode is active"
        
        if len(self.critical_errors) > 3:
            return False, f"Too many critical errors: {len(self.critical_errors)}"
            
        return True, "Training allowed"
    
    def add_critical_error(self, error: str):
        """Add a critical error to the system"""
        self.critical_errors.append(error)
        logger.error(f"CRITICAL ERROR: {error}")
        
        if len(self.critical_errors) > 5:
            self.emergency_mode = True
            logger.error(" EMERGENCY MODE ACTIVATED - Too many critical errors")
    
    def cleanup_invalid_models(self):
        """Clean up invalid model files"""
        try:
            models_dir = Path("ppo/saved_models")
            if not models_dir.exists():
                return
                
            invalid_files = []
            for model_file in models_dir.glob("*.pt"):
                if model_file.stat().st_size < 1000:  # Less than 1KB is likely invalid
                    invalid_files.append(model_file)
            
            for invalid_file in invalid_files:
                try:
                    invalid_file.unlink()
                    logger.info(f" Removed invalid model: {invalid_file.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {invalid_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
    
    def reset_emergency_mode(self):
        """Reset emergency mode and clear errors"""
        self.emergency_mode = False
        self.critical_errors.clear()
        self.blocked_operations.clear()
        logger.info(" Emergency mode reset - system restored")

# Global emergency manager instance
emergency_manager = EmergencyModeManager()
