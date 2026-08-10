"""
JSON utilities for safe serialization of numpy types and other non-serializable objects
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
import logging

logger = logging.getLogger(__name__)

def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string, handling numpy types
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    try:
        # Convert numpy types to Python native types
        converted_obj = convert_numpy_types(obj)
        return json.dumps(converted_obj, **kwargs)
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        # Fallback: try to convert to string representation
        return json.dumps(str(obj), **kwargs)

def safe_json_dump(obj: Any, file_path: str, **kwargs) -> None:
    """
    Safely serialize object to JSON file, handling numpy types
    
    Args:
        obj: Object to serialize
        file_path: Path to save JSON file
        **kwargs: Additional arguments for json.dump
    """
    try:
        # Convert numpy types to Python native types
        converted_obj = convert_numpy_types(obj)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_obj, f, **kwargs)
            
        logger.info(f" Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"JSON file serialization failed: {e}")
        # Fallback: try to save string representation
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(str(obj), f, **kwargs)
            logger.warning(f" Saved string representation to {file_path}")
        except Exception as e2:
            logger.error(f" Failed to save even string representation: {e2}")

def safe_json_load(file_path: str) -> Any:
    """
    Safely load JSON from file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON file loading failed: {e}")
        return None

def validate_json_serializable(obj: Any) -> bool:
    """
    Check if object can be safely serialized to JSON
    
    Args:
        obj: Object to validate
        
    Returns:
        True if serializable, False otherwise
    """
    try:
        converted_obj = convert_numpy_types(obj)
        json.dumps(converted_obj)
        return True
    except Exception:
        return False
