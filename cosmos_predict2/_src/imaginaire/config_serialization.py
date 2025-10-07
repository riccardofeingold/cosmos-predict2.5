# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Config serialization utilities for saving and loading Config instances.

This module provides robust serialization/deserialization for the Config class,
handling complex types like torch enums, class types, LazyDict, and nested attrs classes.
"""

import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Type

import attrs
import yaml
from omegaconf import DictConfig, OmegaConf

from cosmos_predict2._src.imaginaire.config import Config
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict


class ConfigSerializer:
    """
    Serializer for Config instances that handles complex types properly.
    
    This serializer converts problematic types (torch enums, class types, etc.)
    into serializable representations and can reconstruct them on load.
    """
    
    # Special markers for type information
    TYPE_MARKER = "__type__"
    MODULE_MARKER = "__module__"
    VALUE_MARKER = "__value__"
    
    @classmethod
    def save(cls, config: Config, filepath: str) -> None:
        """
        Save a Config instance to a YAML file.
        
        Args:
            config: The Config instance to save
            filepath: Path to the output YAML file
            
        Example:
            >>> config = Config(...)
            >>> ConfigSerializer.save(config, "my_config.yaml")
        """
        # Convert config to serializable dict
        serializable_dict = cls._serialize(config)
        
        # Save to YAML
        with open(filepath, 'w') as f:
            yaml.dump(serializable_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> Config:
        """
        Load a Config instance from a YAML file.
        
        Args:
            filepath: Path to the input YAML file
            
        Returns:
            A Config instance reconstructed from the file
            
        Example:
            >>> config = ConfigSerializer.load("my_config.yaml")
        """
        # Load YAML
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Deserialize back to Config
        config = cls._deserialize(data, Config)
        
        print(f"Config loaded from {filepath}")
        return config
    
    @classmethod
    def _serialize(cls, obj: Any) -> Any:
        """
        Recursively serialize an object to a YAML-compatible representation.
        
        Handles:
        - attrs classes (including nested)
        - torch enums and types
        - Python class types
        - LazyDict
        - Callable types
        - Standard Python types
        """
        # Handle None
        if obj is None:
            return None
        
        # Handle attrs classes
        if attrs.has(obj.__class__):
            result = {
                cls.TYPE_MARKER: f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
            }
            
            for field in attrs.fields(obj.__class__):
                value = getattr(obj, field.name)
                result[field.name] = cls._serialize(value)
            
            return result
        
        # Handle LazyDict
        if isinstance(obj, LazyDict):
            return {
                cls.TYPE_MARKER: "LazyDict",
                cls.VALUE_MARKER: cls._serialize(dict(obj))
            }
        
        # Handle DictConfig (from LazyCall)
        if isinstance(obj, DictConfig):
            return {
                cls.TYPE_MARKER: "DictConfig",
                cls.VALUE_MARKER: OmegaConf.to_container(obj, resolve=False)
            }
        
        # Handle torch types
        if hasattr(obj, '__module__') and obj.__module__.startswith('torch'):
            # Handle torch enums (like memory_format)
            if hasattr(obj, 'name') and hasattr(obj.__class__, '__members__'):
                return {
                    cls.TYPE_MARKER: f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                    cls.VALUE_MARKER: obj.name
                }
            # Handle other torch objects
            return {
                cls.TYPE_MARKER: "torch_object",
                cls.VALUE_MARKER: str(obj)
            }
        
        # Handle Python class types
        if isinstance(obj, type):
            return {
                cls.TYPE_MARKER: "class_type",
                cls.VALUE_MARKER: f"{obj.__module__}.{obj.__qualname__}"
            }
        
        # Handle callable objects (but not classes)
        if callable(obj) and not isinstance(obj, type):
            if hasattr(obj, '__name__') and obj.__name__ == '<lambda>':
                return {
                    cls.TYPE_MARKER: "lambda",
                    cls.VALUE_MARKER: "<lambda>"
                }
            # Try to get a reference to the callable
            if hasattr(obj, '__module__') and hasattr(obj, '__qualname__'):
                return {
                    cls.TYPE_MARKER: "callable",
                    cls.VALUE_MARKER: f"{obj.__module__}.{obj.__qualname__}"
                }
            return {
                cls.TYPE_MARKER: "callable",
                cls.VALUE_MARKER: str(obj)
            }
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: cls._serialize(v) for k, v in obj.items()}
        
        # Handle lists
        if isinstance(obj, (list, tuple)):
            serialized_list = [cls._serialize(item) for item in obj]
            if isinstance(obj, tuple):
                return {
                    cls.TYPE_MARKER: "tuple",
                    cls.VALUE_MARKER: serialized_list
                }
            return serialized_list
        
        # Handle primitive types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Fallback: convert to string
        return str(obj)
    
    @classmethod
    def _deserialize(cls, data: Any, target_type: Optional[Type] = None) -> Any:
        """
        Recursively deserialize data back to Python objects.
        
        Args:
            data: The serialized data
            target_type: Optional type hint for reconstruction
            
        Returns:
            The deserialized object
        """
        # Handle None
        if data is None:
            return None
        
        # Handle primitive types
        if isinstance(data, (str, int, float, bool)):
            return data
        
        # Handle lists
        if isinstance(data, list):
            return [cls._deserialize(item) for item in data]
        
        # Handle dictionaries with type information
        if isinstance(data, dict):
            # Check if this is a typed object
            if cls.TYPE_MARKER in data:
                type_str = data[cls.TYPE_MARKER]
                
                # Handle LazyDict
                if type_str == "LazyDict":
                    inner_dict = cls._deserialize(data[cls.VALUE_MARKER])
                    return LazyDict(inner_dict)
                
                # Handle DictConfig
                if type_str == "DictConfig":
                    inner_dict = cls._deserialize(data[cls.VALUE_MARKER])
                    return DictConfig(inner_dict, flags={"allow_objects": True})
                
                # Handle tuples
                if type_str == "tuple":
                    inner_list = cls._deserialize(data[cls.VALUE_MARKER])
                    return tuple(inner_list)
                
                # Handle torch enums
                if type_str.startswith("torch."):
                    try:
                        # Import the torch module and get the enum
                        parts = type_str.rsplit('.', 1)
                        module_path, class_name = parts[0], parts[1]
                        module = importlib.import_module(module_path)
                        enum_class = getattr(module, class_name)
                        
                        if cls.VALUE_MARKER in data:
                            # Get the enum member by name
                            return getattr(enum_class, data[cls.VALUE_MARKER])
                        return enum_class
                    except (ImportError, AttributeError) as e:
                        print(f"Warning: Could not deserialize torch type {type_str}: {e}")
                        return None
                
                # Handle class types
                if type_str == "class_type":
                    try:
                        class_path = data[cls.VALUE_MARKER]
                        parts = class_path.rsplit('.', 1)
                        module_path, class_name = parts[0], parts[1]
                        module = importlib.import_module(module_path)
                        return getattr(module, class_name)
                    except (ImportError, AttributeError) as e:
                        print(f"Warning: Could not deserialize class type {data[cls.VALUE_MARKER]}: {e}")
                        return None
                
                # Handle lambda functions
                if type_str == "lambda":
                    # Cannot reconstruct lambdas, return a placeholder or default
                    print("Warning: Lambda functions cannot be deserialized, using default")
                    return lambda: dict(enabled=False)  # Common default in the codebase
                
                # Handle callable objects
                if type_str == "callable":
                    try:
                        callable_path = data[cls.VALUE_MARKER]
                        if callable_path.startswith("<"):
                            # Cannot reconstruct, return None or placeholder
                            print(f"Warning: Cannot deserialize callable {callable_path}")
                            return None
                        parts = callable_path.rsplit('.', 1)
                        module_path, callable_name = parts[0], parts[1]
                        module = importlib.import_module(module_path)
                        return getattr(module, callable_name)
                    except (ImportError, AttributeError) as e:
                        print(f"Warning: Could not deserialize callable {data[cls.VALUE_MARKER]}: {e}")
                        return None
                
                # Handle attrs classes
                if type_str.startswith("cosmos_predict2."):
                    try:
                        # Import and reconstruct the attrs class
                        parts = type_str.rsplit('.', 1)
                        module_path, class_name = parts[0], parts[1]
                        module = importlib.import_module(module_path)
                        cls_type = getattr(module, class_name)
                        
                        # Reconstruct the object
                        kwargs = {}
                        for key, value in data.items():
                            if key not in (cls.TYPE_MARKER, cls.MODULE_MARKER):
                                kwargs[key] = cls._deserialize(value)
                        
                        return cls_type(**kwargs)
                    except (ImportError, AttributeError, TypeError) as e:
                        print(f"Warning: Could not deserialize attrs class {type_str}: {e}")
                        raise
            
            # Regular dictionary without type information
            return {k: cls._deserialize(v) for k, v in data.items()}
        
        return data


# Convenience functions
def save_config(config: Config, filepath: str) -> None:
    """
    Save a Config instance to a YAML file.
    
    Args:
        config: The Config instance to save
        filepath: Path to the output YAML file
    """
    ConfigSerializer.save(config, filepath)


def load_config(filepath: str) -> Config:
    """
    Load a Config instance from a YAML file.
    
    Args:
        filepath: Path to the input YAML file
        
    Returns:
        A Config instance
    """
    return ConfigSerializer.load(filepath)

