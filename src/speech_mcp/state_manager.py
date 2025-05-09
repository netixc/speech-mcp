"""
StateManager module for centralized state management in speech-mcp.

This module provides a singleton StateManager class that handles all application state,
replacing the previous file-based state management system.
"""

from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
import os

# Default speech state
DEFAULT_SPEECH_STATE = {
    "listening": False,
    "speaking": False,
    "last_transcript": "",
    "last_response": "",
    "ui_active": False,
    "ui_process_id": None,
    "error": None,
    "debug_mode": False,
    "voice_preference": None
}

class StateManager:
    """
    Singleton class for managing application state.
    
    This class centralizes all state management and provides an observer pattern
    for components to react to state changes.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'StateManager':
        """Get or create the singleton instance."""
        if not cls._instance:
            cls._instance = StateManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the state manager with default values."""
        if StateManager._instance is not None:
            raise RuntimeError("Use get_instance() instead of constructor")
            
        self.state = DEFAULT_SPEECH_STATE.copy()
        self.observers: Dict[str, List[Callable]] = {
            'all': [],  # Observers that want all state updates
            'listening': [],  # Observers for specific state keys
            'speaking': [],
            'transcription': [],
            'response': [],
            'voice_preference': [],
            'ui_ready': [],
            'error': []
        }
        
        # Load any persisted state
        self._load_persisted_state()
    
    def update_state(self, updates: Dict[str, Any], persist: bool = False) -> None:
        """
        Update the state with new values and notify observers.
        
        Args:
            updates: Dictionary of state updates
            persist: Whether to persist this state update to disk
        """
        changed_keys = set()
        
        for key, value in updates.items():
            if key in self.state and self.state[key] != value:
                self.state[key] = value
                changed_keys.add(key)
        
        if changed_keys:
            self._notify_observers(changed_keys)
            
            if persist:
                self._persist_state()
    
    def get_state(self, key: Optional[str] = None) -> Any:
        """
        Get current state value(s).
        
        Args:
            key: Optional specific state key to retrieve
            
        Returns:
            The entire state dict if no key provided, or the specific value
        """
        if key is None:
            return self.state.copy()
        return self.state.get(key)
    
    def add_observer(self, observer: Callable, keys: Optional[List[str]] = None) -> None:
        """
        Add an observer function to be notified of state changes.
        
        Args:
            observer: Callback function that takes state as parameter
            keys: Optional list of specific state keys to observe
        """
        if keys is None:
            self.observers['all'].append(observer)
        else:
            for key in keys:
                if key in self.observers:
                    self.observers[key].append(observer)
    
    def remove_observer(self, observer: Callable, keys: Optional[List[str]] = None) -> None:
        """
        Remove an observer function.
        
        Args:
            observer: The observer function to remove
            keys: Optional list of specific state keys to stop observing
        """
        if keys is None:
            if observer in self.observers['all']:
                self.observers['all'].remove(observer)
        else:
            for key in keys:
                if key in self.observers and observer in self.observers[key]:
                    self.observers[key].remove(observer)
    
    def _notify_observers(self, changed_keys: set) -> None:
        """
        Notify relevant observers of state changes.
        
        Args:
            changed_keys: Set of state keys that changed
        """
        # First notify specific observers
        for key in changed_keys:
            if key in self.observers:
                for observer in self.observers[key]:
                    try:
                        observer({key: self.state[key]})
                    except Exception as e:
                        print(f"Error notifying observer for key {key}: {e}")
        
        # Then notify general observers
        for observer in self.observers['all']:
            try:
                observer(self.state)
            except Exception as e:
                print(f"Error notifying general observer: {e}")
    
    def _persist_state(self) -> None:
        """Persist current state to disk."""
        try:
            config_dir = Path(os.path.expanduser('~/.config/speech-mcp'))
            config_dir.mkdir(parents=True, exist_ok=True)
            
            state_file = config_dir / 'state.json'
            
            # Only persist certain keys
            persist_keys = {'voice_preference', 'debug_mode'}
            persist_state = {k: v for k, v in self.state.items() if k in persist_keys}
            
            with state_file.open('w') as f:
                json.dump(persist_state, f, indent=2)
        except Exception as e:
            print(f"Error persisting state: {e}")
    
    def _load_persisted_state(self) -> None:
        """Load persisted state from disk."""
        try:
            state_file = Path(os.path.expanduser('~/.config/speech-mcp/state.json'))
            if state_file.exists():
                with state_file.open() as f:
                    persisted_state = json.load(f)
                    self.state.update(persisted_state)
        except Exception as e:
            print(f"Error loading persisted state: {e}")