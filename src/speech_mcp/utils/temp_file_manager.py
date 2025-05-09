"""
Temporary file manager for speech-mcp.

This module provides centralized temporary file management to ensure proper cleanup
of temporary files across the application. It uses a singleton pattern to provide
a global registry of temporary files that can be cleaned up when the application exits.
"""

import os
import tempfile
import threading
import atexit
import time
from typing import List, Dict, Set, Optional, Union, Callable
import weakref

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="temp_manager")

class TempFileManager:
    """
    Centralized manager for temporary files.
    
    This class provides a singleton instance for tracking and cleaning up
    temporary files across the application. It ensures that temporary files
    are properly deleted when they are no longer needed or when the application
    exits.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'TempFileManager':
        """Get the singleton instance of TempFileManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the TempFileManager."""
        # Use a lock for thread safety
        self._files_lock = threading.RLock()
        
        # Track temporary files by component
        self._temp_files: Dict[str, Set[str]] = {}
        
        # Track temporary directories
        self._temp_dirs: Set[str] = set()
        
        # Track auto-cleanup times
        self._cleanup_times: Dict[str, float] = {}
        
        # Register cleanup on program exit
        atexit.register(self.cleanup_all)
        
        # Start background cleanup thread
        self._stop_thread = False
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup, 
            daemon=True,
            name="TempFileCleanupThread"
        )
        self._cleanup_thread.start()
        
        logger.info("TempFileManager initialized")
    
    def create_temp_file(self, 
                       suffix: str = ".tmp", 
                       prefix: str = "speech_mcp_", 
                       directory: Optional[str] = None,
                       component: str = "default",
                       auto_cleanup_seconds: Optional[float] = None) -> str:
        """
        Create a temporary file and register it for cleanup.
        
        Args:
            suffix: File suffix (default: ".tmp")
            prefix: File prefix (default: "speech_mcp_")
            directory: Directory to create the file in (default: system temp)
            component: Component identifier for grouping files (default: "default")
            auto_cleanup_seconds: Automatically clean up after this many seconds
                                 (default: None - no auto-cleanup)
        
        Returns:
            str: Path to the created temporary file
        """
        try:
            # Create the temporary file
            with tempfile.NamedTemporaryFile(
                suffix=suffix, 
                prefix=prefix,
                dir=directory,
                delete=False
            ) as temp_file:
                temp_path = temp_file.name
            
            # Register the file for cleanup
            with self._files_lock:
                if component not in self._temp_files:
                    self._temp_files[component] = set()
                self._temp_files[component].add(temp_path)
                
                # Register auto-cleanup time if specified
                if auto_cleanup_seconds is not None:
                    self._cleanup_times[temp_path] = time.time() + auto_cleanup_seconds
            
            logger.debug(f"Created temp file: {temp_path} (component: {component})")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating temp file: {e}")
            return ""
    
    def create_temp_directory(self, 
                            suffix: str = "", 
                            prefix: str = "speech_mcp_", 
                            parent_dir: Optional[str] = None,
                            component: str = "default",
                            auto_cleanup_seconds: Optional[float] = None) -> str:
        """
        Create a temporary directory and register it for cleanup.
        
        Args:
            suffix: Directory suffix (default: "")
            prefix: Directory prefix (default: "speech_mcp_")
            parent_dir: Parent directory (default: system temp)
            component: Component identifier for grouping (default: "default")
            auto_cleanup_seconds: Automatically clean up after this many seconds
                                 (default: None - no auto-cleanup)
        
        Returns:
            str: Path to the created temporary directory
        """
        try:
            # Create the temporary directory
            temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=parent_dir)
            
            # Register the directory for cleanup
            with self._files_lock:
                self._temp_dirs.add(temp_dir)
                
                # Register auto-cleanup time if specified
                if auto_cleanup_seconds is not None:
                    self._cleanup_times[temp_dir] = time.time() + auto_cleanup_seconds
            
            logger.debug(f"Created temp directory: {temp_dir} (component: {component})")
            return temp_dir
            
        except Exception as e:
            logger.error(f"Error creating temp directory: {e}")
            return ""
    
    def register_file(self, 
                    file_path: str, 
                    component: str = "default",
                    auto_cleanup_seconds: Optional[float] = None) -> bool:
        """
        Register an existing file for cleanup.
        
        Args:
            file_path: Path to the file to register
            component: Component identifier for grouping (default: "default")
            auto_cleanup_seconds: Automatically clean up after this many seconds
                                 (default: None - no auto-cleanup)
        
        Returns:
            bool: True if the file was registered successfully, False otherwise
        """
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Attempted to register non-existent file: {file_path}")
            return False
        
        try:
            # Register the file for cleanup
            with self._files_lock:
                if component not in self._temp_files:
                    self._temp_files[component] = set()
                self._temp_files[component].add(file_path)
                
                # Register auto-cleanup time if specified
                if auto_cleanup_seconds is not None:
                    self._cleanup_times[file_path] = time.time() + auto_cleanup_seconds
            
            logger.debug(f"Registered existing file for cleanup: {file_path} (component: {component})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering file: {e}")
            return False
    
    def unregister_file(self, file_path: str, component: Optional[str] = None) -> bool:
        """
        Unregister a file without deleting it.
        
        Args:
            file_path: Path to the file to unregister
            component: Component identifier (if None, search all components)
        
        Returns:
            bool: True if the file was unregistered, False otherwise
        """
        try:
            with self._files_lock:
                # Remove from auto-cleanup if registered
                if file_path in self._cleanup_times:
                    del self._cleanup_times[file_path]
                
                # If component is specified, only check that component
                if component is not None:
                    if component in self._temp_files and file_path in self._temp_files[component]:
                        self._temp_files[component].remove(file_path)
                        logger.debug(f"Unregistered file: {file_path} (component: {component})")
                        return True
                    return False
                
                # Otherwise, search all components
                for comp, files in self._temp_files.items():
                    if file_path in files:
                        files.remove(file_path)
                        logger.debug(f"Unregistered file: {file_path} (component: {comp})")
                        return True
                
                # Check temp directories
                if file_path in self._temp_dirs:
                    self._temp_dirs.remove(file_path)
                    logger.debug(f"Unregistered directory: {file_path}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error unregistering file: {e}")
            return False
    
    def cleanup_file(self, file_path: str, component: Optional[str] = None) -> bool:
        """
        Delete a temporary file and unregister it.
        
        Args:
            file_path: Path to the file to clean up
            component: Component identifier (if None, search all components)
        
        Returns:
            bool: True if the file was cleaned up, False otherwise
        """
        if not file_path:
            return False
            
        # Unregister the file first
        self.unregister_file(file_path, component)
        
        # Delete the file if it exists
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    # Try to remove directory
                    try:
                        os.rmdir(file_path)
                        logger.debug(f"Removed directory: {file_path}")
                        return True
                    except OSError:
                        # Directory might not be empty, log but don't fail
                        logger.warning(f"Could not remove directory (might not be empty): {file_path}")
                        return False
                else:
                    # Remove regular file
                    os.unlink(file_path)
                    logger.debug(f"Deleted file: {file_path}")
                    return True
            else:
                logger.debug(f"File already deleted: {file_path}")
                return True
                
        except Exception as e:
            logger.warning(f"Error deleting file {file_path}: {e}")
            return False
    
    def cleanup_component(self, component: str) -> int:
        """
        Clean up all temporary files for a specific component.
        
        Args:
            component: Component identifier
        
        Returns:
            int: Number of files cleaned up
        """
        if not component:
            return 0
            
        files_to_cleanup = []
        
        # Get the list of files to clean up
        with self._files_lock:
            if component in self._temp_files:
                files_to_cleanup = list(self._temp_files[component])
                self._temp_files[component].clear()
        
        # Clean up each file
        cleaned_count = 0
        for file_path in files_to_cleanup:
            if self.cleanup_file(file_path):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} files for component: {component}")
        return cleaned_count
    
    def _background_cleanup(self):
        """Background thread for automatic cleanup of aged temporary files."""
        logger.info("Starting background cleanup thread")
        
        while not self._stop_thread:
            try:
                # Sleep for a bit to avoid excessive CPU usage
                time.sleep(30)
                
                # Check for files to clean up
                current_time = time.time()
                files_to_cleanup = []
                
                with self._files_lock:
                    # Find files that have exceeded their cleanup time
                    for file_path, cleanup_time in list(self._cleanup_times.items()):
                        if current_time >= cleanup_time:
                            files_to_cleanup.append(file_path)
                            del self._cleanup_times[file_path]
                
                # Clean up the files outside the lock
                if files_to_cleanup:
                    logger.info(f"Auto-cleaning {len(files_to_cleanup)} temporary files")
                    for file_path in files_to_cleanup:
                        self.cleanup_file(file_path)
                        
            except Exception as e:
                logger.error(f"Error in background cleanup thread: {e}")
    
    def cleanup_all(self) -> int:
        """
        Clean up all temporary files and directories.
        
        Returns:
            int: Number of files and directories cleaned up
        """
        logger.info("Cleaning up all temporary files")
        
        # Stop the background thread
        self._stop_thread = True
        
        # Get all files and directories to clean up
        all_files = []
        all_dirs = []
        
        with self._files_lock:
            # Collect all files
            for component, files in self._temp_files.items():
                all_files.extend(files)
                files.clear()
            
            # Collect all directories
            all_dirs = list(self._temp_dirs)
            self._temp_dirs.clear()
            
            # Clear cleanup times
            self._cleanup_times.clear()
        
        # Clean up files first
        cleaned_count = 0
        for file_path in all_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Deleted file: {file_path}")
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Error deleting file {file_path}: {e}")
        
        # Then clean up directories (in reverse order to handle nested dirs)
        all_dirs.sort(reverse=True)  # Longer paths first to handle nested dirs
        for dir_path in all_dirs:
            try:
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                        logger.debug(f"Removed directory: {dir_path}")
                        cleaned_count += 1
                    except OSError:
                        logger.warning(f"Could not remove directory (might not be empty): {dir_path}")
            except Exception as e:
                logger.warning(f"Error deleting directory {dir_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files and directories")
        return cleaned_count

# Create a convenience function to get the singleton instance
def get_temp_manager() -> TempFileManager:
    """Get the singleton instance of TempFileManager."""
    return TempFileManager.get_instance()

# Create shortcut functions for common operations
def create_temp_file(**kwargs) -> str:
    """Create a temporary file and register it for cleanup."""
    return get_temp_manager().create_temp_file(**kwargs)

def create_temp_directory(**kwargs) -> str:
    """Create a temporary directory and register it for cleanup."""
    return get_temp_manager().create_temp_directory(**kwargs)

def register_temp_file(file_path: str, **kwargs) -> bool:
    """Register an existing file for cleanup."""
    return get_temp_manager().register_file(file_path, **kwargs)

def cleanup_temp_file(file_path: str, **kwargs) -> bool:
    """Delete a temporary file and unregister it."""
    return get_temp_manager().cleanup_file(file_path, **kwargs)

def cleanup_all_temp_files() -> int:
    """Clean up all temporary files and directories."""
    return get_temp_manager().cleanup_all()

# Initialize the singleton instance
_manager = get_temp_manager()