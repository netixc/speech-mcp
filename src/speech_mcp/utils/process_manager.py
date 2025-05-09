"""
Process management utilities for speech-mcp.

This module provides tools for managing subprocess creation, monitoring, and cleanup
to ensure that processes are properly tracked and terminated when no longer needed.
"""

import os
import sys
import subprocess
import signal
import time
import threading
import atexit
import psutil
from typing import List, Dict, Set, Optional, Callable, Union, Tuple, Any, IO
import weakref

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="process_mgr")

class ProcessManager:
    """
    Process manager for tracking and cleaning up subprocesses.
    
    This class provides a singleton instance for tracking and managing
    subprocesses across the application. It ensures that processes are
    properly terminated when they are no longer needed or when the
    application exits.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'ProcessManager':
        """Get the singleton instance of ProcessManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the ProcessManager."""
        # Use a lock for thread safety
        self._processes_lock = threading.RLock()
        
        # Track processes by component and process ID
        self._processes: Dict[str, Dict[int, subprocess.Popen]] = {}
        
        # Track exit callbacks
        self._exit_callbacks: Dict[int, Callable] = {}
        
        # Track process timeouts
        self._timeouts: Dict[int, float] = {}
        
        # Register cleanup on program exit
        atexit.register(self.cleanup_all)
        
        # Start background monitoring thread
        self._stop_thread = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_processes, 
            daemon=True,
            name="ProcessMonitorThread"
        )
        self._monitor_thread.start()
        
        logger.info("ProcessManager initialized")
    
    def create_process(self, 
                     cmd: Union[str, List[str]],
                     component: str = "default",
                     shell: bool = False,
                     env: Optional[Dict[str, str]] = None,
                     cwd: Optional[str] = None,
                     detached: bool = False,
                     stdout: Union[int, IO] = subprocess.PIPE,
                     stderr: Union[int, IO] = subprocess.PIPE,
                     timeout_seconds: Optional[float] = None,
                     exit_callback: Optional[Callable[[int, Optional[str]], None]] = None,
                     capture_output: bool = True) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
        """
        Create and register a subprocess.
        
        Args:
            cmd: Command to execute (string or list of arguments)
            component: Component identifier for grouping (default: "default")
            shell: Whether to run the command through the shell (default: False)
            env: Environment variables for the subprocess (default: None = inherit)
            cwd: Working directory for the subprocess (default: None = inherit)
            detached: Whether to detach the process from the parent (default: False)
            stdout: Stdout redirection (default: subprocess.PIPE)
            stderr: Stderr redirection (default: subprocess.PIPE)
            timeout_seconds: Automatically terminate after this many seconds
            exit_callback: Function to call when the process exits
            capture_output: Whether to capture and log process output
        
        Returns:
            Tuple containing:
            - subprocess.Popen instance or None on failure
            - Error message or None on success
        """
        try:
            # Create the subprocess with appropriate settings
            kwargs = {
                'shell': shell,
                'stdout': stdout,
                'stderr': stderr
            }
            
            # Add optional arguments if provided
            if env is not None:
                kwargs['env'] = env
            if cwd is not None:
                kwargs['cwd'] = cwd
                
            # Configure process isolation based on platform
            if detached:
                if sys.platform == 'win32':
                    kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    kwargs['start_new_session'] = True
            
            # Create the process
            logger.info(f"Creating process for component '{component}': {cmd}")
            process = subprocess.Popen(cmd, **kwargs)
            
            # Register the process
            with self._processes_lock:
                if component not in self._processes:
                    self._processes[component] = {}
                self._processes[component][process.pid] = process
                
                # Register timeout if specified
                if timeout_seconds is not None:
                    self._timeouts[process.pid] = time.time() + timeout_seconds
                
                # Register exit callback if specified
                if exit_callback is not None:
                    self._exit_callbacks[process.pid] = exit_callback
            
            logger.info(f"Process created with PID {process.pid} for component '{component}'")
            
            # Set up output monitoring if requested
            if capture_output and stdout == subprocess.PIPE:
                self._start_output_monitor(process, component)
            
            return process, None
            
        except Exception as e:
            error_msg = f"Error creating process: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def _start_output_monitor(self, process: subprocess.Popen, component: str) -> None:
        """Start a thread to monitor process output."""
        def monitor_output():
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    break
                    
                # Read stdout if available
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        try:
                            line_str = line.decode('utf-8').rstrip()
                            logger.debug(f"[{component}:{process.pid}] {line_str}")
                        except Exception:
                            pass
                    else:
                        # End of output
                        break
                else:
                    break
                    
                # Brief pause to avoid spinning
                time.sleep(0.1)
                
            # Process has ended, log any remaining output
            if process.stdout:
                remaining_output = process.stdout.read()
                if remaining_output:
                    try:
                        remaining_str = remaining_output.decode('utf-8').rstrip()
                        for line in remaining_str.splitlines():
                            logger.debug(f"[{component}:{process.pid}] {line}")
                    except Exception:
                        pass
                        
            logger.debug(f"Output monitor for process {process.pid} exited")
        
        # Start the monitor thread
        thread = threading.Thread(
            target=monitor_output,
            daemon=True,
            name=f"ProcessOutput-{process.pid}"
        )
        thread.start()
    
    def _monitor_processes(self) -> None:
        """Monitor processes for completion and timeouts."""
        logger.info("Starting process monitor thread")
        
        while not self._stop_thread:
            try:
                # Sleep briefly to avoid excessive CPU usage
                time.sleep(1)
                
                # Make a copy of process info to avoid holding the lock during checks
                processes_copy = {}
                timeouts_copy = {}
                
                with self._processes_lock:
                    # Gather all processes and timeouts
                    for component, procs in self._processes.items():
                        for pid, proc in procs.items():
                            processes_copy[pid] = (component, proc)
                            
                            # Copy timeout if it exists
                            if pid in self._timeouts:
                                timeouts_copy[pid] = self._timeouts[pid]
                
                # Check each process
                current_time = time.time()
                for pid, (component, proc) in processes_copy.items():
                    try:
                        # Check if process has exited
                        if proc.poll() is not None:
                            exit_code = proc.returncode
                            logger.info(f"Process {pid} (component: {component}) exited with code {exit_code}")
                            
                            # Call exit callback if registered
                            self._handle_process_exit(pid, exit_code)
                            
                            # Remove from tracking
                            with self._processes_lock:
                                if component in self._processes and pid in self._processes[component]:
                                    del self._processes[component][pid]
                                if pid in self._timeouts:
                                    del self._timeouts[pid]
                                    
                        # Check for timeout
                        elif pid in timeouts_copy and current_time >= timeouts_copy[pid]:
                            logger.warning(f"Process {pid} (component: {component}) timed out, terminating")
                            
                            try:
                                self.terminate_process(pid, timeout=5)
                                
                                # Call exit callback with timeout indicator
                                self._handle_process_exit(pid, None, "timeout")
                                
                                # Remove from tracking
                                with self._processes_lock:
                                    if component in self._processes and pid in self._processes[component]:
                                        del self._processes[component][pid]
                                    if pid in self._timeouts:
                                        del self._timeouts[pid]
                            except Exception as e:
                                logger.error(f"Error terminating timed out process {pid}: {str(e)}")
                            
                    except Exception as e:
                        logger.warning(f"Error checking process {pid}: {str(e)}")
                        
                        # Remove from tracking if we can't access it anymore
                        with self._processes_lock:
                            if component in self._processes and pid in self._processes[component]:
                                del self._processes[component][pid]
                            if pid in self._timeouts:
                                del self._timeouts[pid]
                
            except Exception as e:
                logger.error(f"Error in process monitor thread: {str(e)}")
    
    def _handle_process_exit(self, pid: int, exit_code: Optional[int], reason: str = "exit") -> None:
        """Handle process exit including callbacks."""
        with self._processes_lock:
            if pid in self._exit_callbacks:
                callback = self._exit_callbacks[pid]
                del self._exit_callbacks[pid]
                
                try:
                    callback(exit_code, reason)
                except Exception as e:
                    logger.error(f"Error in process exit callback for PID {pid}: {str(e)}")
    
    def terminate_process(self, 
                        pid: int, 
                        force: bool = False, 
                        timeout: float = 5.0) -> bool:
        """
        Terminate a process by PID.
        
        Args:
            pid: Process ID to terminate
            force: Whether to force termination (SIGKILL) if graceful termination fails
            timeout: Time to wait for graceful termination before force termination
        
        Returns:
            bool: True if process was terminated successfully, False otherwise
        """
        try:
            # Try to find the process in our tracking first
            process = None
            component = "unknown"
            
            with self._processes_lock:
                for comp, procs in self._processes.items():
                    if pid in procs:
                        process = procs[pid]
                        component = comp
                        break
            
            if process is not None:
                # We have the Popen object, use subprocess methods
                logger.info(f"Terminating tracked process {pid} (component: {component})")
                
                try:
                    # Try graceful termination first
                    process.terminate()
                    
                    # Wait for process to exit
                    try:
                        process.wait(timeout=timeout)
                        logger.info(f"Process {pid} terminated gracefully")
                        
                        # Call exit callback
                        self._handle_process_exit(pid, process.returncode, "terminated")
                        
                        # Remove from tracking
                        with self._processes_lock:
                            if component in self._processes and pid in self._processes[component]:
                                del self._processes[component][pid]
                            if pid in self._timeouts:
                                del self._timeouts[pid]
                                
                        return True
                    except subprocess.TimeoutExpired:
                        if force:
                            # Graceful termination failed, kill the process
                            logger.warning(f"Process {pid} did not terminate gracefully, killing")
                            process.kill()
                            
                            try:
                                # Wait a bit for the kill to take effect
                                process.wait(timeout=2.0)
                                logger.info(f"Process {pid} killed forcefully")
                                
                                # Call exit callback
                                self._handle_process_exit(pid, process.returncode, "killed")
                            except subprocess.TimeoutExpired:
                                logger.error(f"Failed to kill process {pid}")
                                return False
                            
                            # Remove from tracking
                            with self._processes_lock:
                                if component in self._processes and pid in self._processes[component]:
                                    del self._processes[component][pid]
                                if pid in self._timeouts:
                                    del self._timeouts[pid]
                                    
                            return True
                        else:
                            logger.warning(f"Process {pid} termination timed out and force=False")
                            return False
                except Exception as e:
                    logger.error(f"Error during process {pid} termination: {str(e)}")
                    return False
                    
            else:
                # Process not in our tracking, use psutil
                try:
                    logger.info(f"Terminating untracked process {pid}")
                    
                    if not psutil.pid_exists(pid):
                        logger.info(f"Process {pid} does not exist")
                        return True
                        
                    process = psutil.Process(pid)
                    
                    # Try graceful termination
                    process.terminate()
                    
                    # Wait for process to exit
                    try:
                        process.wait(timeout=timeout)
                        logger.info(f"Untracked process {pid} terminated gracefully")
                        return True
                    except psutil.TimeoutExpired:
                        if force:
                            # Graceful termination failed, kill the process
                            logger.warning(f"Untracked process {pid} did not terminate gracefully, killing")
                            process.kill()
                            
                            try:
                                # Wait a bit for the kill to take effect
                                process.wait(timeout=2.0)
                                logger.info(f"Untracked process {pid} killed forcefully")
                                return True
                            except psutil.TimeoutExpired:
                                logger.error(f"Failed to kill untracked process {pid}")
                                return False
                        else:
                            logger.warning(f"Untracked process {pid} termination timed out and force=False")
                            return False
                except psutil.NoSuchProcess:
                    logger.info(f"Process {pid} does not exist")
                    return True
                except Exception as e:
                    logger.error(f"Error terminating untracked process {pid}: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in terminate_process: {str(e)}")
            return False
    
    def cleanup_component(self, component: str) -> int:
        """
        Clean up all processes for a specific component.
        
        Args:
            component: Component identifier
        
        Returns:
            int: Number of processes terminated
        """
        if not component:
            return 0
            
        pids_to_terminate = []
        
        # Get the list of processes to terminate
        with self._processes_lock:
            if component in self._processes:
                pids_to_terminate = list(self._processes[component].keys())
        
        # Terminate each process
        terminated_count = 0
        for pid in pids_to_terminate:
            if self.terminate_process(pid, force=True):
                terminated_count += 1
        
        logger.info(f"Terminated {terminated_count} processes for component: {component}")
        return terminated_count
    
    def get_process_count(self, component: Optional[str] = None) -> int:
        """
        Get the count of tracked processes.
        
        Args:
            component: Optional component filter
        
        Returns:
            int: Number of tracked processes
        """
        with self._processes_lock:
            if component:
                return len(self._processes.get(component, {}))
            else:
                return sum(len(procs) for procs in self._processes.values())
    
    def cleanup_all(self) -> int:
        """
        Clean up all tracked processes.
        
        Returns:
            int: Number of processes terminated
        """
        logger.info("Cleaning up all processes")
        
        # Stop the monitor thread
        self._stop_thread = True
        
        # Get all processes to terminate
        pids_to_terminate = []
        with self._processes_lock:
            for component, procs in self._processes.items():
                for pid in procs:
                    pids_to_terminate.append((component, pid))
        
        # Terminate each process
        terminated_count = 0
        for component, pid in pids_to_terminate:
            try:
                logger.info(f"Terminating process {pid} (component: {component})")
                if self.terminate_process(pid, force=True):
                    terminated_count += 1
                    
                    # Call exit callback
                    self._handle_process_exit(pid, None, "cleanup")
            except Exception as e:
                logger.error(f"Error terminating process {pid}: {str(e)}")
        
        # Clear all tracking data
        with self._processes_lock:
            self._processes.clear()
            self._timeouts.clear()
            self._exit_callbacks.clear()
        
        logger.info(f"Terminated {terminated_count} processes during cleanup")
        return terminated_count
    
    def is_alive(self, pid: int) -> bool:
        """
        Check if a process is alive.
        
        Args:
            pid: Process ID to check
        
        Returns:
            bool: True if the process is alive, False otherwise
        """
        try:
            # First check if we're tracking this process
            with self._processes_lock:
                for component, procs in self._processes.items():
                    if pid in procs:
                        # Use the Popen object's poll method
                        return procs[pid].poll() is None
            
            # If not tracked, use psutil
            return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

# Create a convenience function to get the singleton instance
def get_process_manager() -> ProcessManager:
    """Get the singleton instance of ProcessManager."""
    return ProcessManager.get_instance()

# Create shortcut functions for common operations
def create_process(**kwargs) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Create and register a subprocess."""
    return get_process_manager().create_process(**kwargs)

def terminate_process(pid: int, **kwargs) -> bool:
    """Terminate a process by PID."""
    return get_process_manager().terminate_process(pid, **kwargs)

def cleanup_component_processes(component: str) -> int:
    """Clean up all processes for a specific component."""
    return get_process_manager().cleanup_component(component)

def cleanup_all_processes() -> int:
    """Clean up all tracked processes."""
    return get_process_manager().cleanup_all()

def is_process_alive(pid: int) -> bool:
    """Check if a process is alive."""
    return get_process_manager().is_alive(pid)

# Initialize the singleton instance
_manager = get_process_manager()