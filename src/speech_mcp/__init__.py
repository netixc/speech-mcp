import argparse
import sys
import os
import signal
import atexit
import faulthandler
from .server import mcp, cleanup_ui_process
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__)

# Enable faulthandler to help debug segfaults and deadlocks
faulthandler.enable()

# Initialize resource managers early for proper lifecycle management
# Initialize temp file manager
try:
    from speech_mcp.utils.temp_file_manager import get_temp_manager
    # Initialize the temp file manager
    temp_manager = get_temp_manager()
    logger.debug("Temp file manager initialized")
except Exception as e:
    logger.warning(f"Failed to initialize temp file manager: {e}")

# Initialize process manager
try:
    from speech_mcp.utils.process_manager import get_process_manager
    # Initialize the process manager
    process_manager = get_process_manager()
    logger.debug("Process manager initialized")
except Exception as e:
    logger.warning(f"Failed to initialize process manager: {e}")

# Ensure UI process is cleaned up on exit
atexit.register(cleanup_ui_process)

# Centralized cleanup function
def cleanup_resources():
    """Clean up all resources before exit."""
    logger.info("Cleaning up resources...")
    
    # Clean up UI process
    try:
        cleanup_ui_process()
    except Exception as e:
        logger.warning(f"Error cleaning up UI process: {e}")
    
    # Clean up processes
    try:
        from speech_mcp.utils.process_manager import cleanup_all_processes
        processes_cleaned = cleanup_all_processes()
        logger.info(f"Cleaned up {processes_cleaned} processes")
    except Exception as e:
        logger.warning(f"Error cleaning up processes: {e}")
    
    # Clean up temp files (do this last as some processes might use them)
    try:
        from speech_mcp.utils.temp_file_manager import cleanup_all_temp_files
        files_cleaned = cleanup_all_temp_files()
        logger.info(f"Cleaned up {files_cleaned} temporary files")
    except Exception as e:
        logger.warning(f"Error cleaning up temp files: {e}")

# Register cleanup function
atexit.register(cleanup_resources)

# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    # Log the signal
    logger.info(f"Received signal {sig}, shutting down...")
    
    # Dump stack traces to help identify where threads might be stuck
    faulthandler.dump_traceback(file=sys.stderr)
    
    # Clean up all resources
    cleanup_resources()
        
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Speech MCP: Voice interaction with speech recognition."""
    try:
        # Check if stdin/stdout are available
        if sys.stdin is None or sys.stdout is None or sys.stdin.closed or sys.stdout.closed:
            print("Error: stdin or stdout is closed or not available, cannot run MCP server")
            
            # Create a special file to indicate the error
            try:
                os.makedirs(os.path.expanduser("~/.speech-mcp"), exist_ok=True)
                with open(os.path.expanduser("~/.speech-mcp/startup_error.log"), "w") as f:
                    f.write("Error: stdin or stdout is closed or not available, cannot run MCP server")
            except Exception as e:
                print(f"Failed to write error log: {e}")
                pass
                
            sys.exit(1)
            
        logger.info("Starting Speech MCP server...")
        
        parser = argparse.ArgumentParser(
            description="Voice interaction with speech recognition."
        )
        parser.parse_args()
        
        logger.info("Running MCP server...")
        
        try:
            # Run the server
            mcp.run()
        finally:
            # Ensure we clean up resources regardless of how the server exits
            logger.info("MCP server exiting, cleaning up resources")
            cleanup_resources()
                
    except Exception as e:
        # Log the exception
        logger.exception(f"Error running MCP server: {e}")
        
        # Dump stack traces on unhandled exceptions as well
        faulthandler.dump_traceback(file=sys.stderr)
        
        # Final attempt to clean up resources
        cleanup_resources()
            
        sys.exit(1)

if __name__ == "__main__":
    main()