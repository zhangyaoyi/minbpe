import logging
import os
import threading

# Global variables for shared logging configuration
_logging_initialized = False
_logging_lock = threading.Lock()
_global_logger = None

def setup_logging(log_dir="logs", log_file="tokenizer.log", log_level=logging.INFO, logger_name=None):
    """
    Setup shared logging configuration for the entire system
    This function ensures that logging is configured only once globally
    
    Args:
        log_dir: Directory to store log files
        log_file: Name of the log file
        log_level: Logging level
        logger_name: Name of the logger (if None, returns root logger)
    
    Returns:
        Logger instance
    """
    global _logging_initialized, _global_logger
    
    with _logging_lock:
        # Configure global logging only once
        if not _logging_initialized:
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            
            # Clear existing handlers to avoid duplicates
            root_logger.handlers.clear()
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            root_logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
            
            _logging_initialized = True
            _global_logger = root_logger
            
            # Log the initialization
            _global_logger.info(f"Global logging initialized - Console and file output: {log_path}")
    
    # Return appropriate logger
    if logger_name:
        return logging.getLogger(logger_name)
    else:
        return logging.getLogger()

def get_logger(name):
    """
    Get a logger instance with the shared configuration
    
    Args:
        name: Name of the logger
    
    Returns:
        Logger instance with shared configuration
    """
    # Ensure logging is initialized with default settings if not already done
    if not _logging_initialized:
        setup_logging()
    
    return logging.getLogger(name)

def reset_logging():
    """
    Reset the global logging configuration (mainly for testing)
    """
    global _logging_initialized, _global_logger
    
    with _logging_lock:
        if _global_logger:
            # Clear all handlers
            for handler in _global_logger.handlers[:]:
                handler.close()
                _global_logger.removeHandler(handler)
        
        _logging_initialized = False
        _global_logger = None