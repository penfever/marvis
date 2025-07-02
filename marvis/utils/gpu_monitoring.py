#!/usr/bin/env python3
"""
GPU monitoring utilities for WandB integration.
Provides a reusable abstraction for logging GPU utilization across scripts.
"""

import logging
import time
import threading
from typing import Optional, Dict, Any
import torch

# Import conditionally to avoid dependency issues
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class GPUMonitor:
    """GPU monitoring utility that logs GPU utilization and memory usage to WandB."""
    
    def __init__(self, log_interval: float = 30.0, enable_detailed_logging: bool = True):
        """
        Initialize GPU monitor.
        
        Args:
            log_interval: Interval in seconds between GPU measurements
            enable_detailed_logging: Whether to log detailed GPU metrics beyond WandB's built-in system stats
        """
        self.log_interval = log_interval
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.gpu_stats = []
        
        # Initialize NVIDIA ML if available
        self.nvml_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.logger.info("NVIDIA ML initialized for detailed GPU monitoring")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVIDIA ML: {e}")
        
        # Check GPU availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.gpu_count = torch.cuda.device_count()
            self.logger.info(f"CUDA available with {self.gpu_count} GPU(s)")
        else:
            self.gpu_count = 0
            self.logger.info("CUDA not available")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics."""
        stats = {}
        
        if not self.cuda_available:
            return stats
        
        # PyTorch-based stats (always available if CUDA is available)
        for i in range(self.gpu_count):
            try:
                # Memory stats from PyTorch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                    max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
                    
                    stats[f'gpu_{i}_memory_allocated_gb'] = allocated
                    stats[f'gpu_{i}_memory_reserved_gb'] = reserved
                    stats[f'gpu_{i}_memory_max_allocated_gb'] = max_allocated
                    
                    # Get device properties
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory / 1024**3  # GB
                    stats[f'gpu_{i}_memory_total_gb'] = total_memory
                    stats[f'gpu_{i}_memory_utilization_percent'] = (allocated / total_memory) * 100
                    
            except Exception as e:
                self.logger.warning(f"Failed to get PyTorch GPU stats for GPU {i}: {e}")
        
        # PYNVML-based stats (more detailed if available)
        if self.nvml_initialized and self.enable_detailed_logging:
            try:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats[f'gpu_{i}_utilization_percent'] = util.gpu
                    stats[f'gpu_{i}_memory_util_percent'] = util.memory
                    
                    # Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        stats[f'gpu_{i}_temperature_celsius'] = temp
                    except:
                        pass  # Temperature might not be available
                    
                    # Power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                        stats[f'gpu_{i}_power_watts'] = power
                    except:
                        pass  # Power might not be available
                    
                    # Clock speeds
                    try:
                        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        stats[f'gpu_{i}_memory_clock_mhz'] = mem_clock
                        stats[f'gpu_{i}_graphics_clock_mhz'] = gpu_clock
                    except:
                        pass  # Clocks might not be available
                        
            except Exception as e:
                self.logger.warning(f"Failed to get detailed GPU stats: {e}")
        
        # GPUtil-based stats (alternative if PYNVML not available)
        elif GPUTIL_AVAILABLE and self.enable_detailed_logging and not self.nvml_initialized:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    stats[f'gpu_{i}_utilization_percent'] = gpu.load * 100
                    stats[f'gpu_{i}_memory_util_percent'] = gpu.memoryUtil * 100
                    stats[f'gpu_{i}_temperature_celsius'] = gpu.temperature
                    stats[f'gpu_{i}_memory_used_mb'] = gpu.memoryUsed
                    stats[f'gpu_{i}_memory_total_mb'] = gpu.memoryTotal
                    stats[f'gpu_{i}_memory_free_mb'] = gpu.memoryFree
            except Exception as e:
                self.logger.warning(f"Failed to get GPUtil stats: {e}")
        
        return stats
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_gpu_stats()
                if stats and WANDB_AVAILABLE and wandb.run is not None:
                    # Add timestamp
                    stats['gpu_monitor_timestamp'] = time.time()
                    wandb.log(stats)
                    
                # Store stats for average calculation
                self.gpu_stats.append(stats)
                
                # Keep only recent stats (last 100 measurements)
                if len(self.gpu_stats) > 100:
                    self.gpu_stats = self.gpu_stats[-100:]
                    
            except Exception as e:
                self.logger.warning(f"Error in GPU monitoring loop: {e}")
            
            time.sleep(self.log_interval)
    
    def start_monitoring(self):
        """Start background GPU monitoring."""
        if not self.cuda_available:
            self.logger.info("GPU monitoring not started - CUDA not available")
            return
            
        if self.monitoring:
            self.logger.warning("GPU monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Started GPU monitoring with {self.log_interval}s interval")
    
    def stop_monitoring(self):
        """Stop background GPU monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Stopped GPU monitoring")
    
    def get_average_stats(self) -> Dict[str, float]:
        """Get average GPU statistics from monitoring period."""
        if not self.gpu_stats:
            return {}
        
        # Calculate averages for numeric fields
        avg_stats = {}
        numeric_fields = []
        
        # Find all numeric fields from the first stat entry
        if self.gpu_stats:
            for key, value in self.gpu_stats[0].items():
                if isinstance(value, (int, float)) and not key.endswith('_timestamp'):
                    numeric_fields.append(key)
        
        # Calculate averages
        for field in numeric_fields:
            values = []
            for stat in self.gpu_stats:
                if field in stat and isinstance(stat[field], (int, float)):
                    values.append(stat[field])
            
            if values:
                avg_stats[f'avg_{field}'] = sum(values) / len(values)
                avg_stats[f'max_{field}'] = max(values)
                avg_stats[f'min_{field}'] = min(values)
        
        return avg_stats
    
    def log_final_stats(self):
        """Log final average statistics to WandB."""
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        
        avg_stats = self.get_average_stats()
        if avg_stats:
            wandb.log({f'final_gpu_stats/{k}': v for k, v in avg_stats.items()})
            self.logger.info("Logged final GPU statistics to WandB")


def init_wandb_with_gpu_monitoring(
    project: str,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    enable_system_monitoring: bool = True,
    gpu_log_interval: float = 30.0,
    enable_detailed_gpu_logging: bool = True
) -> Optional[GPUMonitor]:
    """
    Initialize WandB with GPU monitoring enabled.
    
    Args:
        project: WandB project name
        entity: WandB entity name
        name: WandB run name
        config: Configuration dictionary to log
        output_dir: Output directory for WandB files
        enable_system_monitoring: Enable WandB's built-in system monitoring
        gpu_log_interval: Interval for custom GPU logging
        enable_detailed_gpu_logging: Enable detailed GPU metrics beyond WandB's built-in stats
    
    Returns:
        GPUMonitor instance if GPU monitoring is enabled, None otherwise
    """
    if not WANDB_AVAILABLE:
        logging.getLogger(__name__).warning("WandB not available, skipping GPU monitoring")
        return None
    
    # Configure WandB settings for system monitoring
    settings = None
    if enable_system_monitoring:
        settings = wandb.Settings(
            _disable_stats=False,  # Enable system stats (includes GPU if available)
            _disable_meta=False    # Enable metadata collection
        )
    
    # Initialize WandB
    wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        dir=output_dir,
        settings=settings
    )
    
    # Initialize custom GPU monitor
    gpu_monitor = None
    if enable_detailed_gpu_logging and torch.cuda.is_available():
        gpu_monitor = GPUMonitor(
            log_interval=gpu_log_interval,
            enable_detailed_logging=enable_detailed_gpu_logging
        )
        gpu_monitor.start_monitoring()
    
    return gpu_monitor


def cleanup_gpu_monitoring(gpu_monitor: Optional[GPUMonitor]):
    """Clean up GPU monitoring and log final stats."""
    if gpu_monitor:
        gpu_monitor.stop_monitoring()
        gpu_monitor.log_final_stats()