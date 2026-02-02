from typing import List, Dict
import threading
from rl_pdnn.utils import IoTDevice, DNNLayer, generate_iot_network

class ResourceManager:
    """
    Singleton-like class to manage shared resources of IoT devices.
    Thread-safe if we ever move to threaded simulation, though currently sequential-multi-agent.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, num_devices=5):
        # Avoid re-initialization if already created
        if not hasattr(self, 'initialized'):
            self.devices: Dict[int, IoTDevice] = {}
            for d in generate_iot_network(num_devices):
                self.devices[d.device_id] = d
            self.initialized = True

    def reset(self, num_devices=5):
        """Resets the state of all devices (clears memory usage)."""
        self.devices = {}
        for d in generate_iot_network(num_devices):
             self.devices[d.device_id] = d
    
    def get_state_for_device(self, device_id: int) -> List[float]:
        """Returns the normalized OBSERVATION variables for a specific device."""
        d = self.devices[device_id]
        # CPU, Mem Free, Bandwidth, Privacy
        return [
            d.cpu_speed / 2.0,
            (d.memory_capacity - d.current_memory_usage) / 500.0,
            d.bandwidth / 100.0,
            float(d.privacy_clearance)
        ]

    def try_allocate(self, device_id: int, layer: DNNLayer) -> bool:
        """
        Attempts to allocate a layer to a device.
        Returns True if successful (and updates state), False otherwise.
        """
        device = self.devices[device_id]
        
        # Check Privacy
        if device.privacy_clearance < layer.privacy_level:
            return False
            
        # Check Memory
        if device.current_memory_usage + layer.memory_demand > device.memory_capacity:
            return False
            
        # Allocate
        device.current_memory_usage += layer.memory_demand
        return True

    def release(self, device_id: int, memory_amount: float):
        """Releases memory from a device (e.g., after task completion)."""
        if device_id in self.devices:
            self.devices[device_id].current_memory_usage = max(0.0, self.devices[device_id].current_memory_usage - memory_amount)
