import dataclasses
from typing import List, Dict
import random

@dataclasses.dataclass
class DNNLayer:
    """Represents a single layer of the Neural Network task."""
    layer_id: int
    name: str # e.g. "conv1", "fc1"
    computation_demand: float # Estimated FLOPs or time units
    memory_demand: float # MB required to store weights/output
    output_data_size: float # MB of data to transmit to next layer
    privacy_level: int # Privacy requirement (0=Public, 1=Private)

@dataclasses.dataclass
class IoTDevice:
    """Represents an IoT device in the network."""
    device_id: int
    cpu_speed: float # Relative CPU speed factor (e.g. 1.0 = baseline)
    memory_capacity: float # Total MB
    current_memory_usage: float # Used MB
    bandwidth: float # Mbps link speed
    privacy_clearance: int # Max privacy level this device is allowed to see (0 or 1)
    
    def can_host(self, layer: DNNLayer) -> bool:
        """Check if device has resources and clearance for the layer."""
        if self.current_memory_usage + layer.memory_demand > self.memory_capacity:
            return False # Out of memory
        if self.privacy_clearance < layer.privacy_level:
            return False # Privacy violation
        return True

def generate_specific_model(model_type="lenet") -> List[DNNLayer]:
    """Generates a sequence of layers for specific model profiles."""
    layers = []
    
    if model_type == "lenet": # 6 layers
        profiles = [
            (2.0, 5.0, 3.0),   # Conv1
            (1.0, 5.0, 1.0),   # Conv2
            (1.0, 1.0, 0.5),   # Flatten/Overhead
            (1.5, 120.0, 0.1), # FC1
            (0.5, 10.0, 0.01), # FC2
            (0.1, 0.1, 0.01)   # Output
        ]
        
    elif model_type == "simplecnn": # 5 layers
        profiles = [
             (5.0, 20.0, 15.0), # Conv1 (32)
             (8.0, 40.0, 8.0),  # Conv2 (64)
             (0.5, 0.5, 0.5),   # Flatten
             (5.0, 150.0, 0.2), # Dense (128)
             (0.1, 1.0, 0.01)   # Output
        ]

    elif model_type == "deepcnn": # Deeper, more compute
        profiles = [
             (10.0, 40.0, 20.0), # Block 1 (2x Conv32 + Pool)
             (15.0, 80.0, 10.0), # Block 2 (2x Conv64 + Pool)
             (0.5, 0.5, 0.5),    # Flatten
             (8.0, 200.0, 0.5),  # FC1 (256)
             (0.2, 2.0, 0.01)    # Output
        ]

    elif model_type == "miniresnet": # Skip connections, heavy compute
        profiles = [
             (12.0, 50.0, 25.0), # Block 1 (Conv32)
             (18.0, 60.0, 25.0), # Block 2 (ResBlock)
             (1.0, 1.0, 1.0),    # Pool + Flatten
             (6.0, 150.0, 0.2),  # FC1 (128)
             (0.1, 1.0, 0.01)    # Output
        ]
    else:
        return generate_specific_model("simplecnn") # Fallback

    for i, (comp, mem, out_size) in enumerate(profiles):
        layers.append(DNNLayer(
            layer_id=i,
            name=f"{model_type}_L{i}",
            computation_demand=comp,
            memory_demand=mem,
            output_data_size=out_size,
            privacy_level=1 if i == 0 else 0
        ))
    return layers

def generate_dummy_dnn_model(num_layers=6):
    """Legacy wrapper."""
    return generate_specific_model("lenet")

def generate_iot_network(num_devices=5) -> List[IoTDevice]:
    """Generates a set of heterogeneous IoT devices."""
    devices = []
    for i in range(num_devices):
        devices.append(IoTDevice(
            device_id=i,
            cpu_speed=random.uniform(0.5, 2.0),
            memory_capacity=random.uniform(100, 500), # MB
            current_memory_usage=0.0,
            bandwidth=random.uniform(10, 100), # Mbps
            privacy_clearance=random.choice([0, 1]) # Some trusted, some not
        ))
    return devices
