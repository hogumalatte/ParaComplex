import torch
import torch.nn as nn
import torch.optim as optim
import ParaComplex as pc

def train_example():
    """Simple training example using ParaComplex library"""
    
    # Set devices
    device_real, device_imag = pc.set_devices()
    
    # Choose model type
    model_type = "resnet34"  # or "efficientnet_b6"
    
    if model_type == "efficientnet_b6":
        model = pc.TensorParallelEfficientNetB6(
            num_classes=10,
            width_coefficient=1.8,
            depth_coefficient=2.6,
            head_dropout_rate=0.5,
            block_drop_connect_rate=0.2,
            device_real=device_real,
            device_imag=device_imag
        )
        input_size = (3, 224, 224)
    else:  # resnet34
        model = pc.complex_resnet34_tensor_parallel(
            num_classes=10,
            input_channels=3,
            device_real=device_real,
            device_imag=device_imag
        )
        input_size = (3, 32, 32)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    batch_size = 4
    num_batches = 5
    
    print(f"Starting training with {model_type} model...")
    
    for batch_idx in range(num_batches):
        # Create synthetic data
        input_data = torch.randn(batch_size, *input_size, dtype=torch.complex64)
        target = torch.randint(0, 10, (batch_size,)).to(device_real)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    train_example()
