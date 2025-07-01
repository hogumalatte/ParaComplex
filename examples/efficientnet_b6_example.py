import torch
import ParaComplex as pc

def main():
    # Set devices
    device_real, device_imag = pc.set_devices()
    
    # Create EfficientNetB6 model
    model = pc.TensorParallelEfficientNetB6(
        num_classes=10,
        width_coefficient=1.8,
        depth_coefficient=2.6,
        head_dropout_rate=0.5,
        block_drop_connect_rate=0.2,
        device_real=device_real,
        device_imag=device_imag
    )
    
    print("EfficientNetB6 model created successfully!")
    
    # Example input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224, dtype=torch.complex64)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0][:5]}")

if __name__ == "__main__":
    main()
