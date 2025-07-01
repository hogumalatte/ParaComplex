import torch
import ParaComplex as pc

def main():
    # Set devices
    device_real, device_imag = pc.set_devices()
    
    # Create ResNet34 model
    model = pc.complex_resnet34_tensor_parallel(
        num_classes=10,
        input_channels=3,
        device_real=device_real,
        device_imag=device_imag
    )
    
    print("ResNet34 model created successfully!")
    
    # Example input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32, dtype=torch.complex64)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0][:5]}")

if __name__ == "__main__":
    main()
