# ParaComplex

**복소수 신경망을 위한 모델 병렬화 라이브러리**

ParaComplex는 PyTorch 기반의 복소수 값 신경망(Complex-Valued Neural Networks)에서 효율적인 모델 병렬화를 지원하는 라이브러리입니다. 복소수 연산의 실수부와 허수부를 서로 다른 GPU에서 처리하여 메모리 효율성과 연산 속도를 개선합니다.

## 🚀 주요 기능

- **모델 병렬화**: 복소수 연산의 실수부와 허수부를 다른 GPU에서 병렬 처리
- **완전한 복소수 레이어**: Conv2d, Linear, BatchNorm, Dropout 등 모든 기본 레이어 지원
- **사전 구현된 모델**: ResNet-34, EfficientNet-B6 등 인기 있는 아키텍처
- **쉬운 사용법**: 간단한 팩토리 함수로 모델 생성
- **메모리 효율성**: 대형 모델도 상대적으로 적은 GPU 메모리로 학습 가능

## 📦 설치

```bash
# 개발 모드로 설치 (권장)
pip install -e .

# 또는 직접 임포트
import sys
sys.path.append('/path/to/ParaComplex')
import ParaComplex as pc
```

## 🔧 요구사항

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA 지원 GPU 2개 이상
- numpy

## 🏃‍♂️ 빠른 시작

### 1. ResNet-34 모델 생성

```python
import ParaComplex as pc
import torch

# 모델 생성 (GPU 0에 실수부, GPU 1에 허수부)
model = pc.create_resnet34(
    num_classes=10,
    input_channels=3,
    activation_fn='relu',
    device_real='cuda:0',
    device_imag='cuda:1'
)

# 복소수 입력으로 추론
input_data = torch.randn(32, 3, 32, 32, dtype=torch.complex64)
output = model(input_data)
```

### 2. EfficientNet-B6 모델 생성

```python
# EfficientNet-B6 모델 생성
model = pc.create_efficientnet_b6(
    num_classes=1000,
    width_coefficient=1.8,
    depth_coefficient=2.6,
    device_real='cuda:0',
    device_imag='cuda:1'
)

print(f"파라미터 수: {pc.count_parameters(model):,}")
```

### 3. 사용자 정의 모델

```python
import torch.nn as nn

class CustomComplexModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.device_real = torch.device('cuda:0')
        self.device_imag = torch.device('cuda:1')
        
        self.conv1 = pc.ComplexConv2dModelParallel(
            3, 64, kernel_size=3, padding=1,
            device_real=self.device_real, 
            device_imag=self.device_imag
        )
        self.bn1 = pc.ComplexBatchNorm2dModelParallel(
            64, device_real=self.device_real, 
            device_imag=self.device_imag
        )
        self.fc = pc.ComplexLinearModelParallel(
            64, num_classes,
            device_real=self.device_real,
            device_imag=self.device_imag
        )
    
    def forward(self, x):
        if torch.is_complex(x):
            x_r, x_i = x.real, x.imag
        else:
            x_r, x_i = x, torch.zeros_like(x)
        
        # 복소수 연산
        out_r, out_i = self.conv1((x_r, x_i))
        out_r, out_i = self.bn1((out_r, out_i))
        out_r, out_i = pc.complex_relu_modelparallel(
            (out_r, out_i), 
            device_real=self.device_real, 
            device_imag=self.device_imag
        )
        
        # 분류를 위한 크기 계산
        out_r_calc = out_r.to(self.device_real)
        out_i_calc = out_i.to(self.device_real)
        magnitude = torch.sqrt(out_r_calc**2 + out_i_calc**2 + 1e-9)
        
        return magnitude
```

## 📚 주요 구성 요소

### 핵심 레이어 (Core Layers)

- `ComplexConv2dModelParallel`: 복소수 2D 컨볼루션
- `ComplexLinearModelParallel`: 복소수 선형 레이어
- `ComplexBatchNorm2dModelParallel`: 복소수 배치 정규화
- `ComplexDropoutModelParallel`: 복소수 드롭아웃
- `ComplexAdaptiveAvgPool2d`: 복소수 적응형 평균 풀링

### 활성화 함수

- `complex_relu_modelparallel`: 복소수 ReLU
- `complex_silu_modelparallel`: 복소수 SiLU (Swish)

### 블록 (Blocks)

- `ComplexBasicBlockModelParallel`: ResNet 기본 블록
- `ComplexMBConvBlockModelParallel`: EfficientNet MBConv 블록
- `ComplexSEBlockModelParallel`: Squeeze-and-Excitation 블록

### 유틸리티 함수

- `set_seed()`: 재현 가능한 결과를 위한 시드 설정
- `count_parameters()`: 모델의 파라미터 수 계산
- `get_model_size_mb()`: 모델 크기 계산
- `synchronize_devices()`: GPU 동기화
- `get_memory_usage()`: GPU 메모리 사용량 확인

## 🎯 예제 실행

```python
# 모든 예제 실행
from ParaComplex.examples import run_all_examples
run_all_examples()

# 개별 예제 실행
from ParaComplex.examples import example_resnet34_training
example_resnet34_training()
```

## 🔍 모델 병렬화 원리

ParaComplex는 복소수 연산의 수학적 특성을 활용합니다:

복소수 곱셈: `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`

- **실수부 계산**: `ac - bd` → GPU 0에서 처리
- **허수부 계산**: `ad + bc` → GPU 1에서 처리
- **파라미터 공유**: 가중치는 GPU 0에 저장하고 필요시 GPU 1로 복사
- **그래디언트 집계**: 자동으로 GPU 0에서 그래디언트 집계

## 📊 메모리 효율성

전통적인 단일 GPU 방식 대비:
- **메모리 사용량**: ~50% 감소 (두 GPU에 분산)
- **배치 크기**: 약 2배 증가 가능
- **모델 크기**: 더 큰 모델 학습 가능

## ⚡ 성능 최적화 팁

1. **비동기 전송 사용**:
   ```python
   # 내부적으로 non_blocking=True 사용
   x_r.to(device_real, non_blocking=True)
   ```

2. **디바이스 동기화**:
   ```python
   pc.synchronize_devices('cuda:0', 'cuda:1')
   ```

3. **메모리 정리**:
   ```python
   pc.clear_cache()
   ```

## 🚧 제한사항

- **GPU 요구사항**: 최소 2개의 CUDA GPU 필요
- **통신 오버헤드**: GPU 간 데이터 전송으로 인한 약간의 오버헤드
- **메모리 불균형**: 실수부 계산이 더 복잡할 수 있음

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 문의

질문이나 제안사항이 있으시면 이슈를 생성해주세요.

---

**ParaComplex**로 복소수 신경망의 새로운 가능성을 탐험해보세요! 🚀 