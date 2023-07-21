# ERA-SESSION10

## Sample Images
![image](https://github.com/RaviNaik/ERA-SESSION10/assets/23289802/7d613bed-97e8-4daf-964b-c9c1f373f23e)

## Model Summary
```python3
============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Kernel Shape
============================================================================================================================================
CustomResnet                             [512, 3, 32, 32]          [512, 10]                 --                        --
├─Sequential: 1-1                        [512, 3, 32, 32]          [512, 64, 32, 32]         --                        --
│    └─Conv2d: 2-1                       [512, 3, 32, 32]          [512, 64, 32, 32]         1,728                     [3, 3]
│    └─BatchNorm2d: 2-2                  [512, 64, 32, 32]         [512, 64, 32, 32]         128                       --
│    └─ReLU: 2-3                         [512, 64, 32, 32]         [512, 64, 32, 32]         --                        --
├─Sequential: 1-2                        [512, 64, 32, 32]         [512, 128, 16, 16]        --                        --
│    └─Conv2d: 2-4                       [512, 64, 32, 32]         [512, 128, 32, 32]        73,728                    [3, 3]
│    └─MaxPool2d: 2-5                    [512, 128, 32, 32]        [512, 128, 16, 16]        --                        2
│    └─BatchNorm2d: 2-6                  [512, 128, 16, 16]        [512, 128, 16, 16]        256                       --
│    └─ReLU: 2-7                         [512, 128, 16, 16]        [512, 128, 16, 16]        --                        --
│    └─ResBlock: 2-8                     [512, 128, 16, 16]        [512, 128, 16, 16]        --                        --
│    │    └─Sequential: 3-1              [512, 128, 16, 16]        [512, 128, 16, 16]        295,424                   --
├─Sequential: 1-3                        [512, 128, 16, 16]        [512, 256, 8, 8]          --                        --
│    └─Conv2d: 2-9                       [512, 128, 16, 16]        [512, 256, 16, 16]        294,912                   [3, 3]
│    └─MaxPool2d: 2-10                   [512, 256, 16, 16]        [512, 256, 8, 8]          --                        2
│    └─BatchNorm2d: 2-11                 [512, 256, 8, 8]          [512, 256, 8, 8]          512                       --
│    └─ReLU: 2-12                        [512, 256, 8, 8]          [512, 256, 8, 8]          --                        --
├─Sequential: 1-4                        [512, 256, 8, 8]          [512, 512, 4, 4]          --                        --
│    └─Conv2d: 2-13                      [512, 256, 8, 8]          [512, 512, 8, 8]          1,179,648                 [3, 3]
│    └─MaxPool2d: 2-14                   [512, 512, 8, 8]          [512, 512, 4, 4]          --                        2
│    └─BatchNorm2d: 2-15                 [512, 512, 4, 4]          [512, 512, 4, 4]          1,024                     --
│    └─ReLU: 2-16                        [512, 512, 4, 4]          [512, 512, 4, 4]          --                        --
│    └─ResBlock: 2-17                    [512, 512, 4, 4]          [512, 512, 4, 4]          --                        --
│    │    └─Sequential: 3-2              [512, 512, 4, 4]          [512, 512, 4, 4]          4,720,640                 --
├─MaxPool2d: 1-5                         [512, 512, 4, 4]          [512, 512, 1, 1]          --                        4
├─Linear: 1-6                            [512, 512]                [512, 10]                 5,120                     --
============================================================================================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
Total mult-adds (G): 194.18
============================================================================================================================================
Input size (MB): 6.29
Forward/backward pass size (MB): 2382.41
Params size (MB): 26.29
Estimated Total Size (MB): 2414.99
============================================================================================================================================
```
## Image Transformations
```python3
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            always_apply=True,
        ),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(
            min_holes=1,
            max_holes=1,
            min_height=8,
            min_width=8,
            max_height=8,
            max_width=8,
            fill_value=(0.4914, 0.4822, 0.4465),  # type: ignore
            p=0.5,
        ),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            always_apply=True,
        ),
        ToTensorV2(),
    ]
)
```

## Targets To be Achieved
 - :heavy_check_mark: Write a custom ResNet architecture for CIFAR10 that has the following architecture:
PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax
- :heavy_check_mark: Uses One Cycle Policy such that:
Total Epochs = 24
Max at Epoch = 5
LRMIN = FIND
LRMAX = FIND
NO Annihilation
- :heavy_check_mark: Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
Batch size = 512
- :heavy_check_mark: Use ADAM, and CrossEntropyLoss
- :heavy_check_mark: Target Accuracy: 90%
- :heavy_check_mark: NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training. 
- :heavy_check_mark: Once done, proceed to answer the Assignment-Solution page. 

## LR Finder
![image](https://github.com/RaviNaik/ERA-SESSION10/assets/23289802/7df02efb-e727-490c-bd5d-25211d1f8f0a)

## Loss & Accuracy Curves
![image](https://github.com/RaviNaik/ERA-SESSION10/assets/23289802/d99b459b-a14e-4624-acdb-63319b9923f6)
![image](https://github.com/RaviNaik/ERA-SESSION10/assets/23289802/398bd2fd-de8d-4399-a633-c18b4947ec47)

## LR Plot
![image](https://github.com/RaviNaik/ERA-SESSION10/assets/23289802/7fd57b6b-54a0-43dc-95e6-64be165a5339)





