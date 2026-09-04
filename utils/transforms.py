"""Image transforms for chest X-rays.

Uses torchvision transforms. Grayscale CXRs are expanded to 3 channels so the
ImageNet-pretrained ResNet backbones can be used directly.
"""
from torchvision import transforms

# ImageNet normalization (works fine for 3-channel-replicated CXR).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True):
    """Return a torchvision transform pipeline."""
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
