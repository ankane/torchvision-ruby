## 0.2.2 (unreleased)

- Fixed error with ruby-vips 2.1.2

## 0.2.1 (2021-03-14)

- Added `ImageFolder` and `DatasetFolder`
- Added `CenterCrop` and `RandomResizedCrop` transforms
- Added `crop` method

## 0.2.0 (2021-03-11)

- Added `RandomHorizontalFlip`, `RandomVerticalFlip`, and `Resize` transforms
- Added `save_image` method
- Added `data` and `targets` methods to datasets
- Removed support for Ruby < 2.6

Breaking changes

- Added dependency on libvips
- MNIST datasets return images instead of tensors

## 0.1.3 (2020-06-29)

- Added AlexNet model
- Added ResNet34, ResNet50, ResNet101, and ResNet152 models
- Added ResNeXt model
- Added VGG11, VGG13, VGG16, and VGG19 models
- Added Wide ResNet model

## 0.1.2 (2020-04-29)

- Added CIFAR10, CIFAR100, FashionMNIST, and KMNIST datasets
- Added ResNet18 model

## 0.1.1 (2020-04-28)

- Removed `mini_magick` for performance

## 0.1.0 (2020-04-27)

- First release
