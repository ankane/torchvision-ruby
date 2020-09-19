# TorchVision

:fire: Computer vision datasets, transforms, and models for Ruby

[![Build Status](https://travis-ci.org/ankane/torchvision.svg?branch=master)](https://travis-ci.org/ankane/torchvision)

## Installation

First, [install libvips](https://libvips.github.io/libvips/install.html). For Homebrew, use:

```sh
brew install vips
```

Add this line to your application’s Gemfile:

```ruby
gem 'torchvision'
```

## Getting Started

This library follows the [Python API](https://pytorch.org/docs/stable/torchvision/index.html). Many methods and options are missing at the moment. PRs welcome!

## Datasets

Load a dataset

```ruby
TorchVision::Datasets::MNIST.new("./data", train: true, download: true)
```

Supported datasets are:

- CIFAR10
- CIFAR100
- FashionMNIST
- KMNIST
- MNIST

## Transforms

```ruby
TorchVision::Transforms::Compose.new([
  TorchVision::Transforms::ToTensor.new,
  TorchVision::Transforms::Normalize.new([0.1307], [0.3081])
])
```

## Models

- [AlexNet](#alexnet)
- [ResNet](#resnet)
- [ResNeXt](#resnext)
- [VGG](#vgg)
- [Wide ResNet](#wide-resnet)

### AlexNet

```ruby
TorchVision::Models::AlexNet.new
```

### ResNet

```ruby
TorchVision::Models::ResNet18.new
TorchVision::Models::ResNet34.new
TorchVision::Models::ResNet50.new
TorchVision::Models::ResNet101.new
TorchVision::Models::ResNet152.new
```

### ResNeXt

```ruby
TorchVision::Models::ResNext52_32x4d.new
TorchVision::Models::ResNext101_32x8d.new
```

### VGG

```ruby
TorchVision::Models::VGG11.new
TorchVision::Models::VGG11BN.new
TorchVision::Models::VGG13.new
TorchVision::Models::VGG13BN.new
TorchVision::Models::VGG16.new
TorchVision::Models::VGG16BN.new
TorchVision::Models::VGG19.new
TorchVision::Models::VGG19BN.new
```

### Wide ResNet

```ruby
TorchVision::Models::WideResNet52_2.new
TorchVision::Models::WideResNet101_2.new
```

## Disclaimer

This library downloads and prepares public datasets. We don’t host any datasets. Be sure to adhere to the license for each dataset.

If you’re a dataset owner and wish to update any details or remove it from this project, let us know.

## History

View the [changelog](https://github.com/ankane/torchvision/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/torchvision/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/torchvision/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/torchvision.git
cd torchvision
bundle install
bundle exec rake test
```
