# TorchVision

:fire: Computer vision datasets, transforms, and models for Ruby

This gem is currently experimental. There may be breaking changes between each release. Please report any issues you experience.

[![Build Status](https://travis-ci.org/ankane/torchvision.svg?branch=master)](https://travis-ci.org/ankane/torchvision)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem 'torchvision'
```

## Getting Started

This library follows the [Python API](https://pytorch.org/docs/master/torchvision/). Many methods and options are missing at the moment. PRs welcome!

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

```ruby
TorchVision::Models::Resnet18.new
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
