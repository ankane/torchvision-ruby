# dependencies
require "numo/narray"
require "torch"

# stdlib
require "digest"
require "fileutils"
require "net/http"
require "rubygems/package"
require "tmpdir"

# modules
require "torchvision/version"

# datasets
require "torchvision/datasets/vision_dataset"
require "torchvision/datasets/cifar10"
require "torchvision/datasets/cifar100"
require "torchvision/datasets/mnist"
require "torchvision/datasets/fashion_mnist"
require "torchvision/datasets/kmnist"

# transforms
require "torchvision/transforms/compose"
require "torchvision/transforms/functional"
require "torchvision/transforms/normalize"
require "torchvision/transforms/to_tensor"

module TorchVision
  class Error < StandardError; end
end
