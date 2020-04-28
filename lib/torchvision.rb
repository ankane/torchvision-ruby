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
require "torchvision/datasets/mnist"

# transforms
require "torchvision/transforms/compose"
require "torchvision/transforms/functional"
require "torchvision/transforms/normalize"
require "torchvision/transforms/to_tensor"

module TorchVision
  class Error < StandardError; end
end
