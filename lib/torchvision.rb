# dependencies
require "numo/narray"
require "vips"
require "torch"

# stdlib
require "digest"
require "fileutils"
require "net/http"
require "rubygems/package"
require "tmpdir"

# modules
require_relative "torchvision/utils"
require_relative "torchvision/version"

# datasets
require_relative "torchvision/datasets/vision_dataset"
require_relative "torchvision/datasets/dataset_folder"
require_relative "torchvision/datasets/image_folder"
require_relative "torchvision/datasets/cifar10"
require_relative "torchvision/datasets/cifar100"
require_relative "torchvision/datasets/mnist"
require_relative "torchvision/datasets/fashion_mnist"
require_relative "torchvision/datasets/kmnist"

# models
require_relative "torchvision/models/alexnet"
require_relative "torchvision/models/basic_block"
require_relative "torchvision/models/bottleneck"
require_relative "torchvision/models/resnet"
require_relative "torchvision/models/resnet18"
require_relative "torchvision/models/resnet34"
require_relative "torchvision/models/resnet50"
require_relative "torchvision/models/resnet101"
require_relative "torchvision/models/resnet152"
require_relative "torchvision/models/resnext50_32x4d"
require_relative "torchvision/models/resnext101_32x8d"
require_relative "torchvision/models/vgg"
require_relative "torchvision/models/vgg11"
require_relative "torchvision/models/vgg11_bn"
require_relative "torchvision/models/vgg13"
require_relative "torchvision/models/vgg13_bn"
require_relative "torchvision/models/vgg16"
require_relative "torchvision/models/vgg16_bn"
require_relative "torchvision/models/vgg19"
require_relative "torchvision/models/vgg19_bn"
require_relative "torchvision/models/wide_resnet50_2"
require_relative "torchvision/models/wide_resnet101_2"

# transforms
require_relative "torchvision/transforms/center_crop"
require_relative "torchvision/transforms/compose"
require_relative "torchvision/transforms/functional"
require_relative "torchvision/transforms/normalize"
require_relative "torchvision/transforms/random_horizontal_flip"
require_relative "torchvision/transforms/random_resized_crop"
require_relative "torchvision/transforms/random_vertical_flip"
require_relative "torchvision/transforms/resize"
require_relative "torchvision/transforms/to_tensor"

module TorchVision
  class Error < StandardError; end
end
