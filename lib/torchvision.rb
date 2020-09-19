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
require "torchvision/version"

# datasets
require "torchvision/datasets/vision_dataset"
require "torchvision/datasets/cifar10"
require "torchvision/datasets/cifar100"
require "torchvision/datasets/mnist"
require "torchvision/datasets/fashion_mnist"
require "torchvision/datasets/kmnist"

# models
require "torchvision/models/alexnet"
require "torchvision/models/basic_block"
require "torchvision/models/bottleneck"
require "torchvision/models/resnet"
require "torchvision/models/resnet18"
require "torchvision/models/resnet34"
require "torchvision/models/resnet50"
require "torchvision/models/resnet101"
require "torchvision/models/resnet152"
require "torchvision/models/resnext50_32x4d"
require "torchvision/models/resnext101_32x8d"
require "torchvision/models/vgg"
require "torchvision/models/vgg11"
require "torchvision/models/vgg11_bn"
require "torchvision/models/vgg13"
require "torchvision/models/vgg13_bn"
require "torchvision/models/vgg16"
require "torchvision/models/vgg16_bn"
require "torchvision/models/vgg19"
require "torchvision/models/vgg19_bn"
require "torchvision/models/wide_resnet50_2"
require "torchvision/models/wide_resnet101_2"

# transforms
require "torchvision/transforms/compose"
require "torchvision/transforms/functional"
require "torchvision/transforms/normalize"
require "torchvision/transforms/resize"
require "torchvision/transforms/to_tensor"

# vips hack
if Vips.respond_to?(:vips_image_new_from_memory)
  warn "[torchvision] Vips hack no longer needed"
else
  Vips.attach_function :vips_image_new_from_memory, [:pointer, :size_t, :int, :int, :int, :int], :pointer
end

# iruby enhancement
if defined?(IRuby::Display::Registry)
  module IRuby::Display::Registry
    match do |obj|
      defined?(Vips::Image) && Vips::Image === obj
    end
    format do |obj|
      loader = obj.get("vips-loader") rescue nil
      if loader == "jpegload"
        ["image/jpeg", obj.write_to_buffer(".jpg")]
      else
        ["image/png", obj.write_to_buffer(".png")]
      end
    end
  end
end

module TorchVision
  class Error < StandardError; end
end
