require_relative "lib/torchvision/version"

Gem::Specification.new do |spec|
  spec.name          = "torchvision"
  spec.version       = TorchVision::VERSION
  spec.summary       = "Computer vision datasets, transforms, and models for Ruby"
  spec.homepage      = "https://github.com/ankane/torchvision-ruby"
  spec.license       = "BSD-3-Clause"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 3.1"

  spec.add_dependency "numo-narray"
  spec.add_dependency "ruby-vips", ">= 2.1.2"
  spec.add_dependency "torch-rb", ">= 0.11.1"
end
