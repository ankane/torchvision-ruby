require_relative "lib/torchvision/version"

Gem::Specification.new do |spec|
  spec.name          = "torchvision"
  spec.version       = TorchVision::VERSION
  spec.summary       = "Computer vision datasets, transforms, and models for Ruby"
  spec.homepage      = "https://github.com/ankane/torchvision"
  spec.license       = "BSD-3-Clause"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@chartkick.com"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.4"

  spec.add_dependency "numo-narray"
  spec.add_dependency "ruby-vips"
  spec.add_dependency "torch-rb", ">= 0.2.7"

  spec.add_development_dependency "bundler"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "minitest", ">= 5"
end
