module TorchVision
  module Models
    module VGG11BN
      def self.new(**kwargs)
        VGG.make_model("vgg11_bn", "A", true, **kwargs)
      end
    end
  end
end
