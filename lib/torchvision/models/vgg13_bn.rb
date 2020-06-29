module TorchVision
  module Models
    module VGG13BN
      def self.new(**kwargs)
        VGG.make_model("vgg13_bn", "B", true, **kwargs)
      end
    end
  end
end
