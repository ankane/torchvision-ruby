module TorchVision
  module Models
    module VGG16BN
      def self.new(**kwargs)
        VGG.make_model("vgg16_bn", "D", true, **kwargs)
      end
    end
  end
end
