module TorchVision
  module Models
    module VGG19BN
      def self.new(**kwargs)
        VGG.make_model("vgg19_bn", "E", true, **kwargs)
      end
    end
  end
end
