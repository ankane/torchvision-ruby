module TorchVision
  module Models
    module ResNet152
      def self.new(**kwargs)
        ResNet.make_model("resnet152", Bottleneck, [3, 8, 36, 3], **kwargs)
      end
    end
  end
end
