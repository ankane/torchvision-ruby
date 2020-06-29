module TorchVision
  module Models
    module ResNet101
      def self.new(**kwargs)
        ResNet.make_model("resnet101", Bottleneck, [3, 4, 23, 3], **kwargs)
      end
    end
  end
end
