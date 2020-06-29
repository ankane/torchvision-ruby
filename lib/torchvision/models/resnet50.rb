module TorchVision
  module Models
    module ResNet50
      def self.new(**kwargs)
        ResNet.make_model("resnet50", Bottleneck, [3, 4, 6, 3], **kwargs)
      end
    end
  end
end
