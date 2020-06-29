module TorchVision
  module Models
    module ResNet34
      def self.new(**kwargs)
        ResNet.make_model("resnet34", BasicBlock, [3, 4, 6, 3], **kwargs)
      end
    end
  end
end
