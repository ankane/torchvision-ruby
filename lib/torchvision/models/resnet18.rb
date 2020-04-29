module TorchVision
  module Models
    module ResNet18
      def self.new(**kwargs)
        ResNet.new(BasicBlock, [2, 2, 2, 2], **kwargs)
      end
    end
  end
end
