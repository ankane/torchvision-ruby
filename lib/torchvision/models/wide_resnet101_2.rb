module TorchVision
  module Models
    module WideResNet101_2
      def self.new(**kwargs)
        kwargs[:width_per_group] = 64 * 2
        ResNet.make_model("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], **kwargs)
      end
    end
  end
end
