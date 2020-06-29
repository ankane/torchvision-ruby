module TorchVision
  module Models
    module WideResNet50_2
      def self.new(**kwargs)
        kwargs[:width_per_group] = 64 * 2
        ResNet.make_model("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], **kwargs)
      end
    end
  end
end
