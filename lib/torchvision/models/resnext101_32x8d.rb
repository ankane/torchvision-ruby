module TorchVision
  module Models
    module ResNext101_32x8d
      def self.new(**kwargs)
        kwargs[:groups] = 32
        kwargs[:width_per_group] = 8
        ResNet.make_model("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], **kwargs)
      end
    end
  end
end
