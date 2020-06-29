module TorchVision
  module Models
    module ResNext50_32x4d
      def self.new(**kwargs)
        kwargs[:groups] = 32
        kwargs[:width_per_group] = 4
        ResNet.make_model("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], **kwargs)
      end
    end
  end
end
