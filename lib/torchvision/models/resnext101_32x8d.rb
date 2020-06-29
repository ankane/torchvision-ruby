module TorchVision
  module Models
    module ResNext101_32x8d
      def self.new(pretrained: false, **kwargs)
        kwargs[:groups] = 32
        kwargs[:width_per_group] = 8
        model = ResNet.new(Bottleneck, [3, 4, 23, 3], **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
