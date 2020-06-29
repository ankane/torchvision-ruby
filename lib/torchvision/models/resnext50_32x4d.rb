module TorchVision
  module Models
    module ResNext50_32x4d
      def self.new(pretrained: false, **kwargs)
        kwargs[:groups] = 32
        kwargs[:width_per_group] = 4
        model = ResNet.new(Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
