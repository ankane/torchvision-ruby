module TorchVision
  module Models
    module WideResNet101_2
      def self.new(pretrained: false, **kwargs)
        kwargs[:width_per_group] = 64 * 2
        model = ResNet.new(Bottleneck, [3, 4, 23, 3], **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
