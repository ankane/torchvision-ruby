module TorchVision
  module Models
    module ResNet152
      def self.new(pretrained: false, **kwargs)
        model = ResNet.new(Bottleneck, [3, 8, 36, 3], **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/resnet152-b121ed2d.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
