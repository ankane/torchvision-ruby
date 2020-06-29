module TorchVision
  module Models
    module ResNet101
      def self.new(pretrained: false, **kwargs)
        model = ResNet.new(Bottleneck, [3, 4, 23, 3], **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
