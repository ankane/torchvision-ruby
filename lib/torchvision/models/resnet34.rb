module TorchVision
  module Models
    module ResNet34
      def self.new(pretrained: false, **kwargs)
        model = ResNet.new(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
