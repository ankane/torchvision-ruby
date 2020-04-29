module TorchVision
  module Models
    module ResNet18
      def self.new(pretrained: false, **kwargs)
        model = ResNet.new(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
