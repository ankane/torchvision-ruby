module TorchVision
  module Models
    module WideResNet50_2
      def self.new(pretrained: false, **kwargs)
        kwargs[:width_per_group] = 64 * 2
        model = ResNet.new(Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
