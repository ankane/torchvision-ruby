module TorchVision
  module Models
    module VGG13BN
      def self.new(pretrained: false, **kwargs)
        kwargs[:init_weights] = false if pretrained
        model = VGG.new(VGG.make_layers("B", true), **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
