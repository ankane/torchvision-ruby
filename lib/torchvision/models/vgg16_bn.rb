module TorchVision
  module Models
    module VGG16BN
      def self.new(pretrained: false, **kwargs)
        kwargs[:init_weights] = false if pretrained
        model = VGG.new(VGG.make_layers("D", true), **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
