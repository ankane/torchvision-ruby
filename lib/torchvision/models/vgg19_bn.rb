module TorchVision
  module Models
    module VGG19BN
      def self.new(pretrained: false, **kwargs)
        kwargs[:init_weights] = false if pretrained
        model = VGG.new(VGG.make_layers("E", true), **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
