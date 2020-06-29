module TorchVision
  module Models
    module VGG19
      def self.new(pretrained: false, **kwargs)
        kwargs[:init_weights] = false if pretrained
        model = VGG.new(VGG.make_layers("E", false), **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
