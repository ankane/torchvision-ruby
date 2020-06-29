module TorchVision
  module Models
    module VGG13
      def self.new(pretrained: false, **kwargs)
        kwargs[:init_weights] = false if pretrained
        model = VGG.new(VGG.make_layers("B", false), **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/vgg13-c768596a.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
