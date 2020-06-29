module TorchVision
  module Models
    module VGG11
      def self.new(pretrained: false, **kwargs)
        kwargs[:init_weights] = false if pretrained
        model = VGG.new(VGG.make_layers("A", false), **kwargs)
        if pretrained
          url = "https://download.pytorch.org/models/vgg11-bbd30ac9.pth"
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
