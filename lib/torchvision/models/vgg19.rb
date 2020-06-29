module TorchVision
  module Models
    module VGG19
      def self.new(**kwargs)
        VGG.make_model("vgg19", "E", false, **kwargs)
      end
    end
  end
end
