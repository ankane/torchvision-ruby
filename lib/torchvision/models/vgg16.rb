module TorchVision
  module Models
    module VGG16
      def self.new(**kwargs)
        VGG.make_model("vgg16", "D", false, **kwargs)
      end
    end
  end
end
