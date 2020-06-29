module TorchVision
  module Models
    module VGG11
      def self.new(**kwargs)
        VGG.make_model("vgg11", "A", false, **kwargs)
      end
    end
  end
end
