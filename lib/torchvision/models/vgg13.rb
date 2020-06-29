module TorchVision
  module Models
    module VGG13
      def self.new(**kwargs)
        VGG.make_model("vgg13", "B", false, **kwargs)
      end
    end
  end
end
