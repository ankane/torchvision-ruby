module TorchVision
  module Datasets
    class ImageFolder < DatasetFolder
      IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]

      def initialize(root, transform: nil, target_transform: nil, is_valid_file: nil)
        super(root, extensions: IMG_EXTENSIONS, transform: transform, target_transform: target_transform, is_valid_file: is_valid_file)
        @imgs = @samples
      end
    end
  end
end
