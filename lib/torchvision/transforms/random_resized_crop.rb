module TorchVision
  module Transforms
    class RandomResizedCrop < Torch::NN::Module
      def initialize(size, scale: [0.08, 1.0], ratio: [3.0 / 4.0, 4.0 / 3.0])
        super()
        @size = size
        # @interpolation = interpolation
        @scale = scale
        @ratio = ratio
      end

      def params(img, scale, ratio)
        width, height = F.send(:image_size, img)
        area = height * width

        log_ratio = Torch.log(Torch.tensor(ratio))
        10.times do
          target_area = area * Torch.empty(1).uniform!(scale[0], scale[1]).item
          aspect_ratio = Torch.exp(
            Torch.empty(1).uniform!(log_ratio[0], log_ratio[1])
          ).item

          w = Math.sqrt(target_area * aspect_ratio).round
          h = Math.sqrt(target_area / aspect_ratio).round

          if 0 < w && w <= width && 0 < h && h <= height
            i = Torch.randint(0, height - h + 1, size: [1]).item
            j = Torch.randint(0, width - w + 1, size: [1]).item
            return i, j, h, w
          end
        end

        # Fallback to central crop
        in_ratio = width.to_f / height.to_f
        if in_ratio < ratio.min
          w = width
          h = (w / ratio.min).round
        elsif in_ratio > ratio.max
          h = height
          w = (h * ratio.max).round
        else # whole image
          w = width
          h = height
        end
        i = (height - h).div(2)
        j = (width - w).div(2)
        [i, j, h, w]
      end

      def forward(img)
        i, j, h, w = params(img, @scale, @ratio)
        F.resized_crop(img, i, j, h, w, @size) #, @interpolation)
      end
    end
  end
end
