module TorchVision
  module Transforms
    class Functional
      class << self
        def normalize(tensor, mean, std, inplace: false)
          unless Torch.tensor?(tensor)
            raise ArgumentError, "tensor should be a torch tensor. Got #{tensor.class.name}"
          end

          if tensor.ndimension != 3
            raise ArgumentError, "Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = #{tensor.size}"
          end

          tensor = tensor.clone unless inplace

          dtype = tensor.dtype
          # TODO Torch.as_tensor
          mean = Torch.tensor(mean, dtype: dtype, device: tensor.device)
          std = Torch.tensor(std, dtype: dtype, device: tensor.device)

          # TODO
          if std.to_a.any? { |v| v == 0 }
            raise ArgumentError, "std evaluated to zero after conversion to #{dtype}, leading to division by zero."
          end
          if mean.ndim == 1
            mean = mean[0...mean.size(0), nil, nil]
          end
          if std.ndim == 1
            std = std[0...std.size(0), nil, nil]
          end
          tensor.sub!(mean).div!(std)
          tensor
        end

        def resize(img, size)
          raise "img should be Vips::Image. Got #{img.class.name}" unless img.is_a?(Vips::Image)
          # TODO support array size
          raise "Got inappropriate size arg: #{size}" unless size.is_a?(Integer)

          w, h = img.size
          if (w <= h && w == size) || (h <= w && h == size)
            return img
          end
          if w < h
            ow = size
            oh = (size * h / w).to_i
            img.thumbnail_image(ow, height: oh)
          else
            oh = size
            ow = (size * w / h).to_i
            img.thumbnail_image(ow, height: oh)
          end
        end

        # TODO improve
        def to_tensor(pic)
          if !pic.is_a?(Numo::NArray) && !pic.is_a?(Vips::Image)
            raise ArgumentError, "pic should be Vips::Image or Numo::NArray. Got #{pic.class.name}"
          end

          if pic.is_a?(Numo::NArray) && ![2, 3].include?(pic.ndim)
            raise ArgumentError, "pic should be 2/3 dimensional. Got #{pic.dim} dimensions."
          end

          if pic.is_a?(Numo::NArray)
            if pic.ndim == 2
              pic = pic.reshape(*pic.shape, 1)
            end

            img = Torch.from_numo(pic.transpose(2, 0, 1))
            if img.dtype == :uint8
              return img.float.div(255)
            else
              return img
            end
          end

          case pic.format
          when :uchar
            img = Torch::ByteTensor.new(Torch::ByteStorage.from_buffer(pic.write_to_memory))
          else
            raise Error, "Format not supported yet: #{pic.format}"
          end

          img = img.view(pic.height, pic.width, pic.bands)
          # put it from HWC to CHW format
          img = img.permute([2, 0, 1]).contiguous
          img.float.div(255)
        end

        def hflip(img)
          if img.is_a?(Torch::Tensor)
            img.flip(-1)
          else
            img.flip(:horizontal)
          end
        end

        def vflip(img)
          if img.is_a?(Torch::Tensor)
            img.flip(-2)
          else
            img.flip(:vertical)
          end
        end

        def crop(img, top, left, height, width)
          if img.is_a?(Torch::Tensor)
            indexes = [true] * (img.dim - 2)
            img[*indexes, top...(top + height), left...(left + width)]
          else
            img.crop(left, top, width, height)
          end
        end

        def center_crop(img, output_size)
          if output_size.is_a?(Integer)
            output_size = [output_size.to_i, output_size.to_i]
          elsif output_size.is_a?(Array) && output_size.length == 1
            output_size = [output_size[0], output_size[0]]
          end

          image_width, image_height = image_size(img)
          crop_height, crop_width = output_size

          if crop_width > image_width || crop_height > image_height
            padding_ltrb = [
              crop_width > image_width ? (crop_width - image_width).div(2) : 0,
              crop_height > image_height ? (crop_height - image_height).div(2) : 0,
              crop_width > image_width ? (crop_width - image_width + 1).div(2) : 0,
              crop_height > image_height ? (crop_height - image_height + 1).div(2) : 0
            ]
            # TODO
            img = pad(img, padding_ltrb, fill: 0)
            image_width, image_height = image_size(img)
            if crop_width == image_width && crop_height == image_height
              return img
            end
          end

          crop_top = ((image_height - crop_height) / 2.0).round
          crop_left = ((image_width - crop_width) / 2.0).round
          crop(img, crop_top, crop_left, crop_height, crop_width)
        end

        private

        def image_size(img)
          if img.is_a?(Torch::Tensor)
            assert_image_tensor(img)
            [img.shape[-1], img.shape[-2]]
          else
            [img.width, img.height]
          end
        end

        def assert_image_tensor(img)
          if img.ndim < 2
            raise TypeError, "Tensor is not a torch image."
          end
        end
      end
    end

    # shortcut
    F = Functional
  end
end
