module TorchVision
  module Utils
    class << self
      def make_grid(tensor, nrow: 8, padding: 2, normalize: false, range: nil, scale_each: false, pad_value: 0)
        unless Torch.tensor?(tensor) || (tensor.is_a?(Array) && tensor.all? { |t| Torch.tensor?(t) })
          raise ArgumentError, "tensor or list of tensors expected, got #{tensor.class.name}"
        end

        # if list of tensors, convert to a 4D mini-batch Tensor
        if tensor.is_a?(Array)
          tensor = Torch.stack(tensor, dim: 0)
        end

        if tensor.dim == 2 # single image H x W
          tensor = tensor.unsqueeze(0)
        end
        if tensor.dim == 3 # single image
          if tensor.size(0) == 1 # if single-channel, convert to 3-channel
            tensor = Torch.cat([tensor, tensor, tensor], 0)
          end
          tensor = tensor.unsqueeze(0)
        end

        if tensor.dim == 4 && tensor.size(1) == 1 # single-channel images
          tensor = Torch.cat([tensor, tensor, tensor], 1)
        end

        if normalize
          tensor = tensor.clone # avoid modifying tensor in-place
          if !range.nil? && !range.is_a?(Array)
            raise "range has to be an array (min, max) if specified. min and max are numbers"
          end

          norm_ip = lambda do |img, min, max|
            img.clamp!(min, max)
            img.add!(-min).div!(max - min + 1e-5)
          end

          norm_range = lambda do |t, range|
            if !range.nil?
              norm_ip.call(t, range[0], range[1])
            else
              norm_ip.call(t, t.min.to_f, t.max.to_f)
            end
          end

          if scale_each
            tensor.each do |t| # loop over mini-batch dimension
              norm_range.call(t, range)
            end
          else
            norm_range.call(tensor, range)
          end
        end

        if tensor.size(0) == 1
          return tensor.squeeze(0)
        end

        # make the mini-batch of images into a grid
        nmaps = tensor.size(0)
        xmaps = [nrow, nmaps].min
        ymaps = (nmaps.to_f / xmaps).ceil
        height, width = (tensor.size(2) + padding), (tensor.size(3) + padding)
        num_channels = tensor.size(1)
        grid = tensor.new_full([num_channels, height * ymaps + padding, width * xmaps + padding], pad_value)
        k = 0
        ymaps.times do |y|
          xmaps.times do |x|
            break if k >= nmaps
            grid.narrow(1, y * height + padding, height - padding).narrow(2, x * width + padding, width - padding).copy!(tensor[k])
            k += 1
          end
        end
        grid
      end

      def save_image(tensor, fp, nrow: 8, padding: 2, normalize: false, range: nil, scale_each: false, pad_value: 0)
        grid = make_grid(tensor, nrow: nrow, padding: padding, pad_value: pad_value, normalize: normalize, range: range, scale_each: scale_each)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add!(0.5).clamp!(0, 255).permute(1, 2, 0).to("cpu", dtype: :uint8).numo
        im = image_from_array(ndarr)
        im.write_to_file(fp)
      end

      # Ruby-specific method
      def image_from_array(array)
        raise "Expected Numo::UInt8, not #{array.class.name}" unless array.is_a?(Numo::UInt8)

        # TODO use Numo directly
        array = Torch.from_numo(array)

        width, height = array.shape
        bands = array.shape[2] || 1
        format = 0 # uchar
        data = FFI::Pointer.new(:uint8, array._data_ptr)
        bytesize = array.numel * array.element_size

        image = Vips.vips_image_new_from_memory(data, bytesize, width, height, bands, format)
        Vips::Image.new(image)
      end
    end
  end
end
