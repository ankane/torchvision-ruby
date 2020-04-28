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

        # TODO improve
        def to_tensor(pic)
          if !pic.is_a?(Numo::NArray) && !pic.is_a?(Torch::Tensor)
            raise ArgumentError, "pic should be tensor or Numo::NArray. Got #{pic.class.name}"
          end

          if pic.is_a?(Numo::NArray) && ![2, 3].include?(pic.ndim)
            raise ArgumentError, "pic should be 2/3 dimensional. Got #{pic.dim} dimensions."
          end

          if pic.is_a?(Numo::NArray)
            if pic.ndim == 2
              raise Torch::NotImplementedYet
            end

            img = Torch.from_numo(pic.transpose(2, 0, 1))
            return img.float.div(255)
          end

          pic = pic.float
          pic.unsqueeze!(0).div!(255)
        end
      end
    end

    # shortcut
    F = Functional
  end
end
