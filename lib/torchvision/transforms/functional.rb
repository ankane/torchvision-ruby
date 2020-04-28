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
          # if mean.ndim == 1
          #   raise Torch::NotImplementedYet
          # end
          # if std.ndim == 1
          #   raise Torch::NotImplementedYet
          # end
          tensor.sub!(mean).div!(std)
          tensor
        end

        # TODO improve
        def to_tensor(pic)
          pic = pic.float
          pic.unsqueeze!(0).div!(255)
        end
      end
    end

    # shortcut
    F = Functional
  end
end
