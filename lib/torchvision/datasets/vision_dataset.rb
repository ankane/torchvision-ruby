module TorchVision
  module Datasets
    # TODO inherit Torch::Utils::Data::Dataset
    class VisionDataset
      def initialize(root, transforms: nil, transform: nil, target_transform: nil)
        @root = root

        has_transforms = !transforms.nil?
        has_separate_transform = !transform.nil? || !target_transform.nil?
        if has_transforms && has_separate_transform
          raise ArgumentError, "Only transforms or transform/target_transform can be passed as argument"
        end

        if has_separate_transform
          raise Torch::NotImplementedYet
          # transforms = StandardTransform.new(transform, target_transform)
        end
        @transforms = transforms
      end
    end
  end
end
