module TorchVision
  module Datasets
    class VisionDataset < Torch::Utils::Data::Dataset
      attr_reader :data, :targets

      def initialize(root, transforms: nil, transform: nil, target_transform: nil)
        @root = root

        has_transforms = !transforms.nil?
        has_separate_transform = !transform.nil? || !target_transform.nil?
        if has_transforms && has_separate_transform
          raise ArgumentError, "Only transforms or transform/target_transform can be passed as argument"
        end

        @transform = transform
        @target_transform = target_transform

        if has_separate_transform
          # transforms = StandardTransform.new(transform, target_transform)
        end
        @transforms = transforms
      end

      private

      def download_file(url, download_root:, filename:, sha256:)
        FileUtils.mkdir_p(download_root)

        dest = File.join(download_root, filename)
        return dest if File.exist?(dest)

        uri = URI.parse(url)
        raise "Invalid URL" unless uri.is_a?(URI::HTTP) # includes https

        uri.open(open_timeout: 3, redirect: false) do |download|
          digest =
            if download.respond_to?(:path)
              download.flush
              Digest::SHA256.file(download.path).hexdigest
            else
              Digest::SHA256.hexdigest(download.string)
            end

          if digest != sha256
            raise Error, "Bad hash"
          end

          IO.copy_stream(download, dest.to_str)
        end

        dest
      end

      def check_integrity(path, sha256)
        File.exist?(path) && Digest::SHA256.file(path).hexdigest == sha256
      end
    end
  end
end
