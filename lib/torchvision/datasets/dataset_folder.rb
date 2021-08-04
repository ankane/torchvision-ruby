module TorchVision
  module Datasets
    class DatasetFolder < VisionDataset
      attr_reader :classes

      def initialize(root, extensions: nil, transform: nil, target_transform: nil, is_valid_file: nil)
        super(root, transform: transform, target_transform: target_transform)
        classes, class_to_idx = find_classes(@root)
        samples = make_dataset(@root, class_to_idx, extensions, is_valid_file)
        if samples.empty?
          msg = "Found 0 files in subfolders of: #{@root}\n"
          unless extensions.nil?
            msg += "Supported extensions are: #{extensions.join(",")}"
          end
          raise RuntimeError, msg
        end

        @loader = lambda do |path|
          Vips::Image.new_from_file(path)
        end
        @extensions = extensions

        @classes = classes
        @class_to_idx = class_to_idx
        @samples = samples
        @targets = samples.map { |s| s[1] }
      end

      def [](index)
        path, target = @samples[index]
        sample = @loader.call(path)
        if @transform
          sample = @transform.call(sample)
        end
        if @target_transform
          target = @target_transform.call(target)
        end

        [sample, target]
      end

      def size
        @samples.size
      end
      alias_method :count, :size
      alias_method :length, :size

      private

      def find_classes(dir)
        classes = Dir.children(dir).select { |d| File.directory?(File.join(dir, d)) }
        classes.sort!
        class_to_idx = classes.map.with_index.to_h
        [classes, class_to_idx]
      end

      def has_file_allowed_extension(filename, extensions)
        filename = filename.downcase
        extensions.any? { |ext| filename.end_with?(ext) }
      end

      def make_dataset(directory, class_to_idx, extensions, is_valid_file)
        instances = []
        directory = File.expand_path(directory)
        both_none = extensions.nil? && is_valid_file.nil?
        both_something = !extensions.nil? && !is_valid_file.nil?
        if both_none || both_something
          raise ArgumentError, "Both extensions and is_valid_file cannot be None or not None at the same time"
        end
        if !extensions.nil?
          is_valid_file = lambda do |x|
            has_file_allowed_extension(x, extensions)
          end
        end
        class_to_idx.keys.sort.each do |target_class|
          class_index = class_to_idx[target_class]
          target_dir = File.join(directory, target_class)
          if !File.directory?(target_dir)
            next
          end
          Dir.glob("**", base: target_dir).sort.each do |fname|
            path = File.join(target_dir, fname)
            if is_valid_file.call(path)
              item = [path, class_index]
              instances << item
            end
          end
        end
        instances
      end
    end
  end
end
