using Flux, FileIO, Images

# Define paths
output_dir = "outputs/"
fixed_size = (224, 224)  # Resize all images to 224x224

# Function to load and preprocess images
function load_image(img_path)
    println("Loading: $img_path")
    img = load(img_path) |> imresize(fixed_size) |> channelview |> Float32
    img = reshape(img, size(img, 1), size(img, 2), 3, 1)  # Ensure shape (H, W, C, Batch)
    return img
end

# Define CNN model
conv_layers = Chain(
    Conv((3, 3), 3 => 16, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, relu),
    MaxPool((2, 2)),
    Flux.flatten
)

# Compute the correct Dense layer input size
dummy_input = rand(Float32, fixed_size[1], fixed_size[2], 3, 1)  # Create a dummy input
flattened_size = length(conv_layers(dummy_input))  # Get the number of features after flattening

# Fully connected layers
fc_layers = Chain(
    Dense(flattened_size, 10),  # Use correct size
    softmax
)

# Full model
model = Chain(conv_layers, fc_layers)

# Process all images in outputs/
for i in 1:3
    img_path = output_dir * "preprocessed_image_$i.png"
    if isfile(img_path)
        img = load_image(img_path)
        output = model(img)
        println("Output for $img_path: $output")
    else
        println("File not found: $img_path")
    end
end
