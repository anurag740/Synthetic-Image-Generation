using Flux
using Images
using CUDA

# Load Image
img = load("synthetic_image_1.png")
img = imresize(img, (224, 224))
img_tensor = Float32.(channelview(img))

# Define Simple Flux Model
model = Chain(
    Conv((3, 3), 3=>16, relu),
    MaxPool((2, 2)),
    Dense(2704, 10),  # Adjust the input size accordingly
    softmax
)

# Forward Pass
output = model(img_tensor[:])
println("Output:", output)
