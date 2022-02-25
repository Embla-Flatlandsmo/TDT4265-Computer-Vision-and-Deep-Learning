import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np

# Task 4c
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)

weight = model.conv1.weight.data.cpu()
print("Filter/Weight/kernel size:", weight.shape)

def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


"""
# Task 4b, uncomment to run
# indices = [0, 1, 2, 3, 14]
indices = [14, 26, 32, 49, 52]

num_filters = len(indices)
# print("Shape of weight[0]: " + weight[0].shape)
weight_numpy = torch_image_to_numpy(weight[0])

fig, ax = plt.subplots(2, num_filters, figsize=(20,4))
n = 0
for i in indices:
    ax[0,n].imshow(torch_image_to_numpy(weight[i]))
    ax[1,n].imshow(torch_image_to_numpy(activation[0,i]), cmap="gray")
    n += 1
fig.tight_layout()

plt.savefig("plots/task_4b_plot.png")
plt.show()
"""


# Task 4c

"""
# For viewing the node names. Uncomment to run
node_names = get_graph_node_names(model)[0]
print(node_names)
"""
feature_extractor = create_feature_extractor(model, return_nodes={'layer4':'layer4'})
activation = feature_extractor(image)
print(activation['layer4'].shape)

# indices = [range(0,10)]
indices = list(range(0,10))
print(indices)

num_filters = len(indices)
# # print("Shape of weight[0]: " + weight[0].shape)
# weight_numpy = torch_image_to_numpy(weight[0])
fig, ax = plt.subplots(2, num_filters//2, figsize=(10,4))

n = 0
for i in range(0,num_filters//2):
    ax[0,i].set_title(str(i+1))
    ax[0,i].axis('off')
    ax[0,i].imshow(torch_image_to_numpy(activation['layer4'][0,indices[i]]))
    ax[1,i].set_title(str(i+1+5))
    ax[1,i].axis('off')
    ax[1,i].imshow(torch_image_to_numpy(activation['layer4'][0,indices[i+5]]))
# for i in indices:
#     ax[0,n].imshow(torch_image_to_numpy(activation['layer4'][0,i]), cmap="gray")
#     n += 1
fig.tight_layout()

plt.savefig("plots/task_4c_plot.png")
plt.show()