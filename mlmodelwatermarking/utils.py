import torchvision
from torchvision import transforms


def load_trigger(path, shape):
    """Load trigger data.

    Returns:
        path (str): trigger images folder
        shape (tuple): tuple for dimension of the data

    """
    x, y, z = shape
    modification = [transforms.Resize((y, z)), transforms.CenterCrop(y)]
    # Convert 3 channels to 1
    if x == 1:
        modification.append(transforms.Grayscale(num_output_channels=1))
        modification.append(transforms.ToTensor())
    else:
        modification.append(transforms.ToTensor())
    transformation = torchvision.transforms.Compose(modification)
    specialset = torchvision.datasets.ImageFolder(path,
                                                  transform=transformation)
    return specialset
