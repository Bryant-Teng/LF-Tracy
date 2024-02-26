import numpy as np
import torch
import torchvision.transforms as transforms

def mixup_images(main_image, image_list, alpha=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = transforms.Normalize(mean=mean, std=std)

    main_image = normalize(main_image)
    mixed_images = []
    for img in image_list:
        img = normalize(img)
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * main_image + (1 - lam) * img
        mixed_images.append(mixed_img)

    return mixed_images


import torch

def mixup_images2(main_image, image_list, alpha=0.5): #optimized to import more randomness

    if torch.rand(1).item() < 0.1:
        img_to_mix = image_list[torch.randint(0, len(image_list), (1,)).item()]
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        main_image = lam * main_image + (1 - lam) * img_to_mix

    new_image_list = []
    for img in image_list:
        if torch.rand(1).item() < 0.5:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
            mixed_img = lam * main_image + (1 - lam) * img
            new_image_list.append(mixed_img)
        else:
            new_image_list.append(img)

    return main_image, new_image_list
