from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torch
from torchvision import transforms
from tqdm import tqdm
from detectron2.structures import ImageList

image_dir = Path('./compile_and_profile/images/default')
annotation_path = Path('./compile_and_profile/annotations/instances_default.json')
coco = COCO(annotation_path)


def main():
    evaluate()


def evaluate():
    IoUs = []
    for ann_id in tqdm(coco.anns):
        ann = coco.anns[ann_id]

        image_id = ann['image_id']
        file_name = coco.imgs[image_id]['file_name']
        torch_output_dir = Path("./compile_and_profile/torch/output")
        onnx_output_dir = Path("./compile_and_profile/onnx/output")

        texts = [v for v in ann['attributes'].values() if isinstance(v, str) and len(v) > 0]
        for text in texts:
            np_path = torch_output_dir / 'numpy' / f"{file_name.replace(' ', '_')}_{ann_id}_{text.replace(' ', '_')}.npy"
            # np_path = torch_output_dir / 'numpy' / f"{file_name.replace(' ', '_')}_{ann_id}_{text.replace(' ', '_')}.npy"
            output = np.load(np_path)
            output_bool = output > 0

            mask = coco.annToMask(ann)
            mask_image = Image.fromarray(mask)
            mask_resized = resize_and_pad_mask(mask_image)[0]
            mask_resized_bool = mask_resized > 0

            IoUs.append(comput_IoU(output_bool, mask_resized_bool))
    mIoU = sum(IoUs) / len(IoUs)
    print(IoUs)
    print(len(IoUs))
    print(mIoU)
    return


def comput_IoU(pred_mask, gold_mask):
    I = pred_mask & gold_mask
    U = pred_mask | gold_mask
    return I.sum() / (U.sum() + 1e-6)


def resize_and_pad_mask(mask):
    # resizing
    transform = transforms.Compose([transforms.Resize(1000, max_size=1024)])
    image = transform(mask)

    # padding
    image = torch.from_numpy(np.asanyarray(image)).float()
    size_divisibility = 1024  # Resize and pad all images to 1024x1024
    images = [image]
    image_input = ImageList.from_tensors(images, size_divisibility).tensor
    return image_input.numpy()


if __name__ == "__main__":
    main()
