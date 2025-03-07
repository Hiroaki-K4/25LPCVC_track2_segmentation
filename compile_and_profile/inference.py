import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_anything import sam_model_registry

from compile_profile_inference_aihub import prepare_data
from models import GroundedSAM, GroundingDino


def main():
    # Prepare input data
    data = load_data()

    # Inference
    inference(data)


def load_data(image_id: Optional[int] = None):
    image_dir = Path('./compile_and_profile/images/default')
    annotation_path = Path('./compile_and_profile/annotations/instances_default.json')
    with annotation_path.open() as f:
        annotations = json.load(f)

    data = list()
    for annotation in tqdm(annotations['annotations']):
        if image_id is not None and annotation['image_id'] != image_id:
            continue
        for image in annotations['images']:
            if image['id'] == annotation['image_id']:
                break
        else:
            raise Exception("image not found")
        image_path = image_dir / image['file_name']
        texts = [v for v in annotation['attributes'].values() if isinstance(v, str) and len(v) > 0]
        for text in texts:
            image_input, text_input = prepare_data(image_path, text)
            data.append({
                'image_name': image_path.name,
                'annotation_id': annotation['id'],
                'text': text,
                'image_input': image_input,
                'text_input': text_input
            })
    return data


def inference(data):
    GROUNDING_DINO_CONFIG_PATH = "./configs/groundingdino/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./lpcvc_track2_models/groundingdino_swint_ogc.pth"
    grounding_dino_model = GroundingDino(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    SAM_ENCODER_VERSION = "vit_b"
    SAM_CHECKPOINT_PATH = "./lpcvc_track2_models/sam_vit_b_01ec64.pth"
    sam_model = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)

    model = GroundedSAM(grounding_dino_model, sam_model)

    # Inference
    # onnx_path = Path("./compile_and_profile/onnx")
    # output_path = onnx_path / "output"
    torch_path = Path("./compile_and_profile/torch")
    output_path = torch_path / "output"
    numpy_path = output_path / 'numpy'
    images_path = output_path / 'images'
    numpy_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    # providers = ["CPUExecutionProvider"]

    # session = onnxruntime.InferenceSession(
    #     f"{onnx_path}/model.onnx", providers=providers
    # )
    # for d in tqdm(data):
    #     inputs = {
    #         "image_input": d['image_input'].detach().cpu().numpy(),
    #         "text_input": d['text_input'].detach().cpu().numpy(),
    #     }
    #     outputs = session.run(None, inputs)
    #     # save output
    #     np.save(numpy_path / f"{d['image_name'].replace(' ', '_')}_{d['annotation_id']}_{d['text'].replace(' ', '_')}.npy", outputs[0])

    #     # Save visualization of model output
    #     plt.matshow(outputs[0])
    #     plt.axis("off")
    #     plt.savefig(images_path / f"{d['image_name'].replace(' ', '_')}_{d['annotation_id']}_{d['text'].replace(' ', '_')}.jpg")
    for d in tqdm(data):
        unique_fp = f"{d['image_name'].replace(' ', '_')}_{d['annotation_id']}_{d['text'].replace(' ', '_')}"
        image_save_dir = Path('./compile_and_profile') / 'annotated' / unique_fp
        output = model(d['image_input'], d['text_input'], image_save_dir)
        # save output
        np.save(numpy_path / f"{unique_fp}.npy", output)

        # Save visualization of model output
        plt.matshow(output)
        plt.axis("off")
        plt.savefig(images_path / f"{unique_fp}.jpg")


if __name__ == '__main__':
    main()
