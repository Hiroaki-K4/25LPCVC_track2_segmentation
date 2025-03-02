import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import onnxruntime
from tqdm import tqdm

from compile_profile_inference_aihub import prepare_data


def main():
    # Prepare input data
    image_dir = Path('./compile_and_profile/images/default')
    annotation_path = Path('./compile_and_profile/annotations/instances_default.json')

    with annotation_path.open() as f:
        annotations = json.load(f)
    data = list()
    for annotation in tqdm(annotations['annotations']):
        for image in annotations['images']:
            if image['id'] == annotation['image_id']:
                break
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

    # Inference
    onnx_path = Path("./compile_and_profile/onnx")
    output_path = onnx_path / "output"
    providers = ["CPUExecutionProvider"]

    session = onnxruntime.InferenceSession(
        f"{onnx_path}/model.onnx", providers=providers
    )
    for d in tqdm(data):
        inputs = {
            "image_input": d['image_input'].detach().cpu().numpy(),
            "text_input": d['text_input'].detach().cpu().numpy(),
        }
        outputs = session.run(None, inputs)
        # save output
        np.save(output_path / 'numpy' / f"{d['image_name'].replace(' ', '_')}_{d['annotation_id']}_{d['text'].replace(' ', '_')}.npy", outputs[0])

        # Save visualization of model output
        plt.matshow(outputs[0])
        plt.axis("off")
        plt.savefig(output_path / 'images' / f"{d['image_name'].replace(' ', '_')}_{d['annotation_id']}_{d['text'].replace(' ', '_')}.jpg")


if __name__ == '__main__':
    main()
