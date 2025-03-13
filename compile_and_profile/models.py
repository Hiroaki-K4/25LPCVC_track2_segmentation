from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import supervision as sv
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import SamPredictor
from torchvision.ops import box_convert
from transformers import CLIPTokenizer

import torch


def load_dino_model(
    model_config_path: str, model_checkpoint_path: str, device: str = "cpu"
):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


class GroundingDino(nn.Module):
    def __init__(
        self, model_config_path: str, model_checkpoint_path: str, device: str = "cpu"
    ):
        super().__init__()
        self.model = load_dino_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: torch.Tensor,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> sv.Detections:
        boxes, logits, phrases = self.predict(
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        _, source_h, source_w = image.shape
        detections = self.post_process_result(
            source_h=source_h, source_w=source_w, boxes=boxes, logits=logits
        )
        class_id = np.array([i for i, _ in enumerate(phrases)])
        detections.class_id = class_id
        return detections

    def predict(
        self,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        caption = self.preprocess_caption(caption=caption)

        # model = self.to(device)
        image = image.to(device)

        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])

        prediction_logits = (
            outputs["pred_logits"].cpu().sigmoid()[0]
        )  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[
            0
        ]  # prediction_boxes.shape = (nq, 4)

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        phrases = [
            get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            ).replace(".", "")
            for logit in logits
        ]

        return boxes, logits.max(dim=1)[0], phrases

    @staticmethod
    def preprocess_caption(caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    @staticmethod
    def post_process_result(
        source_h: int, source_w: int, boxes: torch.Tensor, logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @classmethod
    def phrases2classes(cls, phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            try:
                # class_ids.append(classes.index(phrase))
                class_ids.append(cls.find_index(phrase, classes))
            except ValueError:
                class_ids.append(None)
        return np.array(class_ids)

    @staticmethod
    def find_index(string, lst):
        # if meet string like "lake river" will only keep "lake"
        # this is an hack implementation for visualization which will be updated in the future
        string = string.lower().split()[0]
        for i, s in enumerate(lst):
            if string in s.lower():
                return i
        print(
            "There's a wrong phrase happen, this is because of our post-process merged wrong tokens, which will be modified in the future. We will assign it with a random label at this time."
        )
        return 0


class GroundedSAM(nn.Module):
    def __init__(self, object_detection_model: GroundingDino, sam_model, device="cpu"):
        super().__init__()
        self.object_detection_model = object_detection_model
        self.sam_model = sam_model
        # self.sam_predictor = SamPredictor(self.sam_model)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
        self.image_resolution = 1024

        self.pixel_mean = (
            # torch.tensor(self.opt["INPUT"]["PIXEL_MEAN"])
            torch.tensor([123.675, 116.280, 103.530])
            .view(1, -1, 1, 1)
            .repeat(1, 1, self.image_resolution, self.image_resolution)
            .to(device)
        )
        self.pixel_std = (
            # torch.tensor(self.opt["INPUT"]["PIXEL_STD"])
            torch.tensor([58.395, 57.120, 57.375])
            .view(1, -1, 1, 1)
            .repeat(1, 1, self.image_resolution, self.image_resolution)
            .to(device)
        )

    def pre_processing(self, image_input, text_input):
        """
        Preprocess inputs before model forward pass.

        Args:
            image_input: Raw input image tensor
            text_input: Raw input text tensor

        Returns:
            Tuple of processed (images, tokens)
        """
        # downsample image for faster inference
        down_sample_size = 256
        images = (image_input - self.pixel_mean) / self.pixel_std
        images = F.interpolate(
            images,
            size=down_sample_size,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        image_input = F.interpolate(
            image_input,
            size=down_sample_size,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )

        tokens = {"input_ids": text_input[0], "attention_mask": text_input[1]}
        return images, tokens, image_input

    def forward(
        self,
        image_input,
        text_input,
        image_save_dir: Path = Path("./compile_and_profile/annotated"),
    ):
        image_save_dir.mkdir(parents=True, exist_ok=True)
        images, tokens, image_input = self.pre_processing(image_input, text_input)
        image_cv2 = self.tensor_to_cv2(image_input[0])

        pretrained_tokenizer = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)

        # Predict classes and hyper-param for GroundingDINO
        caption = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8
        # detect objects
        detections = self.object_detection_model.predict_with_caption(
            image=images[0],
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        # annotate image with detections
        bbox_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"{caption} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]
        annotated_frame = bbox_annotator.annotate(
            scene=image_cv2.copy(), detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        cv2.imwrite(
            image_save_dir / "groundingdino_annotated_image.jpg", annotated_frame
        )

        print(f"Before NMS: {len(detections.xyxy)} boxes")
        # NMS post process
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(
            sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
        ) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box, multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )
        detections.mask = detections.mask.astype(int)
        detections.mask = detections.mask != 0

        bbox_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{caption} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]
        annotated_image = mask_annotator.annotate(
            scene=image_cv2.copy(), detections=detections
        )
        annotated_image = bbox_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
        # save the annotated grounded-sam image
        cv2.imwrite(image_save_dir / "groundedsam_annotated_image.jpg", annotated_image)

        mask_pred = self.post_processing(detections)
        return mask_pred

    def post_processing(self, detections):
        if detections.mask.size == 0:  # predicted 0 boxes
            return torch.zeros(1024, 1024, dtype=torch.float32)
        mask_pred = detections.mask[0]  # np.ndarray (n, 256, 256)
        mask_pred = (
            torch.from_numpy(mask_pred)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(dtype=torch.float32)
        )
        # upsampling to input image size, 1024
        mask_pred_results = F.interpolate(
            mask_pred,
            size=self.image_resolution,  # 1024
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0]
        return (
            mask_pred_results.squeeze()
        )  # bool variables are not supported by QNN as output features

    @staticmethod
    def tensor_to_cv2(image_input: torch.Tensor):
        return cv2.cvtColor(
            image_input.permute(1, 2, 0).numpy().astype(dtype=np.uint8),
            cv2.COLOR_RGB2BGR,
        )

    def convert_to_onnx(self):
        ...
