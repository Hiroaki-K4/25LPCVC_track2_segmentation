import cv2
from imread_from_url import imread_from_url

from sam2 import SAM2Image, draw_masks


def main():
    encoder_model_path = "lpcvc_track2_models/sam2.1_hiera_base_plus_encoder.onnx"
    decoder_model_path = "lpcvc_track2_models/decoder.onnx"

    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Racing_Terriers_%282490056817%29.jpg/1280px-Racing_Terriers_%282490056817%29.jpg"
    img = imread_from_url(img_url)

    # Initialize model
    sam2 = SAM2Image(encoder_model_path, decoder_model_path)

    # Encode image
    sam2.set_image(img)

    point_coords = (345, 300)
    box_coords = ((0, 285), (600, 700))
    label_id = 0
    is_positive = False

    # Decode image
    sam2.add_point(point_coords, is_positive, label_id)
    sam2.set_box(box_coords, label_id)
    masks = sam2.get_masks()

    masked_img = draw_masks(img, masks)
    cv2.circle(masked_img, point_coords, 5, (0, 0, 255), -1)
    cv2.rectangle(masked_img, box_coords[0], box_coords[1], (0, 255, 0), 2)

    cv2.imshow("masked_img", masked_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
