from functools import partial
import numpy as np

import cv2
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import (
    slow_r50_detection,
)

from visualization import VideoVisualizer

device = "cpu"
video_model = slow_r50_detection(True)
video_model = video_model.eval().to(device)

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)
predictor = DefaultPredictor(cfg)


# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())["instances"].to("cpu")
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(
        predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    )
    predicted_boxes = boxes[
        np.logical_and(classes == 0, scores > 0.75)
    ].tensor.cpu()  # only person
    return predicted_boxes


def ava_inference_transform(
    clip,
    boxes,
    num_frames=4,  # if using slowfast_r50_detection, change this to 32
    crop_size=256,
    data_mean=[0.45, 0.45, 0.45],
    data_std=[0.225, 0.225, 0.225],
    slow_fast_alpha=None,  # if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes


# Create an id to label name mapping
label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map(
    "ava_action_list.pbtxt"
)
# Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres", thres=0.5)

# Load the video
encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path("soccer.mp4")
print("Completed loading encoded video.")

time_stamp_range = [
    0,
    0.25,
    0.5,
    0.75,
    1,
    1.25,
    1.5,
    1.75,
    2,
]  # time stamps in video for which clip is sampled.
clip_duration = 0.25  # Duration of clip
gif_imgs = []

for time_stamp in time_stamp_range:
    print("Generating predictions for time stamp: {} sec".format(time_stamp))

    # Generate clip
    inp_imgs = encoded_vid.get_clip(
        time_stamp - clip_duration / 2.0,
        time_stamp + clip_duration / 2.0,
    )
    inp_imgs = inp_imgs["video"]

    # Generate people bbox predictions
    inp_img = inp_imgs[:, inp_imgs.shape[1] // 2, :, :]
    inp_img = inp_img.permute(1, 2, 0)

    predicted_boxes = get_person_bboxes(inp_img, predictor)
    if len(predicted_boxes) == 0:
        print("Skipping clip no frames detected at time stamp: ", time_stamp)
        continue

    inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())
    # Prepend data sample id for each bounding box.
    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)

    # Generate actions predictions for the bounding boxes
    preds = video_model(inputs.unsqueeze(0).to(device), inp_boxes.to(device))

    preds = preds.to("cpu")
    preds = torch.cat([torch.zeros(preds.shape[0], 1), preds], dim=1)

    # Plot predictions on the video and save.
    inp_imgs = inp_imgs.permute(1, 2, 3, 0)
    inp_imgs = inp_imgs / 255.0
    out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
    gif_imgs += out_img_pred

print("Finished generating predictions.")

height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]

vide_save_path = "output.mp4"
video = cv2.VideoWriter(
    vide_save_path, cv2.VideoWriter_fourcc(*"DIVX"), 7, (width, height)
)

for image in gif_imgs:
    img = (255 * image).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    video.write(img)
video.release()

print("Predictions saved to file: ", vide_save_path)
