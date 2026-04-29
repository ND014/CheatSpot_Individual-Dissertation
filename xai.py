import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO


model = YOLO("yolov8s/weights/best.pt")
pytorch_model = model.model
pytorch_model.eval()


class YOLOTarget:
    def __call__(self, model_output):
        if isinstance(model_output, (list, tuple)):
            pred = model_output[0]
            if hasattr(pred, "shape"):
                return pred.sum()
            return model_output[0].sum()
        return model_output.sum()


def generate_gradcam(image_path, output_path="gradcam_output.jpg"):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True

    target_layer = pytorch_model.model[-3]

    with GradCAM(model=pytorch_model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[YOLOTarget()],
        )[0]

    img_float = img_resized.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    result = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, result)
    print(f"✅ GradCAM saved: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_gradcam("uploads/image.jpg", "gradcam_result.jpg")