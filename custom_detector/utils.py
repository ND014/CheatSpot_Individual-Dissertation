import torch
import numpy as np

def iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    Both boxes: [x_center, y_center, width, height] normalized 0-1
    """
    # Convert to corner format
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2

    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * \
                 (inter_y2 - inter_y1).clamp(0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-6

    return inter_area / union_area


def nms(boxes, scores, iou_threshold=0.4):
    """
    Non-Maximum Suppression — removes duplicate boxes
    boxes:  (N, 4) tensor [x1, y1, x2, y2]
    scores: (N,)   tensor confidence scores
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort by confidence score descending
    sorted_idx = torch.argsort(scores, descending=True)
    keep = []

    while sorted_idx.numel() > 0:
        best = sorted_idx[0]
        keep.append(best.item())

        if sorted_idx.numel() == 1:
            break

        rest    = sorted_idx[1:]
        iou_val = iou(
            boxes[best].unsqueeze(0).expand(rest.shape[0], -1),
            boxes[rest]
        )
        # Keep only boxes with low overlap
        sorted_idx = rest[iou_val < iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def decode_predictions(predictions, conf_threshold=0.5, iou_threshold=0.4):
    """
    Convert raw model output → final bounding boxes
    predictions: (batch, 6, S, S) where 6 = [x, y, w, h, obj, class]
    Returns list of detections per image
    """
    batch_size = predictions.shape[0]
    S          = predictions.shape[2]   # grid size (13)
    results    = []

    for b in range(batch_size):
        pred   = predictions[b]         # (6, S, S)
        boxes  = []
        scores = []

        for i in range(S):
            for j in range(S):
                obj_score  = torch.sigmoid(pred[4, i, j]).item()
                cls_score  = torch.sigmoid(pred[5, i, j]).item()
                confidence = obj_score * cls_score

                if confidence < conf_threshold:
                    continue

                # Decode box coordinates
                tx = torch.sigmoid(pred[0, i, j]).item()
                ty = torch.sigmoid(pred[1, i, j]).item()
                tw = pred[2, i, j].item()
                th = pred[3, i, j].item()

                # Convert to absolute coordinates
                cx = (j + tx) / S
                cy = (i + ty) / S
                w  = np.exp(tw) / S
                h  = np.exp(th) / S

                # Convert to corner format [x1, y1, x2, y2]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)

        if len(boxes) == 0:
            results.append([])
            continue

        boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        keep     = nms(boxes_t, scores_t, iou_threshold)

        final = []
        for k in keep:
            final.append({
                'box':   boxes_t[k].tolist(),
                'score': scores_t[k].item()
            })
        results.append(final)

    return results


def draw_boxes(image, detections, color=(0, 255, 0)):
    """
    Draw bounding boxes on image (numpy array HxWxC)
    """
    import cv2
    H, W = image.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det['box']
        score           = det['score']

        x1 = int(x1 * W); y1 = int(y1 * H)
        x2 = int(x2 * W); y2 = int(y2 * H)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"Cheating {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )
    return image
