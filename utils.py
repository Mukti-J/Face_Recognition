import cv2
import numpy as np

def draw_box(img, box, label=None, color=(0,255,0), thickness=2):
    # Adjust thickness based on image size for better visibility
    h, w = img.shape[:2]
    scale = max(w, h) / 640  # 640 is a reference size
    dynamic_thickness = max(int(thickness * scale), 2)
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, dynamic_thickness)
    if label:
        font_scale = 0.6 * scale
        font_thickness = max(int(2 * scale), 1)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness)

def resize_image(image, width=None, height=None):
    h, w = image.shape[:2]
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    elif height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        return image
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
