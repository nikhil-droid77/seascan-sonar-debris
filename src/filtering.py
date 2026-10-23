"""Stage 2 - noise / false-positive filtering of detections.

A detection is a dict: {"x1","y1","x2","y2","conf","cls","label"} in pixel coordinates.
"""


def area(d):
    return max(d["x2"] - d["x1"], 0) * max(d["y2"] - d["y1"], 0)


def iou(a, b):
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, iou_thr=0.5):
    """Non-maximum suppression: keep the most confident box among overlapping ones."""
    kept = []
    for d in sorted(dets, key=lambda d: d["conf"], reverse=True):
        if all(iou(d, k) < iou_thr for k in kept):
            kept.append(d)
    return kept


def filter_detections(dets, img_w, img_h, conf_thr=0.25, iou_thr=0.5,
                      min_area=25, max_area_frac=0.5, nadir_frac=0.0):
    """Drop weak, tiny, huge or nadir-gap boxes, then apply NMS."""
    out = []
    for d in dets:
        if d["conf"] < conf_thr:
            continue
        a = area(d)
        if a < min_area or a > max_area_frac * img_w * img_h:
            continue
        cx = (d["x1"] + d["x2"]) / 2
        if nadir_frac > 0 and abs(cx - img_w / 2) < nadir_frac * img_w / 2:
            continue
        out.append(d)
    return nms(out, iou_thr)
