from src.filtering import filter_detections, nms


def box(x1, y1, x2, y2, conf, cls=0):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls": cls, "label": "x"}


def test_low_confidence_removed():
    assert filter_detections([box(10, 10, 60, 60, 0.1)], 200, 200) == []


def test_tiny_box_removed():
    assert filter_detections([box(10, 10, 12, 12, 0.9)], 200, 200) == []


def test_nms_keeps_best_of_overlapping():
    out = nms([box(10, 10, 60, 60, 0.9), box(12, 12, 62, 62, 0.5)])
    assert len(out) == 1 and out[0]["conf"] == 0.9
