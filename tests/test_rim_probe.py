from types import SimpleNamespace

import torch

from tacticalai.rim_probe import detections_from_result, summarize_frames


def test_detections_from_result_serializes_tensors():
    result = SimpleNamespace(boxes=SimpleNamespace(
        xyxy=torch.tensor([[1.0, 2.0, 30.0, 40.0]]),
        conf=torch.tensor([0.75]),
        cls=torch.tensor([1.0]),
    ))

    detections = detections_from_result(result, {1: "basketball hoop"})

    assert detections == [{
        "class_id": 1,
        "class_name": "basketball hoop",
        "confidence": 0.75,
        "bbox_xyxy": [1.0, 2.0, 30.0, 40.0],
    }]


def test_summarize_frames_does_not_call_frequency_accuracy():
    summary = summarize_frames([
        {"detections": [{"class_name": "basketball rim"}]},
        {"detections": []},
        {"detections": [{"class_name": "basketball rim"},
                         {"class_name": "basketball backboard"}]},
    ])

    assert summary["sampled_frames"] == 3
    assert summary["frames_with_any_detection"] == 2
    assert summary["detection_counts_by_prompt"] == {
        "basketball backboard": 1,
        "basketball rim": 2,
    }
    assert "not precision" in summary["note"]
