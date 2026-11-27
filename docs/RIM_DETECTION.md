# Basketball rim detection probe

## Conclusion

The sampled zero-shot YOLO-World baseline did **not** detect the real rim reliably
in either supplied clip. It is not suitable for integration into the POC based on
this run. Standard YOLO26 COCO checkpoints are not a rim baseline because COCO has
`sports ball` but no `rim`, `hoop`, or `backboard` class.

This does not show that YOLO-style models cannot detect rims. It shows that this
specific open-vocabulary checkpoint and prompt setup failed on the supplied
broadcast footage. The appropriate next comparison is a basketball-specific
fine-tuned checkpoint evaluated against a small independent set of rim boxes.

## Executed experiment

- Model: Ultralytics `yolov8s-worldv2.pt`, downloaded by Ultralytics 8.4.158.
- Device: local CPU.
- Input: the original two clips; originals were not modified.
- Sampling: every 10th source frame, `imgsz=960`.
- First variant: prompts `basketball rim`, `basketball hoop`, and
  `basketball backboard`, confidence 0.05.
- Control variant: only `basketball rim`, confidence 0.01, to remove prompt
  competition and inspect very weak candidates.
- Outputs: sampled annotated MP4, JSONL detections, summary, run configuration,
  package versions, timings, hashes, and source-video hash.

Observed prompt-match frequency:

| Variant | Clip 1 | Clip 2 | Visual review |
|---|---:|---:|---|
| Three prompts, conf 0.05 | 2/15 sampled frames | 7/18 sampled frames | Real rim missed; candidates were false positives |
| Rim only, conf 0.01 | 0/15 sampled frames | 9/18 sampled frames | Real rim missed; candidates were false positives |

The false positives included the basketball, scoreboard/team graphics, and small
high-contrast background structures. The actual rim and net were visibly present
in reviewed frames but were not boxed. These counts are development diagnostics,
not precision/recall: no complete rim ground-truth set was used.

The first-pass artifacts are under:

- `outputs/rim-probe-yoloworld-clip1-v1/`
- `outputs/rim-probe-yoloworld-clip2-v1/`
- `outputs/rim-probe-yoloworld-rim-only-clip1-v1/`
- `outputs/rim-probe-yoloworld-rim-only-clip2-v1/`

## Reproduce

Install the optional prompt encoder dependency once:

```powershell
python -m pip install -e ".[rim-probe]"
```

Run the stricter three-prompt probe:

```powershell
python scripts/probe_rim_detector.py `
  --source data/raw/attack_clip.mp4 `
  --output outputs/rim-probe-yoloworld-clip1-v1 `
  --model yolov8s-worldv2.pt `
  --prompts "basketball rim" "basketball hoop" "basketball backboard" `
  --sample-stride 10 --imgsz 960 --conf 0.05 --device cpu
```

The output directory must not already exist; the probe refuses to overwrite a
previous run.

## Recommended next experiment

1. Label the rim once on about 20-30 diverse frames across the two clips,
   including partial visibility and the clip-2 pan. Keep some frames held out.
2. Compare at least one basketball-specific YOLO-family checkpoint against the
   same images and thresholds. Public model pages are leads, not trustworthy
   accuracy evidence; inspect dataset provenance, leakage, license, and domain.
3. Report precision/recall and box IoU on the held-out frames. Also report temporal
   coverage because a mostly static rim can be propagated between confident
   measurements, but propagation must remain distinct from detection.
4. If the available checkpoints fail, fine-tune a small detector for a broader
   `hoop assembly` or `backboard` target first, then estimate the rim anchor inside
   that stable structure. The larger backboard is usually a more learnable target
   in wide broadcast views than the thin rim itself.

Do not infer shot success merely from a ball/rim 2D overlap. Occlusion, perspective,
and ball depth make scoring a separate temporal event-estimation problem.
