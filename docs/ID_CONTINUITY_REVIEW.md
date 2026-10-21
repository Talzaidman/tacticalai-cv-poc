# Clean ID-continuity review

Local page: `outputs/id-continuity-review-v1/index.html`, served on port 8885.
Both clips default to confirmed tracks whose automatic court gate is inside.
Pending detections and boundary/uncertain cases are off in this page and the
coverage presentation. These controls affect display only, not association.

The ID-source selector keeps two existing automatic results separate:

- **Updated T IDs:** updated coverage tracking, with online appearance ReID
  enabled (`yolo26n-cls.pt`), GMC sparseOptFlow and buffer 30. New tracker gates
  can change association behavior. ReID being enabled is verified in run
  metadata; it does not establish ID stability or physical-person accuracy.
- **Earlier P IDs:** prior automatic temporal splitting and CLIP-ReIdent
  gallery linking, including raw T labels for comparison. These use the older
  detection/tracking observations. They are not a newly computed identity
  solution for the updated higher-coverage tracks.

No manual role tags, identity corrections or known-switch rules are applied.
Every displayed non-unknown ID is checked for uniqueness within its frame;
this does not prove a physical player keeps the same ID over time.
Unknown P identity remains `?` with its source track shown.

Select an ID to highlight it, optionally hide other IDs, inspect the observation
timeline and jump to first/last appearance. Slow playback and frame stepping
allow visual inspection of switches. The timeline describes saved observations,
including observations hidden by court controls. Filtering a boundary person
from the display can create a visible gap without a tracking failure.

Both source modes have clean H.264 exports for both complete clips. Exports
use the default filters; interactive ID focus does not alter the downloads.
No networks, tracking or identity inference are rerun for this presentation.
Hashes, source definitions and frame-count verification accompany the page.
All four clean MP4s were fully decoded at 150/175 frames. Artifact hashes and
null role fields were checked. Browser checks covered both clips and source
selectors, T/P highlighting, first/last appearance jumps, slow playback and
default-off filters. Initial media setup is cleared after loading so later
frame jumps do not reset to frame zero.

```powershell
.venv/Scripts/python.exe -m tacticalai.id_review --output outputs/id-continuity-review-v1
.venv/Scripts/python.exe scripts/serve_review.py outputs/id-continuity-review-v1 --port 8885
```

Use a fresh output directory. Default inputs are the existing coverage review
and automatic-only earlier identity presentation; override `--coverage` and
`--linked` as needed.
