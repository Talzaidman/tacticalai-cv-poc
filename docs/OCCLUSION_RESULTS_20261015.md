# Occlusion experiment results — 2026-10-15

Run: `occlusion-20261015T204150Z-029a7a35`.
[Drive outputs](https://drive.google.com/drive/folders/16nzMF9OsmJFcjl0XgKD28gISebaecF2t).

All 16 runs completed. The remote suite reports passed artifact checks, no
execution errors, and four successfully rendered previews. The reports, four
comparison MP4s, and all 16 metadata/frames.jsonl files were retrieved locally
under `outputs/occlusion-20261015T204150Z-029a7a35/`. The 368.9 MB full ZIP exceeds
the connector's 256 MiB limit; individual retrieval succeeded.

All four downloaded comparison videos decoded completely: 150 or 175 frames
at 25 FPS. Inspection of all run metadata confirms the requested 30/60-frame
retention and a consistent ReID encoder SHA256 prefix `0dd6f8dbc4`. Detector
output hashes are identical across all eight runs of each clip. Thus changed
tracking results here are not explained by changed primary detections.

## Counts and runtime

Each cell lists **unique IDs / track observations / pipeline FPS**. These
include all tracked people; they are not player accuracy or ID-switch metrics.

| Variant | Clip 1 BoT-SORT | Clip 1 TrackTrack | Clip 2 BoT-SORT | Clip 2 TrackTrack |
|---|---|---|---|---|
| Baseline | 37 / 2399 / 14.98 | 26 / 1267 / 17.80 | 64 / 2751 / 7.41 | 27 / 1848 / 8.88 |
| Buffer 60 | 36 / 2399 / 15.78 | 25 / 1285 / 18.61 | 64 / 2751 / 7.72 | 27 / 1848 / 8.82 |
| ReID | 46 / 2521 / 8.17 | 18 / 1358 / 13.35 | 87 / 3384 / 4.49 | 18 / 1894 / 6.80 |
| Buffer 60 + ReID | 46 / 2521 / 8.19 | 18 / 1358 / 13.57 | 87 / 3384 / 4.55 | 18 / 1894 / 7.30 |

These are single sequential runs with warmup and I/O, not isolated tracker
benchmarks. ReID adds substantial cost. Increased observations can include
referees, spectators and incorrect associations.

## Effect of retention

For each run, IDs were normalized by first observed appearance, then full
per-frame records were compared. ReID versus buffer60+ReID is identical on all
four clip/tracker combinations (zero changed frames). Baseline versus buffer60
is identical on clip 2 for both trackers. On clip 1, 43 frames differ for
BoT-SORT and 39 for TrackTrack. These are changed output frames, not counts of
ID switches or verified improvements.

There is no demonstrated benefit from increasing retention once ReID is
enabled on these two clips. This is consistent with many failures happening
before track expiry, or returning detections failing association; it does not
identify the cause of every failure.

## Appearance matching and visual inspection

TrackTrack with ReID is a candidate for further evaluation: unique IDs decline
26 to 18 and 27 to 18 while track observations increase. BoT-SORT with this
generic classification encoder produces more IDs and observations on both
clips; there is no basis to call it an accuracy improvement.

Sampled frames at 1–5 seconds were inspected, along with a focused clip 1
TrackTrack comparison at 1.0, 2.4 and 4.6 seconds. The example shows baseline
IDs 180/197/200 versus ReID ID 85 in the selected spatial region. At 2.4 seconds
the selected detection overlaps yellow- and black-shirted players. This is
ambiguous identity evidence: apparent numerical continuity may conceal a
wrong-person association or mixed detection. It is not counted as a verified
successful recovery. Selected examples are not an exhaustive frame-by-frame
identity annotation of all players.

## Next decision

Keep the original baseline and TrackTrack+ReID as comparison candidates;
buffer60+ReID adds no tracking change here. Annotate several actual player
occlusions with the same physical player's identity before and after each gap,
count correct reconnections and wrong-person matches, and inspect missed or
mixed detections. Do not choose the winner using unique IDs alone. Team-aware
association or a sports-trained appearance encoder may be investigated after
these failure events are labeled. No off-screen/hidden positions were measured.

Comparison layout: baseline top left, buffer60 top right, ReID bottom left,
buffer60+ReID bottom right. IDs are local to each run.

## Verified clip 1 event: 87 to 96 (follow-up inspection)

The user identified the yellow-shirted player starting as ID 87 and ending as
ID 96 in TrackTrack+ReID. Inspection of raw footage with exported boxes confirms
a wrong-person association: ID 96 already belonged to a dark-clothed sideline
person, before being assigned to the yellow-shirted player.

- Frame 104 (4.16 s): last exported observation of player track 87; ID 96 is
  simultaneously attached to the sideline person.
- Frame 130 (5.20 s): ID 96 remains on the sideline person.
- Frame 131 (5.24 s): ID 96 changes to a box on the yellow-shirted player and
  continues following that player through the end of the clip.

The interval from the last track-87 observation to the erroneous player match
is 27 frame intervals, or 1.08 s, within the 30-frame retention setting. ID 96
was not newly created at that moment. Retention expiry alone therefore does
not explain this event. Buffer60+ReID has identical output and the same failure.
The event establishes at least one identity-association failure, but not a
whole-clip ID-switch metric. Exact internal appearance/motion cost contributions
have not been logged or determined.

Visual evidence is saved locally as
`outputs/occlusion-20261015T204150Z-029a7a35/id87-to96-evidence.jpg`.
This motivates testing conservative court/role filtering before association
and rejecting cross-role matches, while also inspecting why the original
player track failed to reconnect. These changes are proposed, not validated.
