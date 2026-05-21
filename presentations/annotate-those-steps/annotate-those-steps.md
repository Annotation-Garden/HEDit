# Annotate Those Steps -- how to curate analysis-ready MoBI datasets

> *Source content (abstract + blog body) for the talk. The deck source lives next to this file as `annotate-those-steps.json`. The rendered presentation viewer and the blog page are hosted by the [Annotation Garden documentation site](https://docs.annotation.garden).*

## Abstract

Mobile Brain/Body Imaging (MoBI) experiments capture rich synchronized streams: high-density electroencephalography (EEG), motion capture, electromyography (EMG), force, and virtual reality (VR). The annotations that reach re-users are typically a handful of string labels. Peterson and Ferris (2018) shared one of the cleanest MoBI datasets on NEMAR (ds003739) and even there, the events.tsv carries no pull force, no rotation angle, no gait phase, and no balance-loss flag. That gap is a format and culture problem, not a data limitation; the motion-capture markers, load-cell traces, and condition metadata are already on the lab's disk.

This talk frames that asymmetry as the central MoBI annotation problem. The Hierarchical Event Descriptors (HED) sidecar pattern carries column-level semantics in events.json without touching the timeseries, but hand-tagging stalls in practice. HEDit, a multi-agent LangGraph pipeline, converts plain-English event descriptions into validated HED tags. The Annotation Garden Initiative wraps that engine and a future GUI (hedify) into a community-curated commons. The case study walks through what ds003739 shares today, what is missing, and what a HEDit-enriched sidecar restores for downstream analysis.

## The MoBI annotation gap

A MoBI session captures six or more synchronized streams: 128-channel EEG at 512 Hz, lower-leg and neck EMG at 1 kHz, full-body motion capture with 16 reflective markers, force from load cells in the perturbation rig, VR rotation timestamps, and a task/behavior log. Each is recorded with timing that lines up to the millisecond on the lab's disk.

What reaches NEMAR, OpenNeuro, or any downstream re-user is `events.tsv` -- a small text file with four columns: `onset`, `duration`, `trial_type`, `value`. For ds003739, the `value` column is one of `pull_left`, `pull_right`, `rotate_cw`, `rotate_ccw`. That is it.

This is not a worst case. ds003739 is one of the cleanest MoBI shares in existence; Peterson and Ferris published the dataset under CC0 with extensive documentation, and the events file is unambiguous within the four labels it provides. The point of the case study is that even the cleanest shared dataset hides most of the experiment's metadata behind a paper paywall and the lab's local hard drives.

## Three concrete gaps

**Gait phases and balance-loss events.** The 16 motion-capture markers track foot, sacrum, neck, and head trajectories. Heel-strike timing falls out of foot-marker velocity peaks; balance loss is detectable from harness-load spikes or from center-of-mass displacement crossing a threshold. None of this reaches the shared events file. A re-analysis that wants to phase-lock theta oscillations to heel-strike has to start by re-deriving the heel-strike column from raw motion capture.

**Physical parameters of the perturbation.** The mediolateral pull is delivered by an electromechanical motor with an inline load cell on the cable. Peak force on each trial varies between roughly 40 and 80 N. Visual rotations are 20 degrees over 0.5 seconds, but the per-trial direction and angular velocity are recorded only in the experimenter's MATLAB log. A dose-response analysis on theta as a function of pull force is, today, blocked by the format.

**Trial context.** Was the subject standing or walking? Pull or rotate? Which session? All of that is implied by the BIDS filename (`task-pullstand`, `task-pullwalk`, etc.). Filenames are human-readable but not designed to be parsed; a re-user opening `sub-003_ses-02_task-pullwalk_events.tsv` learns the condition from the string in the filename, not from a column in the event row.

## HED in 60 seconds

Hierarchical Event Descriptors (HED) is a controlled vocabulary maintained at [hedtags.org](https://hedtags.org). A HED tag is a comma-separated path through a schema tree: `Sensory-event, Visual-presentation, Rotation/(Direction/Clockwise), (Magnitude, deg, #)`. Tags inherit meaning from their parents, compose with one another, and validate against the schema. Value slots (the `#`) accept typed numbers with units. The schema is a contract; if a tag is in it, the tag means the same thing across labs.

HED rides on the BIDS sidecar pattern. `events.tsv` stays unchanged; `events.json` grows a `HED` key under each column, attaching tag strings either to the column as a whole or to the levels of a categorical value. This keeps existing analyses working while making every event row machine-queryable.

## Why manual HED tagging stalls

Three frictions stall HED adoption in MoBI labs that did not co-author the schema:

1. **Reading the paper** to extract event semantics at the right granularity takes hours per session.
2. **Learning the schema** -- roughly 2000 tags, deep hierarchy, value-slot conventions -- is an expert-level task.
3. **Writing JSON** with the right escaping, value-slot syntax, and column-versus-value scope is error-prone, and the validator messages are cryptic.

The net effect is that HED stays in the labs that build it. Everyone else ships `events.tsv` with four string labels.

## HEDit -- what it does today

HEDit (v0.6) is a natural-language to HED tag converter. The user writes a plain-English description of each event level -- what would already go in the README -- and HEDit returns a validated HED tag string ready to drop into `events.json`.

Under the hood, three LangGraph agents grounded in the HED schema:

- **Parser** extracts structured facts from the description: action, body part, direction, magnitude, unit.
- **Tagger** retrieves candidate HED nodes via retrieval-augmented generation over the schema and composes the tag string.
- **Validator** runs the official HED JavaScript validator on the composed tag. If it fails, the error context is fed back to the tagger for another shot.

The schema is the contract; agents do not invent vocabulary. The output is a BIDS-compliant `events.json` with HED keys plus a provenance trail noting which paper section and which schema version each tag was derived from.

Today's scope is deliberately narrow: **one detailed natural-language description in, one validated HED tag string out, per event-value level.** Two shape constraints follow from that:

- HEDit is tuned for *paragraph-grade* detail. Terse inputs ("same, right side") produce thin tags. Each description should include action, body part, direction, magnitude, unit, duration, modality, and experimental context.
- HEDit does not yet emit composite tags for events that co-occur (e.g. *pull while walking*). Each event-value level is tagged independently. Compositional annotation, paper PDF parsing, `events.tsv` ingestion, and a GUI editor all land in the planned `hedify` layer (target: 2026 Q4).

## The Annotation Garden Initiative

HEDit is one component of the [Annotation Garden Initiative](https://annotation.garden), an open infrastructure for collaborative annotation of naturalistic neuroscience stimuli. The initiative combines Stim-BIDS (BEP044), HED, and GitHub-versioned annotations into a community-curated commons. Companion projects include `image-annotation` (vision-language-model-based image annotation), `hedify` (the planned GUI editor), and flagship-dataset curation (Natural Scenes Dataset, Forrest Gump, the HBN movies).

The journey is: write descriptions today (HEDit CLI), get an audited GUI tomorrow (hedify), share validated annotations with the field (annotation.garden).

## Talk outline (slide map)

The 20-slide deck (`annotate-those-steps.json`) covers, in order:

1. Title slide
2. MoBI as a multi-stream experiment
3. The annotation reality: rich raw data, thin shared annotations
4. Case study: ds003739 -- beam walking with pull and visual-rotation perturbations
5. One trial = baseline + perturbation + recovery
6. Gap #1 -- gait phases and balance loss
7. Gap #2 -- pull force and rotation parameters
8. Gap #3 -- the why behind the trial
9. Re-analyses you cannot do today
10. The fix in principle -- HED
11. The HED sidecar pattern
12. Why current HED workflows stall
13. HEDit -- what it does, today
14. HEDit -- the architecture
15. Inside the pipeline -- three agents, one schema
16. Demo -- the input you write
17. Demo -- the output HEDit returns
18. What's next -- hedify, the GUI
19. How it all fits -- the Annotation Garden
20. Try it / contribute / cite

## References

- Peterson, S. M. and Ferris, D. P. (2018). Differentiation in theta and beta electrocortical activity between visual and physical perturbations to walking and standing balance. *eNeuro* 5(4). [doi:10.1523/ENEURO.0207-18.2018](https://doi.org/10.1523/ENEURO.0207-18.2018)
- Dataset: [ds003739 on NEMAR](https://nemar.org/dataexplorer/detail?dataset_id=ds003739) / [OpenNeuro](https://openneuro.org/datasets/ds003739)
- HED schema and validators: [hedtags.org](https://hedtags.org)
- Stim-BIDS (BEP044): [bids.neuroimaging.io](https://bids.neuroimaging.io)
- HEDit repository: [github.com/Annotation-Garden/HEDit](https://github.com/Annotation-Garden/HEDit)
- Annotation Garden Initiative: [annotation.garden](https://annotation.garden)

## Acknowledgements

Steven Peterson and Daniel Ferris for the open dataset; the HED working group and the EEGLAB team; the Swartz Center for Computational Neuroscience; and contributors to the Annotation Garden Initiative.
