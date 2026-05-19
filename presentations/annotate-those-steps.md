# Annotate Those Steps -- how to curate analysis-ready MoBI datasets

A talk on the Mobile Brain/Body Imaging (MoBI) annotation gap and how the HEDit + Annotation Garden Initiative closes it. Case study: Peterson and Ferris (2018) perturbed beam-walking dataset (ds003739).

## Interactive Slides

<div class="embed-container">
  <iframe
    src="./annotate-those-steps/presentation.html?presentation=./annotate-those-steps.json"
    title="Annotate Those Steps -- HEDit for MoBI"
    frameborder="0"
    allowfullscreen>
  </iframe>
</div>

<p class="slide-hint">Use arrow keys to navigate. Press <kbd>F</kbd> for fullscreen, <kbd>S</kbd> for presenter view, <kbd>?</kbd> for shortcuts.</p>

<style>
.embed-container {
  position: relative;
  padding-bottom: 56.25%; /* 16:9 */
  height: 0;
  overflow: hidden;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 8px;
  margin-bottom: 1.5rem;
}
.embed-container iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border-radius: 8px;
}
.slide-hint {
  margin: -0.5rem 0 1.5rem 0;
  font-size: 0.75rem;
  color: var(--md-default-fg-color--lighter);
}
.slide-hint kbd {
  font-size: 0.7rem;
  padding: 0.1rem 0.3rem;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 3px;
  background: var(--md-code-bg-color);
}
</style>

## Abstract

Mobile Brain/Body Imaging (MoBI) experiments capture rich synchronized streams: high-density EEG, motion capture, electromyography, force, and virtual reality. The annotations that reach re-users are typically a handful of string labels. Peterson and Ferris (2018) shared one of the cleanest MoBI datasets on NEMAR (ds003739) and even there, the events.tsv carries no pull force, no rotation angle, no gait phase, and no balance-loss flag. That gap is a format and culture problem, not a data limitation; the motion-capture markers, load-cell traces, and condition metadata are already on the lab's disk.

This talk frames that asymmetry as the central MoBI annotation problem. The Hierarchical Event Descriptors (HED) sidecar pattern carries column-level semantics in events.json without touching the timeseries, but hand-tagging stalls in practice. HEDit, a multi-agent LangGraph pipeline, converts plain-English event descriptions into validated HED tags. The Annotation Garden Initiative wraps that engine and a future GUI (hedify) into a community-curated commons. The case study walks through what ds003739 shares today, what is missing, and what a HEDit-enriched sidecar restores for downstream analysis.

## Talk outline

- **Why MoBI annotation is hard** -- rich raw data, thin shared annotations
- **Case study: ds003739** -- beam walking with mediolateral pull and visual rotation perturbations
- **Three gaps** -- gait phases and balance loss; pull force and rotation parameters; trial context
- **The fix in principle** -- HED sidecars
- **Why hand-tagging stalls** -- schema size, JSON complexity, cryptic validation
- **HEDit today (v0.6)** -- natural-language to HED via Parser, Tagger, Validator agents
- **Demo** -- before/after on the ds003739 sidecar
- **Roadmap** -- hedify GUI (planned), Annotation Garden ecosystem

## Links

- **Project repository**: [github.com/Annotation-Garden/hedit](https://github.com/Annotation-Garden/hedit)
- **API endpoint** (planned): `api.annotation.garden/hedit`
- **Annotation Garden Initiative**: [annotation.garden](https://annotation.garden)
- **HED schemas and validator**: [hedtags.org](https://hedtags.org)
- **Dataset**: [ds003739 on NEMAR](https://nemar.org/dataexplorer/detail?dataset_id=ds003739) / [OpenNeuro](https://openneuro.org/datasets/ds003739)
- **Reference paper**: Peterson SM and Ferris DP (2018). *Differentiation in theta and beta electrocortical activity between visual and physical perturbations to walking and standing balance.* eNeuro. [doi:10.1523/ENEURO.0207-18.2018](https://doi.org/10.1523/ENEURO.0207-18.2018)

## Acknowledgements

Steven Peterson and Daniel Ferris for the open dataset; the HED working group and EEGLAB team; the Swartz Center for Computational Neuroscience; and the Annotation Garden Initiative contributors.

---

<small>Interactive slides built with [Agentic Presentation Builder](https://github.com/casual-vibers/agent-presentation), a JSON-to-Reveal.js presentation tool.</small>
