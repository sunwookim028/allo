# dev/

Working notes for whoever is developing this fork next, not documentation
for a reader of the published site (https://sunwookim028.github.io/allo/).
Nothing under here is in `docs/source/`, so Sphinx never builds it and it is
not on the site. It moved out on purpose, not by accident: see `CLAUDE.md`'s
page table for what belongs here versus in `docs/source/`, and why.

- `toolchains.rst` -- this host's paths, licences and env setup.
- `fork_maintenance.rst` -- branch layout and the upstream-merge procedure.
- `records/` -- dated measurement records (`*.rst`) and the raw evidence
  behind TinyTPU-isa's published numbers (`records/tinytpu/`): csynth/cosim
  logs, gap-attribution results, and the CHIA harness's run evidence. Read
  once, by us; never re-run, never asserted.
- `repo_layout.md` -- the target layout for designs, flows and the core
  package, and what is wrong with today's.
- `systemc/` -- the working notes that arrived with the SystemC emitter, from
  `choonsik1/allo` (`SystemC-emitter`). They are that author's notes about
  that fork, kept whole for provenance rather than rewritten: `BACKEND.md`
  and `SYSTEMC_BACKEND.md` describe the emitter, `DATAFLOW_LINKS.md` the
  `Stream`/`Wire`/`Channel` link types, `ALLO_GOTCHAS.md` is worth reading
  before writing Allo, and `noc/FINDINGS_wire_channel.md` carries the
  measurements `docs/source/backends/catapult.rst` cites. Two caveats:
  `README.md` and `STATE.md` describe that fork's branches and remotes, not
  this one's, and `SIMULATOR.md` documents the timed dataflow simulator on
  `SystemC-emitter` that this fork has **not** merged.
- `docs_style.md` -- how the published pages are written.
- `paper_outline.md` -- claim, evidence and gap per section.
- `asic_handoff.md` -- the ASIC/PD evaluation handoff.
- `SESSION_REPORT.md` -- what the last working session produced.
