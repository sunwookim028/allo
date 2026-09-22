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
