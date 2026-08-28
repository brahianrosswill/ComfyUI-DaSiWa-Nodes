# Director 2.0 — frozen

The **Director 2.0** variant is **frozen** (2026-08-28) and is **not shipped** in
the nodepack. Development is paused; the code is preserved here, not deleted, so
it can be re-activated later.

## What Director 2.0 was

A self-contained variant of the MiniMax H3 Director: instead of the shipped
Director + Guide two-node flow (author timeline → thin Guide adapter → native
ComfyUI H3 sampler/decoder), the 2.0 variant ran the whole pipeline **inside the
node** — in-node sampling, tiny-VAE step decode/preview, and direct media
output. Distinct surface vs. the shipped Director:

- **In-node execution** (sampler/scheduler/steps/shift_video/shift_audio),
  persisted in the hidden `internal_execution` block.
- **External override sockets**: `external_sampler`, `external_scheduler`,
  `external_steps`, `external_shift_video`, `external_shift_audio`
  (precedence external > internal > default).
- **Built-in live step preview**: per-step tiny-VAE decode streamed into the
  node's own Preview & Output panel (`preview_tiny_vae` combo widget,
  `preview_vae` socket, `preview_max_resolution`, `preview_frames`, `preview_fps`).

Everything else (timeline lanes, prompt builders, resolution panel, seed
control, reference handling) is shared with the shipped Director and lives in the
normal `nodes/` + `js/` tree, not here.

## Files in this folder

| File | Role |
|---|---|
| `nodes_minimax_h3_director_v2.py` | `MiniMaxH3DirectorV2` node class + `NODE_CLASS_MAPPINGS` |
| `helper_minimax_h3_director_v2.py` | v2 prompt-builder / assembly helpers |
| `helper_minimax_h3_director_execute_v2.py` | in-node execution / sampling logic |
| `helper_minimax_h3_director_preview_v2.py` | tiny-VAE step-decode / preview |
| `helper_minimax_h3_prompt_builder_v2.py` | v2 prompt builder |
| `helper_media_output_v2.py` | media output helpers |
| `minimax_h3_director_v2.js` | v2 frontend (registers `MiniMaxH3DirectorV2` only) |
| `test_minimax_h3_director_v2.py` | v2 pytest tests (path-based loading) |
| `test_director_seed_control.mjs` | v2 seed-control source-pinning test |
| `test_minimax_h3_director_execute.py` | v2 execute tests (path-based loading) |
| `test_minimax_h3_director_output_bridge.py` | v2 output-bridge test |
| `test_minimax_h3_director_output_publisher.py` | v2 output-publisher test (path-based loading) |
| `test_director_field_heights.py` | v1+v2 field-height test (v1 refs point to repo `js/`) |

## Why it was removed from the pack

The shipped release keeps the two-node **Director + Guide** flow and the
standalone Seed Control node; the self-contained 2.0 execution/preview stack was
parked so the nodepack could ship without carrying the in-node sampling surface.
No version bump — this is a removal/freeze, not a new feature.

## Re-activation recipe

To ship Director 2.0 again, do the **inverse** of the freeze:

1. **Move the modules back to `nodes/`:**
   ```sh
   git mv frozen/director_v2/nodes_minimax_h3_director_v2.py      nodes/
   git mv frozen/director_v2/helper_minimax_h3_director_v2.py     nodes/
   git mv frozen/director_v2/helper_minimax_h3_director_execute_v2.py nodes/
   git mv frozen/director_v2/helper_minimax_h3_director_preview_v2.py nodes/
   git mv frozen/director_v2/helper_minimax_h3_prompt_builder_v2.py nodes/
   git mv frozen/director_v2/helper_media_output_v2.py            nodes/
   git mv frozen/director_v2/minimax_h3_director_v2.js            js/
   ```
   (`git mv` the frozen tests back into `.tests/` / `test/` if you want them in
   the live suite again; or keep running them in place via the paths below.)

2. **Re-register the node in `__init__.py`** (three touch points):
   - import: `from .nodes.nodes_minimax_h3_director_v2 import MiniMaxH3DirectorV2`
   - `NODE_CLASS_MAPPINGS["MiniMaxH3DirectorV2"] = MiniMaxH3DirectorV2`
   - `NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3DirectorV2"] = "MiniMax H3 Director 2.0"`

3. **Re-add it to the bug-report node list** in
   `.github/ISSUE_TEMPLATE/bug-report.yml`.

4. **Version bump** — re-shipping the variant is a new feature, so bump
   `pyproject.toml` (per the user's version policy: new features DO bump).

5. **Docs** — un-freeze the README note + the two `_ (Director 2.0 — frozen)_`
   sections in `docs/minimax_h3_director.md`, and restore this file's link
   targets.

### Verifying the frozen tests in place

The frozen tests load their modules by **absolute path** relative to the repo
root (`Path(__file__).resolve().parents[2]`), so they run without moving them
back:

```sh
python -m pytest frozen/director_v2/ -q          # pytest suites
node frozen/director_v2/test_director_seed_control.mjs   # node source-pinning
```

The mixed v1+v2 `test_director_field_heights.py` asserts against the **shipped**
`js/minimax_h3_director.js` (v1) and the co-located `minimax_h3_director_v2.js`
(v2); run it from this folder so both resolve.
