# Third-Party Notices

NadirClaw is licensed under the PolyForm Noncommercial License 1.0.0. It can optionally use the following third-party
components, declared as opt-in extras. Their licenses and attributions are
reproduced here.

## headroom-ai

- **Used by:** the optional `headroom` optimizer backend
  (`NADIRCLAW_OPTIMIZE_BACKEND=headroom`), installed via `pip install nadirclaw[headroom]`.
- **Project:** Headroom — https://github.com/chopratejas/headroom
- **License:** Apache License 2.0
- **NOTICE:** Headroom, Copyright 2025 Headroom Contributors.

NadirClaw integrates Headroom only through its public Python API
(`headroom.compress`); no Headroom source code is copied or vendored into this
project. A full copy of the Apache License 2.0 is available at
https://www.apache.org/licenses/LICENSE-2.0 and is distributed with the
`headroom-ai` package when installed.

## laya and the Laya typed-decisions weights

- **Used by:** the optional Laya classifier (`NADIRCLAW_COMPLEXITY_ANALYZER=laya`),
  installed via `pip install nadirclaw[laya]`.
- **Project:** Laya, by Convai Innovations. Package https://pypi.org/project/laya/ (pinned `0.3.20`);
  weights https://huggingface.co/convaiinnovations/laya-typed-decisions (pinned revision
  `1a793eb568e6718f15941d08f85432581df534e3`).
- **License:** Apache License 2.0 (package and weights).

NadirClaw calls Laya only through its public Python API (`laya.load`, `Agent.predict`).
No Laya source code or weights are copied or vendored into this project; the weights are
downloaded from the Hugging Face Hub on first use. A full copy of the Apache License 2.0
is available at https://www.apache.org/licenses/LICENSE-2.0.
