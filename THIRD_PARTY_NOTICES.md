# Third-Party Notices

NadirClaw is MIT-licensed. It can optionally use the following third-party
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
