# Harness²

Parts of agent-reflect are adapted from Harness² (https://github.com/google-research/harness2,
read at commit 6868817518876de8fd170c6a8354daca273b6191).

Copyright 2026 Google LLC. Licensed under the Apache License, Version 2.0: see [LICENSE](LICENSE)
in this folder. The rest of agent-reflect is under the MIT License in the repository root.

## What is adapted, and how it was changed

- `agent_reflect/prompts/reflect.md` adapts the experience and evidence-boundary prompts of
  `harness2/core/prompts.py`: translated into Russian and rewritten for reading Claude Code
  session transcripts. Its first line says so.

## Ideas taken, written anew

These follow Harness²'s method; no text of it is copied:

- the second look at a new lesson runs on another model (`Config.verify_model`);
- a hint shown right after an error whose signature recurs (`error_trigger`, the guard hook after
  a shell command);
- a blocking hook only on repeated evidence that a shown rule is ignored (`guard block`, which only
  the user turns on);
- a verdict for each edit after a window — helped, not helped, confounded — preferring confounded
  when unsure (`agent_reflect/effect.py`);
- workspace facts kept apart from procedures, with how to check them again (`kind: fact`);
- a file every session loads does not grow: a rule that goes in takes the place of one it covers
  (`edits.growth_error`).
