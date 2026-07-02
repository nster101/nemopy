---
description: (legacy) scaffold the current repo for the crew — use the setup skill instead
---
The scaffolder is now the **`setup` skill**. Run it instead:

```
/the-girls:setup
```

(Or just ask: "set up this repo to use the agent crew.") It creates `coordination/`,
the `.claude/settings.json` permissions allowlist, the per-repo `SessionStart` hook,
optional `/einstein`-style persona commands, and a `CLAUDE.md` section. See
`skills/setup/SKILL.md` for the details.
