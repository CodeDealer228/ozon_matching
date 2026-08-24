---
name: ods-submissions
description: "Read-only viewer for the ODS.ai E-CUP 2026 'Матчинг товаров' submissions page (https://ods.ai/competitions/e-cup-2026-matching/submissions) via browser automation across possibly multiple logged-in browsers/accounts (e.g. Comet and Chrome, or a teammate's own login): checking submission status/score/leaderboard-relevant Mean PR-AUC, reading a failed submission's container log, and figuring out which ODS identity is behind a given browser session. Use whenever the user asks to check status/score of a real ods.ai submission, view why a submission Failed/errored, or asks about switching accounts/browsers to check ods.ai — even if they just say 'глянь статус на ODS' or 'что там с сабмитом' without naming the skill. This skill deliberately does NOT submit anything — see 'Out of scope' below for what to do if the user asks to submit. Not for Kaggle (see the kaggle/kaggle-experiment-safety skills) — this is the actual contest judging platform, separate from Kaggle dev/training infra."
allowed-tools: mcp__claude-in-chrome__list_connected_browsers mcp__claude-in-chrome__tabs_context_mcp mcp__claude-in-chrome__navigate mcp__claude-in-chrome__read_page mcp__claude-in-chrome__find mcp__claude-in-chrome__computer mcp__claude-in-chrome__get_page_text mcp__claude-in-chrome__switch_browser Bash Read
---

# ODS.ai E-CUP submissions — status viewer

## Out of scope: this skill does not submit

This skill is **read-only by design**. It does not fill in the file input,
does not touch the comment box, and must never click "Отправить решение" —
there is no submit flow documented here on purpose, not because one was
forgotten. If the user asks to submit/засабмитить a solution, say plainly
that this skill only covers checking status/score/logs, and that submitting
would need to be done by hand on the page (or via a separate skill, if one
exists) — don't try to improvise a submit flow from the page elements
described below.

## Requirements

This skill only works inside **Claude Code / the Claude app with the
"Claude for Chrome" browser extension** installed and connected (the
`mcp__claude-in-chrome__*` tools). It reads an already-open, already
logged-in browser tab — any Chromium-based browser the extension supports
(Chrome, Edge, Brave, Comet, etc.) is fine, but without that extension
connected there is no way to carry out anything below, and Claude should say
so plainly rather than attempt a workaround. It does not work with the
"Claude in Chrome" iOS/Android app integration, plain `WebFetch`, or any
non-browser HTTP approach — see the closing note in this file for why.

Page: `https://ods.ai/competitions/e-cup-2026-matching/submissions`. Requires
an existing ODS.ai login — **never log in yourself, never touch a password
field.** All automation here rides an already-authenticated browser session
(cookies), driven via the `claude-in-chrome` MCP tools. The in-app
`Claude_Browser` tool is a *different*, unauthenticated sandbox — useless
here, do not use it for this page.

## Page anatomy (as of 2026-08-20)

Interactive elements on `.../submissions` relevant to reading status (refs
are unstable across navigations/scrolls — always re-`read_page`/`find`
right before clicking, don't reuse refs from an earlier turn):

- **"Решения" table** (own submission history) — the main thing this skill
  reads. Columns: Время (relative, e.g. "19 часов назад"), Участник
  (team-member id), Имя файла (a direct link straight to the file on Yandex
  Object Storage —
  `storage.yandexcloud.net/ds-ods/files/submissions/{team_id}/{submission_id}/{filename}`,
  so past submissions can be re-downloaded without touching the UI), Статус,
  Mean PR-AUC, Описание (icon), Финал (checkbox — marks a submission as the
  final one to be judged).
- Above the table, near the (out-of-scope) submit form, a
  **"Не осталось попыток"** message appears when the team's daily submission
  quota is exhausted. Purely informational here — read it if the user asks
  about remaining attempts, but this skill never acts on it (no submit flow
  to gate).

## Reading status/score

Just navigate + `get_page_text`/`read_page`/screenshot the "Решения" table.
This is pure reading and needs no per-action permission — safe to run on a
schedule (see below) or on demand.

- The **Описание** column's icon already carries the submitted comment as
  its accessible name (visible via `read_page`/`find` without clicking) when
  non-empty — e.g. `button "catboost( PCA_32(BERTA+Lora embeddings)...)"`.
  Empty comment → icon has no name / renders grey.
- Статус is one of: `Success` (Mean PR-AUC populated), `Failed` (no further
  detail available), or a red **`Error: <reason>`** (e.g. "Container
  finished with non-zero exit code", "Container did not finish in time") —
  these carry a **"Показать" button**.

## Viewing a failed submission's log

Click that row's "Показать" button (grab a *fresh* ref for it first — a
stale ref from an earlier `find` call once mis-clicked and opened an
unrelated leaderboard link in a new tab). It opens a modal with the full
container stdout/stderr — confirmed real content, e.g. CUDA banner, then the
actual pipeline's own log lines, e.g.:

```
=== Starting FRIDA LoRA (adapter, no merge) prediction pipeline ===
[SETUP] offline-installing [...] ...
ERROR: Could not install packages due to an OSError: [Errno 30] Read-only
file system: '/root/.local'
```

Close with the `×` in the modal's top-right corner (`get_page_text` /
`read_page` right after click to pull the log text out, then screenshot to
sanity-check if the text extraction looks incomplete).

## Browsers and accounts

- `mcp__claude-in-chrome__list_connected_browsers` shows what's currently
  reachable — could be zero, one, or several instances depending on which
  browsers on this machine have the Claude-for-Chrome extension installed
  and active. Don't assume which one it is: if a connected instance's blank
  new-tab page is branded (e.g. "Perplexity" for Comet) that's a tell, since
  the UA string alone won't say "Comet" — it reports as a generic recent
  Chrome/Chromium build regardless of which Chromium-based browser it
  actually is.
- If the extension is installed in more than one browser, call
  `mcp__claude-in-chrome__switch_browser` to reach a different one (sends a
  connect request; **the user must click "Connect" inside the target
  browser** — tell them to do that and which browser you're waiting on). If
  no other browser currently has the extension active/listening, it returns
  "No other browsers available to switch to" — that's a signal to ask the
  user to open/focus that browser, not a dead end.
- Everything else in this file works identically regardless of which
  browser is currently connected — the page doesn't care.

### Which ODS identity is behind a given browser session

Don't assume — a browser could be logged into any of the team's accounts.
Check the profile link in the page's top-left avatar: `read_page`/`find` for
a link whose `href` matches `/users/<id>` (an opaque ODS user id). That same
id (or the person's display name) is also what shows up in the "Решения"
table's Участник column, so it lets you tell whose submission is whose
without asking.

### Multiple accounts, one shared status view

**Confirmed directly on this team's page: the "Решения" table and the daily
submission quota are shared by the whole team**, not per-account — the
table showed submissions from several different Участник entries (different
team members' own logins) all in one running list. That means checking
status from any teammate's browser/account gives you the same picture — if
one account's session isn't reachable, switching to another (via
`switch_browser`) is a legitimate way to keep monitoring, not just a
fallback. This is very likely a platform-wide mechanic (the data belongs to
the team entity, not the individual login), but if you're setting this up
for a different team, it costs nothing to re-confirm by checking whether
teammates' entries share one table before relying on it.

## Autonomous status monitoring

Reading the "Решения" table is read-only and safe to do unprompted, so it's
a good candidate for a recurring scheduled task (via the `scheduled-tasks`
MCP tools, or whatever cron mechanism your Claude setup offers) instead of
making the user ask every time. If the user wants this, set one up with:

- A cron cadence they choose (every 1-2h is reasonable for a competition
  with a daily submission cap — no point checking more often than the
  underlying data can change).
- A **self-contained prompt** (scheduled tasks start fresh, with no memory
  of this conversation): navigate to the submissions URL, read the table,
  diff against a small JSON cache file it maintains itself (any writable
  path under the user's own Claude config directory works — resolve it at
  setup time rather than hardcoding one person's path), and report only
  what changed since the previous run.
- Also make it runnable on-demand (most schedulers support a manual
  trigger), so it's useful right after the user submits by hand and wants a
  quick status check without waiting for the fixed cadence.

This is optional infrastructure each person sets up for their own machine —
it isn't part of what this SKILL.md file itself does.

## Why this depends on the Chrome extension

Everything here rides an already-authenticated cookie session that lives
inside a real browser — that's what lets Claude never touch a password. In
principle a headless/scripted browser (Playwright, Selenium, etc.) could
read the same page, but it would still need *some* logged-in session to
attach to, or Claude would have to log in itself — which the standing
safety rules already rule out (no entering credentials, ever). A raw HTTP
client reading the page's data API directly is also possible in theory, but
means depending on an undocumented, unstable internal endpoint instead of
the public UI, fragile to any front-end change, for a read-only convenience
that the extension already provides safely. The extension dependency isn't
an arbitrary implementation choice — it's what buys "never touches your
password, only reads a session you already opened."
