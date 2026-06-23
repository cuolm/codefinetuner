# Project Blueprint

Conventions and reusable scaffolding for any new repo. Language-agnostic in structure; per-language slots are called out.

> **AI agents:** scroll to [Runbook for agents](#runbook-for-agents) for an ordered checklist. Everything else is reference.

---

## 1. Branch flow

```
develop  →  release/<version>  →  main          (planned releases)
                hotfix/<version> →  main         (emergency patches)
```

- **`develop`** — daily integration target. All feature/fix PRs land here. **Never** the source branch of a PR to `main` (see "main" below).
- **`release/<version>`** (e.g. `release/0.1.0`) — pre-release stabilization branch cut from `develop`. Bump VERSION + CHANGELOG date here.
- **`hotfix/<version>`** (e.g. `hotfix/0.1.1`) — emergency patch branch cut directly from `main` when `develop` has unfinished work that can't ship. PR back to `main`; after merge, fast-forward `main → develop` so the fix enters integration. Use sparingly; prefer the regular `release/*` flow.
- **`main`** — stable releases only. Source of any PR to `main` is always `release/*` or `hotfix/*` — **never `develop` directly**. The repo's `delete_branch_on_merge: true` (section 5) auto-deletes the source branch on merge; routing through `release/*` / `hotfix/*` keeps `develop` from being wiped out by that setting on every release.

Never force-push shared branches. Never skip hooks.

---

## 2. Versioning

- **Source of truth:** a `VERSION` file at repo root (`0.1.0`, no `v` prefix). Manually bumped on the release branch before merging to `main`.
- **Semver:** strict. Breaking change → major. New feature → minor. Bug fix → patch.
- **Pre-release identifiers:** `-alpha.N`, `-beta.N`, `-rc.N` (incrementing per stage).
- **Dev builds:** `0.0.<run_number>-dev` from CI, never tagged.
- **Stamped into the binary** via build flags from CI:

  | Language    | How                                                                                                   |
  | ----------- | ----------------------------------------------------------------------------------------------------- |
  | Go          | `-ldflags="-X main.version=$(cat VERSION) -X main.commit=$GITHUB_SHA -X main.date=$(date -u +%FT%T)"` |
  | Rust        | Set `version` in `Cargo.toml` from `VERSION`; embed via `env!("CARGO_PKG_VERSION")`.                  |
  | Node/TS     | Mirror `VERSION` into `package.json` `version` field.                                                 |
  | Python      | Single-source via `__version__` re-exported from `_version.py` written by CI.                         |
  | C/C++       | `-DVERSION="..."` compile flag.                                                                       |

  In every case the binary should respond to `--version` with `<name> <version> (<commit>, <date>)`.

---

## 3. CI/CD workflows

Three (sometimes four) workflows. All use:

```yaml
concurrency:
  group: <name>-${{ github.ref }}
  cancel-in-progress: true
```

### 3.1 `ci.yml` — gate

Runs lint + build + test on every push to `main`, `develop`, `release/**` and on PRs to `main`/`develop`. Must pass before any other workflow's outcome matters. Matrix at minimum across Linux + macOS + Windows.

### 3.2 `prerelease.yml` — develop + release branches

Two jobs gated by `if: github.ref == ...`:

- **`dev-build`** (on `develop`): build snapshot artifacts, upload as workflow artifact named `<project>-0.0.<run>-dev`, retention 90d, trim to last 3 via `gh api`. Requires `permissions: actions: write`.
- **`prerelease`** (on `release/**`): parse base version from branch (`release/0.1.0` → `0.1.0`), pick stage from commit-message keyword (default `alpha`, `[beta]`, `[rc]`), count existing matching tags, increment, push tag, run release tooling (skip publish to package managers — those have `skip_upload: auto` for prereleases). Trim to last 5 prereleases via `gh release list/delete --cleanup-tag`.

### 3.3 `release.yml` — main

Triggered on push to `main`. Reads `VERSION`, idempotence-checks the tag exists, pushes `v<VERSION>`, runs release tooling to publish to all channels. Idempotent: if the tag already exists (e.g. main got a non-version-bump commit), the workflow exits cleanly.

```yaml
on:
  push:
    branches: [main]
permissions:
  contents: write
  packages: write   # if pushing container images
concurrency:
  group: release-main
  cancel-in-progress: false   # never cancel a release mid-flight
```

### 3.4 `pages.yml` (optional) — docs site

Triggered on `docs/**` changes on `develop` or `main`. Builds Jekyll (or Hugo/MkDocs/Astro) from `/docs`. Deploy step gated `if: github.ref == 'refs/heads/main'` because GitHub Pages environment is protected to the default branch.

---

## 4. Required files & directories

### Top-level

| File              | Purpose                                                                           |
| ----------------- | --------------------------------------------------------------------------------- |
| `README.md`       | Hero + install + usage + checks/feature table + branch-flow section + license link. |
| `LICENSE`         | MIT or Apache-2.0 unless there's reason otherwise.                                |
| `CHANGELOG.md`    | Keep a Changelog 1.1.0. `[Unreleased]` on top, versioned sections below.          |
| `CONTRIBUTING.md` | Quick start, branch flow recap, how to add a feature, PR expectations.            |
| `CODE_OF_CONDUCT.md` | Contributor Covenant 2.1.                                                       |
| `SECURITY.md`     | How to report vulns (private advisory link). Scope of attack surface.             |
| `VERSION`         | Single line, e.g. `0.1.0`.                                                         |
| `.gitattributes`  | `* text=auto eol=lf` + binary classifications. Kills CRLF warnings on Windows.    |
| `.gitignore`      | Language-specific + `dist/`, `node_modules/`, `.env`, build outputs, `.claude/`.   |

### `.github/`

| File / dir                              | Purpose                                                              |
| --------------------------------------- | -------------------------------------------------------------------- |
| `FUNDING.yml`                           | `github: <username>` (or other sponsor links).                       |
| `dependabot.yml`                        | Per-ecosystem entry, target `develop`, weekly schedule.              |
| `PULL_REQUEST_TEMPLATE.md`              | Summary, Changes, Test plan, screenshots-if-visual.                  |
| `ISSUE_TEMPLATE/bug_report.yml`         | YAML form, not markdown — better UX.                                 |
| `ISSUE_TEMPLATE/feature_request.yml`    | YAML form.                                                           |
| `ISSUE_TEMPLATE/config.yml`             | `blank_issues_enabled: false`, route questions → Discussions, security → private advisory. |
| `workflows/ci.yml` `prerelease.yml` `release.yml` `pages.yml` | See section 3.                            |

### Optional

- `docs/` — Jekyll/Hugo/MkDocs source for the Pages site.
- `.editorconfig` — consistent indentation across editors.
- `Dockerfile` — if shipping container images.

---

## 5. Repo settings (run once, via `gh`)

```sh
gh repo edit <OWNER>/<REPO> \
  --description "<one-sentence tagline, max 350 chars>" \
  --homepage "https://<owner>.github.io/<repo>/" \
  --enable-wiki=false \
  --enable-discussions=true \
  --delete-branch-on-merge=true \
  --add-topic <topic1> --add-topic <topic2> ...   # up to 20

# Enable Pages with Actions build type
gh api -X POST repos/<OWNER>/<REPO>/pages \
  -H "Accept: application/vnd.github+json" \
  -f "build_type=workflow"
```

`--delete-branch-on-merge=true` keeps `release/*` and `hotfix/*` tidy after they merge to `main`. It's the reason PRs to `main` must come from those branches and **not from `develop`** — the setting deletes the source branch of every merged PR, so PR'ing `develop` directly to `main` nukes `develop` on the remote.

**Topics:** pick 10–20 from the relevant ecosystem (language, domain, audience). They drive GitHub's discovery surface.

**Branch protection on `main`** (recommended once releasing):

```sh
gh api -X PUT repos/<OWNER>/<REPO>/branches/main/protection \
  --input - <<'EOF'
{
  "required_status_checks": {"strict": true, "contexts": ["lint-build-test"]},
  "enforce_admins": false,
  "required_pull_request_reviews": {"required_approving_review_count": 0},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
```

---

## 6. Distribution channels

Pick by audience, not by what's possible.

### Universal
- **GitHub Releases** — always. `goreleaser` (Go), `cargo-dist` (Rust), `release-please` (multi-lang) handle it.
- **`install.sh`** — POSIX-sh installer that detects OS/arch, downloads from latest release. Useful fallback.

### By language

| Lang     | Native channel        | Cross-channel       |
| -------- | --------------------- | ------------------- |
| Go       | `go install`          | Homebrew, Scoop, GHCR |
| Rust     | `cargo install`, crates.io | Homebrew, Scoop |
| Node/TS  | npm                   | (rarely external)   |
| Python   | PyPI (pip), pipx      | Homebrew            |
| Ruby     | RubyGems              | Homebrew            |
| C/C++    | distro repos, vcpkg, conan | Homebrew         |

### By OS (audience)

| OS          | Channel                | Effort                       |
| ----------- | ---------------------- | ---------------------------- |
| macOS       | Homebrew tap           | Low — goreleaser / cargo-dist |
| Linux       | Homebrew (linuxbrew), AUR, Snap | Low–medium          |
| Windows     | Scoop bucket           | Low — goreleaser             |
| Windows     | WinGet                 | Medium — PR per release      |
| Cross-distro Linux | Snap, Flatpak    | Medium                       |
| Containers  | GHCR (free), Docker Hub | Low — multi-arch via buildx |

### Cargo-cult skip list

- npm/pip/cargo for tools NOT written in that language unless the audience already lives there (e.g. esbuild on npm for JS-toolchain users).
- nixpkgs as the *first* channel — wait for community PRs once popular.
- Snap if your audience runs server-class Linux — they often disable snapd.

---

## 7. Commit & PR conventions

### Commits
Conventional commits style, lowercased subject, no trailing period:

```
feat: add no-flock heuristic for high-frequency jobs
fix: handle CRLF line endings in system crontabs
docs: clarify --calendar collision call-out format
chore(deps): bump actions/checkout from 4 to 6
ci: grant actions:write for artifact trimming
```

Keywords with workflow side-effects:
- `[beta]`, `[rc]` in commit messages on `release/*` branches → stage selection.

### PRs
- One logical change per PR.
- Title mirrors the commit subject style.
- Body uses the PR template: Summary, Changes, Test plan, Screenshots.
- CI must be green to merge.
- Squash or merge commit (pick one repo-wide; usually merge for branch-flow projects).

### CHANGELOG
- Add entries under `[Unreleased]` while working.
- The Unreleased heading **always names the target version**: `## [Unreleased] — next: 0.1.0`. This makes it obvious what queued entries will ship as and avoids version drift when the release is cut later.
- On release branch, rename to `[X.Y.Z] - YYYY-MM-DD` and start a fresh `[Unreleased] — next: <next version>` block.

---

## 8. What NOT to do

- ❌ Push directly to `main`.
- ❌ PR `develop` directly to `main`. Always cut a `release/*` (planned) or `hotfix/*` (emergency) branch first — otherwise the `delete_branch_on_merge` repo setting wipes `develop` on the remote.
- ❌ Force-push shared branches.
- ❌ Skip hooks (`--no-verify`).
- ❌ Commit secrets, `.env`, or local IDE state.
- ❌ Add a co-author trailer for AI tools (your call — but be consistent).
- ❌ Mock-up dynamic data in shields.io badges (e.g. fake "A+" Go Report Card).
- ❌ Declare any `permissions:` block in a workflow without listing every permission you need — unlisted ones default to `none`.
- ❌ Use `goreleaser` `repository:` blocks without explicit `token:` — defaults to `GITHUB_TOKEN` which is scoped to the current repo only and fails with 403 when pushing to a tap/bucket repo.

---

## 9. Tooling required on local machine

- `git` (with `user.name` and `user.email` set globally)
- `gh` CLI, authenticated (`gh auth status`)
- The toolchain for whatever language you're using
- `goreleaser` (Go projects) or equivalent
- A code editor with EditorConfig support

---

## 10. Required GitHub secrets per project

Set via `gh secret set <NAME> --repo <owner>/<repo>` or web UI:

| Secret                       | Required by                          | How to create                                                          |
| ---------------------------- | ------------------------------------ | ---------------------------------------------------------------------- |
| `GITHUB_TOKEN`               | every workflow                       | auto-provided by Actions                                               |
| `HOMEBREW_TAP_GITHUB_TOKEN`  | release.yml when publishing brew     | fine-grained PAT on `<owner>/homebrew-tap` with Contents: read/write   |
| `SCOOP_GITHUB_TOKEN`         | release.yml when publishing scoop    | fine-grained PAT on `<owner>/scoop-bucket` with Contents: read/write   |
| `WINGET_PKGS_GITHUB_TOKEN`   | release.yml when publishing winget   | fine-grained PAT on your fork of `microsoft/winget-pkgs`               |
| `NPM_TOKEN`                  | npm publish                          | npm.com → Access Tokens → Automation                                   |
| `PYPI_API_TOKEN`             | PyPI publish                         | pypi.org → Account → API tokens                                        |
| `CARGO_REGISTRY_TOKEN`       | crates.io publish                    | crates.io → Account → API Tokens                                       |

---

## 11. Runbook for agents

You're being asked to scaffold a new project (or retrofit an existing one). Execute in order; don't skip steps.

1. **Determine basics** with the user (or read from existing files):
   - Project name, owner, primary language, one-sentence description, license.
2. **Repo metadata** via `gh repo edit` (section 5). Topics + description + homepage + discussions on + wiki off.
3. **Enable Pages** with `build_type: workflow` (section 5).
4. **Top-level files** (section 4). Use the templates in this blueprint as starting points; replace placeholders.
5. **`.github/` files** (section 4). YAML issue forms (not markdown).
6. **VERSION = `0.0.1`** for a new project, or read existing.
7. **Three workflows** (section 3): `ci.yml`, `prerelease.yml`, `release.yml`. Add `pages.yml` if `/docs` exists.
8. **Distribution config** (e.g. `.goreleaser.yml` for Go). Always set explicit `repository.token` for any cross-repo publish.
9. **Branches:** create `develop` if it doesn't exist; do all work there. `main` stays untouched until the first release.
10. **First commit on `develop`:** scaffolding + initial code. Push.
11. **Verify CI green** on `develop`.
12. **Create `release/0.0.1`** from `develop`. Move `[Unreleased]` → `[0.0.1] - <today>` in CHANGELOG. Push.
13. **Verify prerelease workflow** publishes `v0.0.1-alpha.1` cleanly.
14. **Open PR** `release/0.0.1` → `main`. Wait for CI.
15. **Confirm with the user** before merging the PR (hard-to-reverse).
16. **Merge.** `release.yml` runs, tags `v0.0.1`, publishes everywhere.
17. **Verify** GitHub Release + tap + bucket + container registry + Pages all updated.
18. **Merge `main` back to `develop`** to close the cycle.
19. **Delete the release branch** (local + remote).
20. **Document** anything project-specific in the project's own `CLAUDE.md` or `README.md`, not here.

### Anti-patterns to refuse

- Pushing directly to `main`.
- Adding co-author trailers without explicit permission.
- Inventing dynamic-looking badges with hardcoded values.
- Adding "best practices" the user didn't ask for (extra error handling, unused abstractions, premature monitoring).
- Skipping the idempotence guard on the release workflow.
- Granting `permissions: write-all` instead of the minimum set.
- Treating prereleases (`-alpha`, `-beta`, `-rc`) as stable when computing `:latest` Docker tags or moving Homebrew formula heads.
