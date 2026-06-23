# Contributing to tree-sitter-iec61131-3-st

Thanks for your interest! Contributions of any size are welcome — bug reports, fixes, new features, doc improvements.

For dialect-grammar contributions (TwinCAT, Codesys, B&R, Siemens, Rockwell), see [EXTENDING.md](EXTENDING.md) — those live in separate repos and extend this base.

## Branch flow

```
develop  →  release/<version>  →  main
```

- All feature work, bug fixes, and refactors target `develop`.
- `main` is reserved for stable releases — never PR directly to it.
- Release stabilization happens on `release/<version>` branches cut from `develop`.

See the [README's "Branching & releases"](README.md) section (or [`BLUEPRINT.md`](BLUEPRINT.md)) for the full versioning, tagging, and CI flow.

## Pull requests

- One logical change per PR; keep diffs reviewable.
- Run `tree-sitter generate` and commit the regenerated `src/parser.c` along with any `grammar.js` change. CI verifies these stay in sync.
- Run `tree-sitter test` (corpus tests) before opening the PR.
- Make sure every file in `examples/` still parses without errors.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Don't bump `VERSION` in feature PRs — that happens on the release branch.
- The `ci` workflow must pass before a PR can merge.

## Commit messages

Conventional-commits style is appreciated:

```
feat: add <thing>
fix: handle <edge case>
docs: clarify <section>
chore(deps): bump <dep> from X to Y
ci: <workflow change>
```

The release-branch workflow looks for `[beta]` / `[rc]` keywords in commit messages to choose pre-release stages — keep those out of normal commits.

## Sign your work (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org) v1.1 as a lightweight contribution attestation. It is **not** a CLA — no form, no rights transfer. By signing off your commits, you are certifying that you wrote the patch (or otherwise have the right to submit it under the project's license — MIT). The full text is at the link above.

Every commit must carry a `Signed-off-by` trailer matching the commit author's real name and email:

```
Signed-off-by: Jane Doe <jane@example.com>
```

Git adds this trailer automatically when you commit with the `-s` flag:

```
git commit -s -m "feat: add range subtype"
```

If you forget on the most recent commit, fix it with `git commit --amend --signoff`. For older commits in your branch, use `git rebase --signoff <base>`.

Anonymous or pseudonymous sign-offs are not accepted.

## Reporting bugs

Use the bug report template under "New Issue". A minimal reproducing example is gold.

## Asking questions / proposing ideas

Use [Discussions](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/discussions) for open-ended questions and ideas. Reserve issues for actionable bugs and concrete feature requests.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
