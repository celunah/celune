# Documentation standard

This page defines the format for Celune's technical documentation. It is for
contributors adding or updating a page under `docs/`; `AGENTS.md` makes these
rules part of the repository workflow.

## Page structure

Every page follows this outline, adapting section names to its subject:

```markdown
# Subject

Explain who the page is for and what it covers in one short paragraph.

## Overview or prerequisites

State the model, boundary, assumptions, or setup the reader needs.

## Task, contract, or reference sections

Explain the behavior in the order the reader needs it. Use examples beside
the call, option, file field, or workflow they demonstrate.

## Verification, errors, or troubleshooting

Describe expected results and the most relevant failure or compatibility
cases when the subject can vary or fail.

## See also

Link to the canonical related pages when another documented boundary is the
next useful step.
```

The final sections are conditional: a short glossary or gallery does not need
artificial troubleshooting content, while an installation or API page should
not omit it.

## Writing rules

- Use sentence case for headings. Keep acronyms and names such as `CEDTS`,
  `CEVOICE`, `CECHAR`, `Celune`, and `ReadTheDocs` unchanged.
- Start with the reader's goal. Prefer direct instructions and concrete
  expected results over project history or promotional language.
- Use numbered lists for ordered procedures and bullets for unordered facts.
  Use tables for stable comparisons, options, fields, and return values.
- Mark every fenced code block with its language (`python`, `yaml`, `bash`,
  `powershell`, `http`, `json`, or `text` as appropriate). Keep examples
  runnable or clearly label them as illustrative.
- Document public calls with their signature or call shape, purpose,
  arguments, return value, exceptions, side effects, and a usage example.
- Document formats and protocols with their version, layout, invariants,
  compatibility rules, and failure behavior. Link to the canonical format
  page instead of copying a second competing definition.
- Use repository paths and canonical commands. Do not invent placeholder
  commands, options, endpoints, or behavior.
- Link related pages with relative Markdown links and keep every page reachable
  from `mkdocs.yml` navigation.
- Technical pages describe implemented behavior. Do not treat
  `resources/about/about-celune.md` as a factual technical source.

## Local validation

Run MarkdownLint and the strict site build before committing documentation:

```bash
npx --yes markdownlint-cli2 "docs/**/*.md"
uv run --with-requirements docs/requirements.txt mkdocs build --strict
```

The current MarkdownLint CLI requires Node.js 22 or newer. The CI action runs
with Node.js 24; use an equivalent supported Node.js runtime when running the
command locally.

The CI workflow runs the same MarkdownLint configuration through the official
`markdownlint-cli2` action before building the MkDocs site.

## Review checklist

Before committing documentation, confirm that:

- the page has an H1 and a purpose paragraph;
- headings use sentence case and sections follow the reader's workflow;
- commands, paths, calls, fields, and examples match the current source;
- failure, compatibility, or verification behavior is documented where needed;
- links and navigation resolve;
- `uv run --with-requirements docs/requirements.txt mkdocs build --strict`
  succeeds; and
- `git diff --check` reports no whitespace errors.
