"""LabGene consultation RAG evaluation (spec: docs/superpowers/specs/2026-09-29-labgene-rag-eval-design.md).

Kept outside src/labgene on purpose: a frozen evaluation pins the src/labgene source-tree hash, and a new module there
would stop `resume` of that run (spec R7). It reads a finished harness run and never writes to it."""
