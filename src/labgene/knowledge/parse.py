"""Structured-text parser and chunker (T05, spec §13.2.2-3). Real PDF parser selection is T08.

Input: markdown-like text with optional YAML front matter (doc metadata). Conventions:
  <!-- page: N -->                   page marker; `<!-- page: N ocr-suspect -->` marks the page uncertain
  # .. ######                        heading hierarchy -> section path
  | a | b |, |---|, | [unit] |       pipe table; a row of [bracketed] cells right after the separator = unit row
  $$ ... $$                          display formula
  1. / 2)                            procedure steps
  [ocr?]                             inline OCR-suspect marker -> block uncertain
Chunk text is a verbatim slice of the source (formulas, inequalities, ranges, subscripts untouched); split tables
repeat the verbatim header/separator/unit lines. Uncertain transcriptions are flagged, never marked verified.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

PARSER_VERSION = "md-struct-v1"
OCR_MARK = "[ocr?]"
OCR_UNCERTAINTY = "ocr_suspect: transcription not verified; numbers in this excerpt may be misread"
_PAGE = re.compile(r"<!--\s*page:\s*(\d+)(\s+ocr-suspect)?\s*-->")
_HEADING = re.compile(r"(#{1,6})\s+(.+?)\s*#*")
_STEP = re.compile(r"\d+[.)]\s")
_SEP = re.compile(r"\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*")


@dataclass
class Block:
    kind: str                       # paragraph | table | formula | procedure
    start: int                      # char span in Document.text
    end: int
    page: int | None
    section: tuple[str, ...]
    uncertain: bool


@dataclass
class Document:
    doc_id: str
    meta: dict
    text: str                       # full original text; spans index into it
    body: int                       # offset after front matter
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Chunk:
    text: str
    header: str                     # generated from doc metadata + section path + page
    locator: dict                   # doc_id, page, section, span (+ table rows / parent span)
    kind: str
    uncertain: bool

    @property
    def uncertainty(self) -> list[str]:
        return [OCR_UNCERTAINTY] if self.uncertain else []


def parse_document(text: str, doc_id: str) -> Document:
    meta, body = {}, 0
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            meta = yaml.safe_load(text[4:end]) or {}
            nl = text.find("\n", end + 4)
            body = len(text) if nl == -1 else nl + 1
    doc = Document(doc_id, meta, text, body)
    page, page_ocr, section = None, False, ()
    cur: list | None = None          # [kind, start, end]
    in_formula = False

    def close() -> None:
        nonlocal cur
        if cur:
            k, s, e = cur
            doc.blocks.append(Block(k, s, e, page, section, page_ocr or OCR_MARK in text[s:e]))
            cur = None

    pos = body
    for line in text[body:].splitlines(keepends=True):
        start, pos = pos, pos + len(line)
        end = start + len(line.rstrip("\r\n"))
        s = line.strip()
        if in_formula:
            cur[2] = end
            if s.endswith("$$"):
                in_formula = False
                close()
            continue
        if m := _PAGE.fullmatch(s):
            close()
            page, page_ocr = int(m[1]), bool(m[2])
            continue
        if not s:
            close()
            continue
        if m := _HEADING.fullmatch(s):
            close()
            section = section[:len(m[1]) - 1] + (m[2],)
            continue
        kind = ("table" if s.startswith("|") else "formula" if s.startswith("$$")
                else "procedure" if _STEP.match(s) else "paragraph")
        if cur and cur[0] != kind:
            close()
        if cur:
            cur[2] = end
        else:
            cur = [kind, start, end]
        if kind == "formula" and (s == "$$" or not s.endswith("$$")):
            in_formula = True
    close()
    return doc


def _is_unit_row(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return any(cells) and all(not c or (c.startswith("[") and c.endswith("]")) for c in cells)


def chunk_document(doc: Document, max_chars: int = 1200, table_rows: int = 12) -> list[Chunk]:
    """Chunks by headings/paragraphs/tables/formulas/procedures. Consecutive paragraphs of one section and page
    merge up to max_chars. Tables over table_rows split into row groups that repeat header + unit row and carry
    the parent table span. ponytail: an over-long single paragraph/procedure stays one chunk."""
    title = doc.meta.get("title") or doc.doc_id
    out: list[Chunk] = []

    def header(b: Block) -> str:
        return " | ".join(x for x in (title, " > ".join(b.section), f"p. {b.page}" if b.page else "") if x)

    def loc(b: Block, s: int, e: int, **kw) -> dict:
        return {"doc_id": doc.doc_id, "page": b.page, "section": list(b.section), "span": [s, e], "block": b.kind, **kw}

    group: list[Block] = []

    def flush() -> None:
        if group:
            s, e = group[0].start, group[-1].end
            out.append(Chunk(doc.text[s:e], header(group[0]), loc(group[0], s, e), "paragraph",
                             any(g.uncertain for g in group)))
            group.clear()

    for b in doc.blocks:
        if b.kind == "paragraph" and group and (group[0].section, group[0].page) == (b.section, b.page) \
                and b.end - group[0].start <= max_chars:
            group.append(b)
            continue
        flush()
        if b.kind == "paragraph":
            group.append(b)
        elif b.kind == "table":
            out.extend(_table_chunks(doc, b, header(b), loc, table_rows))
        else:
            out.append(Chunk(doc.text[b.start:b.end], header(b), loc(b, b.start, b.end), b.kind, b.uncertain))
    flush()
    return out


def _table_chunks(doc: Document, b: Block, head: str, loc, max_rows: int) -> list[Chunk]:
    lines = doc.text[b.start:b.end].split("\n")
    offs, p = [], b.start
    for ln in lines:
        offs.append(p)
        p += len(ln) + 1
    n_head = 2 if len(lines) > 1 and _SEP.fullmatch(lines[1]) else 1
    units = lines[n_head] if len(lines) > n_head and _is_unit_row(lines[n_head]) else None
    n_head += units is not None
    rows = lines[n_head:]
    table = {"header": lines[0], "units": units, "parent_span": [b.start, b.end]}
    if len(rows) <= max_rows:
        return [Chunk(doc.text[b.start:b.end], head, loc(b, b.start, b.end, table={**table, "rows": [0, len(rows)]}),
                      "table", b.uncertain)]
    out = []
    for i in range(0, len(rows), max_rows):
        part = rows[i:i + max_rows]
        s, e = offs[n_head + i], offs[n_head + i + len(part) - 1] + len(part[-1])
        out.append(Chunk("\n".join(lines[:n_head] + part), head,
                         loc(b, s, e, table={**table, "rows": [i, i + len(part)]}), "table", b.uncertain))
    return out
