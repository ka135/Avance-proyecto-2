"""Chunking en dos niveles: (1) corte por secciones, (2) división recursiva con solapamiento."""
import re
from dataclasses import dataclass
from typing import List, Optional

from .ingest import Document

_HEADING = re.compile(r"^\s*(?:#{1,4}\s+(?P<md>.+?)|(?P<up>(?:SECCI[ÓO]N|CAP[ÍI]TULO)\s+.+?))\s*$", re.I)


@dataclass
class Chunk:
    text: str
    source: str
    section: str
    page: Optional[int]
    chunk_id: str


def split_sections(text: str):
    """Divide el texto en (titulo, cuerpo). Un fragmento nunca cruza el límite de sección."""
    sections, title, buf = [], "", []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            if "".join(buf).strip():
                sections.append((title, "\n".join(buf)))
            title, buf = (m.group("md") or m.group("up")).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((title, "\n".join(buf)))
    return sections


def _atoms(text: str, size: int):
    """Unidades mínimas: líneas -> oraciones -> palabras (solo si no caben)."""
    atoms = []
    for line in re.findall(r"[^\n]*\n|[^\n]+\Z", text):
        if len(line) <= size:
            atoms.append(line)
            continue
        for sent in re.findall(r".+?(?:[.!?;:]\s+|\Z)", line, flags=re.S):
            if len(sent) <= size:
                atoms.append(sent)
            else:
                atoms.extend(re.findall(r"\S+\s*", sent))
    return atoms


def recursive_split(text: str, size: int, overlap: int) -> List[str]:
    chunks, cur, cur_len = [], [], 0
    for a in _atoms(text, size):
        if cur and cur_len + len(a) > size:
            chunks.append("".join(cur).strip())
            tail, l = [], 0
            for x in reversed(cur):
                if l + len(x) > overlap or l + len(x) + len(a) > size:
                    break
                tail.insert(0, x)
                l += len(x)
            cur, cur_len = tail, l
        cur.append(a)
        cur_len += len(a)
    last = "".join(cur).strip()
    if last:
        chunks.append(last)
    return [c for c in chunks if c]


def chunk_document(doc: Document, size: int, overlap: int) -> List[Chunk]:
    out = []
    for title, body in split_sections(doc.text):
        for piece in recursive_split(body, size, overlap):
            out.append(Chunk(piece, doc.source, title, doc.page, ""))
    return out


def chunk_corpus(docs: List[Document], size: int, overlap: int) -> List[Chunk]:
    counters, result = {}, []
    for doc in docs:
        for c in chunk_document(doc, size, overlap):
            n = counters.get(doc.source, 0) + 1
            counters[doc.source] = n
            c.chunk_id = f"{doc.source}#{n:03d}"
            result.append(c)
    return result
