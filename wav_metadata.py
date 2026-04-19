"""
RIFF LIST/INFO chunks for exported WAV files (Windows-friendly tags + long comment).

Used by Voice Engine and pslicer so trims carry transcript, attribution, and program name.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping

PROGRAM_NAME = "Voice Engine"
PROGRAM_URL = "https://github.com/gmater/voice-engine"
PROGRAM_AUTHOR = "gmater"


def _info_subchunk(tag: bytes, text: str, *, max_payload: int = 65000) -> bytes:
    if len(tag) != 4:
        raise ValueError("RIFF INFO subchunk tag must be 4 bytes")
    raw = (text or "").encode("utf-8", errors="replace")
    if len(raw) > max_payload:
        raw = raw[: max_payload - 20] + b"\n...(truncated)"
    sz = len(raw)
    pad = b"\x00" if sz % 2 else b""
    return tag + struct.pack("<I", sz) + raw + pad


def _build_list_info_chunk(fields: Mapping[str, str]) -> bytes:
    """LIST chunk with list type INFO and IART / ISFT / INAM / ICMT-style subkeys (4 ASCII letters each)."""
    body = b"INFO"
    for key, val in fields.items():
        k = key.strip().encode("ascii", errors="ignore")[:4]
        if len(k) < 4:
            k = k + b" " * (4 - len(k))
        body += _info_subchunk(k, str(val))
    size = len(body)
    return b"LIST" + struct.pack("<I", size) + body


def embed_list_info_in_wav(wav_path: str, fields: Mapping[str, str]) -> None:
    """
    Insert or replace a RIFF LIST/INFO block in a WAVE file (before ``data``).

    Skips any existing LIST chunk(s). Updates the top-level RIFF size field.
    """
    list_chunk = _build_list_info_chunk(fields)
    path = str(Path(wav_path).resolve())
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return

    out = bytearray(raw[0:12])
    pos = 12
    inserted = False
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        sz = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        nxt = pos + 8 + sz + (sz % 2)
        if cid == b"LIST":
            pos = nxt
            continue
        if cid == b"data" and not inserted:
            out.extend(list_chunk)
            inserted = True
        out.extend(raw[pos:nxt])
        pos = nxt
    if not inserted:
        out.extend(list_chunk)
    struct.pack_into("<I", out, 4, len(out) - 8)
    with open(path, "wb") as f:
        f.write(out)


def embed_voice_engine_wav_metadata(
    wav_path: str,
    *,
    source_audio_basename: str,
    speaker: str,
    transcript: str,
    trim_export_start_ms: int,
    trim_export_end_ms: int,
) -> None:
    """Standard Voice Engine / pslicer metadata for one exported clip."""
    sp = (speaker or "").strip() or "(none)"
    tx = (transcript or "").strip() or "(none)"
    icmt_lines = [
        f"Source file: {source_audio_basename}",
        f"Export window (ms in source file): {trim_export_start_ms} – {trim_export_end_ms}",
        f"Speaker label: {sp}",
        f"Transcript: {tx}",
        "",
        f"Created with {PROGRAM_NAME} by {PROGRAM_AUTHOR}.",
        PROGRAM_URL,
    ]
    icmt = "\n".join(icmt_lines)
    title = (tx[:240] + ("…" if len(tx) > 240 else "")) if tx != "(none)" else source_audio_basename[:240]
    fields = {
        "INAM": title,
        "IART": PROGRAM_AUTHOR,
        "ISFT": f"{PROGRAM_NAME} — {PROGRAM_URL}",
        "ICMT": icmt[:62000],
    }
    embed_list_info_in_wav(wav_path, fields)
