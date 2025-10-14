import re
from dataclasses import dataclass

@dataclass
class Utterance:
    call_id: str
    section: str      # 'prepared' | 'qa'
    turn_type: str    # 'question' | 'answer' | 'remark'
    speaker_role: str # 'CEO','CFO','IR','Analyst','Operator','Other'
    speaker: str
    text: str
    order: int

_SPEAKER = re.compile(r"^\s*([A-Z][A-Za-z .'-]+):\s*(.*)$")
_QA_HDR = re.compile(r"^\s*(Q&A|Questions?\s*&\s*Answers?)\s*$", re.I)

def _infer_role(name: str) -> str:
    n = name.lower()
    if 'operator' in n: return 'Operator'
    if 'analyst' in n or 'research' in n: return 'Analyst'
    if 'chief executive' in n or 'ceo' in n: return 'CEO'
    if 'chief financial' in n or 'cfo' in n: return 'CFO'
    if 'investor relations' in n or 'ir' in n: return 'IR'
    return 'Other'

def parse_transcript(call_id: str, raw: str):
    section = 'prepared'
    out, buf, speaker, role, order = [], [], None, None, 0

    def flush(turn='remark'):
        nonlocal order, buf, speaker, role
        if buf:
            out.append(Utterance(call_id, section, turn, role or 'Other', speaker or 'Unknown',
                                 " ".join(buf).strip(), order))
            order += 1; buf = []

    for line in raw.splitlines():
        if _QA_HDR.match(line):
            flush()
            section = 'qa'; continue
        m = _SPEAKER.match(line)
        if m:
            flush('remark' if section=='prepared' else None)
            speaker = m.group(1).strip()
            role = _infer_role(speaker)
            buf = [m.group(2).strip()]
        elif line.strip():
            buf.append(line.strip())
    flush('remark' if section=='prepared' else None)

    for u in out:
        if u.section=='qa':
            if u.speaker_role=='Analyst': u.turn_type='question'
            elif u.speaker_role not in {'Operator','Analyst'}: u.turn_type='answer'
            else: u.turn_type='remark'
    return out
