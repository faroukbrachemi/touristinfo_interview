import os
import re
import argparse
import ast
from datetime import date
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv() 

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data"
COLLECTION_NAME = "touristinfo"
TOP_K = 8
GROQ_MODEL = "llama-3.3-70b-versatile" 
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BEGIN_MARKER = re.compile(r"^### BEGIN (.+)$", re.MULTILINE)
FNAME_RE = re.compile(r"ti_export_(?P<place>[^_]+)_(?P<category>[a-z]+)_(?P<subcat>.+)\.txt$")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ingestion 
# ---------------------------------------------------------------------------


def parse_filename(path: Path) -> dict:
    m = FNAME_RE.match(path.name)
    if m:
        return {
            "place": m.group("place"),
            "category": m.group("category"),
            "subcategory": m.group("subcat"),
        }
    return {"place": "unknown", "category": "misc", "subcategory": path.stem}


def load_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta = parse_filename(path)
    matches = list(BEGIN_MARKER.finditer(raw))

    if not matches:
        # No entry markers -> likely a website/info page. Treat as one chunk.
        text = raw.strip()
        if not text:
            return []
        return [Chunk(f"{path.stem}-0", text, {**meta, "source": path.name, "entry_type": "page"})]

    header = raw[: matches[0].start()].strip()
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        entry_text = raw[start:end].strip()
        # Prepend the file header so category context travels with the entry
        # even if the entry text itself doesn't repeat those keywords.
        full_text = f"{header}\n\n{entry_text}" if header else entry_text
        name_match = re.search(r"\*\*\s*(.+?)\s*\*\*", entry_text)

        # Chroma rejects None metadata values, so only set keys that have one.
        metadata = {**meta, "source": path.name, "entry_type": m.group(1).strip()}
        if name_match:
            metadata["name"] = name_match.group(1)
        hours = parse_hours_blocks(entry_text)
        if hours:
            metadata["hours"] = hours

        chunks.append(Chunk(id=f"{path.stem}-{i}", text=full_text, metadata=metadata))
    return chunks


def load_data_dir(data_dir: str) -> list[Chunk]:
    p = Path(data_dir)
    if not p.exists():
        return []
    chunks = []
    for file_path in sorted(p.glob("*.txt")):
        chunks.extend(load_file(file_path))
    return chunks


def load_toy_documents() -> list[Chunk]:
    """Fallback/demo dataset from documents.py, used if data/ isn't present
    or to supplement it."""
    try:
        from documents import documents as toy_docs
    except ImportError:
        return []
    return [
        Chunk(id=d["id"], text=d["text"], metadata={"category": "demo", "source": "documents.py"})
        for d in toy_docs
    ]



HOURS_BLOCK_RE = re.compile(r"\{[^{}]*\}")
BLOCK_SEP, FIELD_SEP = ";", "|"
 
# The flag takes English abbreviations; German/Italian day names are accepted
# too, since a German-speaking user will reach for "sa"/"Samstag" not "sat".
DAY_ALIASES = {
    "mo": "mo", "mon": "mo", "monday": "mo", "montag": "mo", "lun": "mo", "lunedi": "mo",
    "tu": "tu", "tue": "tu", "tuesday": "tu", "di": "tu", "dienstag": "tu", "mar": "tu", "martedi": "tu",
    "we": "we", "wed": "we", "wednesday": "we", "mi": "we", "mittwoch": "we", "mer": "we", "mercoledi": "we",
    "th": "th", "thu": "th", "thursday": "th", "do": "th", "donnerstag": "th", "gio": "th", "giovedi": "th",
    "fr": "fr", "fri": "fr", "friday": "fr", "freitag": "fr", "ven": "fr", "venerdi": "fr",
    "sa": "sa", "sat": "sa", "saturday": "sa", "samstag": "sa", "sab": "sa", "sabato": "sa",
    "su": "su", "sun": "su", "sunday": "su", "so": "su", "sonntag": "su", "dom": "su", "domenica": "su",
}
 
 
def parse_hours_blocks(text: str) -> str:
    """Serialise an entry's hours blocks. Returns "" when there are none."""
    marker = re.search(r"\*\*Opening Hours[^*]*\*\*", text)
    region = text[marker.end():] if marker else text
 
    blocks = []
    for raw in HOURS_BLOCK_RE.findall(region):
        try:
            d = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(d, dict) or "hours" not in d or "days" not in d:
            continue
        days = ",".join(str(x).strip().lower() for x in d.get("days", []))
        hours = str(d["hours"]).replace(" ", "")
        start = str(d.get("period_start", "01.01")).replace(".", "")
        end = str(d.get("period_end", "31.12")).replace(".", "")
        if days and "-" in hours:
            blocks.append(FIELD_SEP.join([f"{start}-{end}", days, hours]))
    return BLOCK_SEP.join(blocks)
 
 
def _in_season(start_ddmm: str, end_ddmm: str, when: date) -> bool:
    """DDMM range, assumed to repeat yearly. Handles ranges crossing new year."""
    try:
        s_day, s_mon = int(start_ddmm[:2]), int(start_ddmm[2:])
        e_day, e_mon = int(end_ddmm[:2]), int(end_ddmm[2:])
    except (ValueError, IndexError):
        return True  # unparseable season -> don't exclude on it
    cur, start, end = (when.month, when.day), (s_mon, s_day), (e_mon, e_day)
    return start <= cur <= end if start <= end else (cur >= start or cur <= end)
 
 
def _in_hours(hours: str, minutes: int) -> bool:
    """'HH:MM-HH:MM'. Handles ranges crossing midnight."""
    try:
        open_s, close_s = hours.split("-", 1)
        oh, om = (int(x) for x in open_s.split(":"))
        ch, cm = (int(x) for x in close_s.split(":"))
    except ValueError:
        return False
    open_m, close_m = oh * 60 + om, ch * 60 + cm
    if open_m <= close_m:
        return open_m <= minutes <= close_m
    return minutes >= open_m or minutes <= close_m
 
 
def is_open(serialised: str, day: str, minutes: int, when: date | None = None) -> bool:
    """Open if ANY block matches.
 
    Entries with no hours data return True: absence of data is not evidence of
    being closed, and dropping them would silently hide places (Pizzeria Petra,
    most churches) from every time-filtered query.
    """
    if not serialised:
        return True
    when = when or date.today()
    for block in serialised.split(BLOCK_SEP):
        parts = block.split(FIELD_SEP)
        if len(parts) != 3:
            continue
        season, days, hours = parts
        start, _, end = season.partition("-")
        if _in_season(start, end, when) and day in days.split(",") and _in_hours(hours, minutes):
            return True
    return False
 
 
def parse_open_now(value: str) -> tuple[str, int]:
    """'sa:10:30' -> ('sa', 630)."""
    try:
        day_raw, hh, mm = value.split(":")
    except ValueError:
        raise ValueError("--open-now expects DAY:HH:MM, e.g. sa:10:30")
    day = DAY_ALIASES.get(day_raw.strip().lower())
    if day is None:
        raise ValueError(f"unknown day '{day_raw}' (use mo/tu/we/th/fr/sa/su)")
    try:
        minutes = int(hh) * 60 + int(mm)
    except ValueError:
        raise ValueError("--open-now expects DAY:HH:MM, e.g. sa:10:30")
    if not 0 <= minutes < 24 * 60:
        raise ValueError("time out of range")
    return day, minutes
 
 
def open_ids(collection, day: str, minutes: int) -> list[str]:
    """IDs of chunks whose place is open at the given day/time."""
    got = collection.get(include=["metadatas"])
    return [cid for cid, md in zip(got["ids"], got["metadatas"])
            if is_open(md.get("hours", ""), day, minutes)]
 
 

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class FastEmbedFunction(EmbeddingFunction):
    """Multilingual embeddings via ONNX runtime."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in self._model.embed(list(input))]

def build_collection(chunks: list[Chunk]):
    client = chromadb.Client()  

    # ef = embedding_functions.DefaultEmbeddingFunction()
    ef = FastEmbedFunction()

    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return collection


def retrieve(collection, query: str, k: int = TOP_K, only_ids: list[str] | None = None):
    """Returns (document, metadata) per hit."""
    if only_ids is not None:
        if not only_ids:
            return []
        # Restrict the vector search to a pre-computed id set rather than
        # filtering afterwards, so closed places don't consume top-k slots.
        results = collection.query(query_texts=[query], n_results=min(k, len(only_ids)),
                                   ids=only_ids)
    else:
        results = collection.query(query_texts=[query], n_results=k)
    return list(zip(results["documents"][0], results["metadatas"][0]))

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_prompt(query: str, retrieved: list[tuple[str, dict]]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[Source: {m.get('source', 'unknown')} | {m.get('name') or m.get('category')}]\n{doc}"
        for doc, m in retrieved
    )
    system = (
        "You are a South Tyrol tourism assistant. Answer ONLY using the context "
        "provided below. If the context doesn't contain the answer, say you don't "
        "have that information — never make anything up. Keep answers short and "
        "mention which place(s) you used."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def generate(messages: list[dict]) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content


def answer(collection, query: str, k: int = TOP_K, only_ids: list[str] | None = None) -> str:
    retrieved = retrieve(collection, query, k=k, only_ids=only_ids)
    if not retrieved:
        print(f"\nQ: {query}\nA: Nothing in the data is open at that time.")
        return ""
    reply = generate(build_prompt(query, retrieved))
    print(f"\nQ: {query}\nA: {reply}")
    print("Sources:", [m.get("name") or m.get("source") for _, m in retrieved])
    return reply


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--open-now", type=str, default=None,
                        help="filter to places open at DAY:HH:MM, e.g. sa:10:30")
    args = parser.parse_args()
 
    chunks = load_data_dir(DATA_DIR) or load_toy_documents()
    if not chunks:
        raise SystemExit(f"No documents found in '{DATA_DIR}/' or documents.py")
    print(f"Loaded {len(chunks)} chunks")
 
    collection = build_collection(chunks)
 
    allowed = None
    if args.open_now:
        day, minutes = parse_open_now(args.open_now)
        allowed = open_ids(collection, day, minutes)
        print(f"open at {day} {minutes // 60:02d}:{minutes % 60:02d}: "
              f"{len(allowed)}/{collection.count()} chunks")
 
    if args.query:
        answer(collection, args.query, k=args.k, only_ids=allowed)
 
    else:
        for q in [

        "Dove posso comprare del pane fresco?",

        "Kann ich meinen Hund ins Restaurant mitnehmen?",

        "Wie viel kostet ein Skipass?",

        "Welche Kirche liegt am höchsten?",

        "Ho un'intolleranza al glutine, dove posso mangiare?",

        "Elenca tutti i bed and breakfast",
        ]:
            answer(collection, q, k=args.k, only_ids=allowed)
            


if __name__ == "__main__":
    main()