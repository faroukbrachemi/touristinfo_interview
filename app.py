import os
import re
import argparse
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
        chunks.append(
            Chunk(
                id=f"{path.stem}-{i}",
                text=full_text,
                metadata={**meta, "source": path.name, "entry_type": m.group(1).strip(),
                          "name": name_match.group(1) if name_match else None},
            )
        )
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


def retrieve(collection, query: str, k: int = TOP_K):
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


def answer(collection, query: str, k: int = TOP_K) -> str:
    retrieved = retrieve(collection, query, k=k)      
    reply = generate(build_prompt(query, retrieved))
    sources = [m.get("name") or m.get("source") for _, m in retrieved]
    print(f"\nQ: {query}\nA: {reply}\nSources: {sources}")
    return reply


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    chunks = load_data_dir(DATA_DIR) or load_toy_documents()
    if not chunks:
        raise SystemExit(f"No documents found in '{DATA_DIR}/' or documents.py")
    print(f"Loaded {len(chunks)} chunks")

    collection = build_collection(chunks)

    if args.query:
        answer(collection, args.query, k=args.k)
    else:
        for q in [

        "Dove posso comprare del pane fresco?",

        "Kann ich meinen Hund ins Restaurant mitnehmen?",

        "Wie viel kostet ein Skipass?",

        "Welche Kirche liegt am höchsten?",

        "Ho un'intolleranza al glutine, dove posso mangiare?",

        "Elenca tutti i bed and breakfast",
        ]:
            answer(collection, q, k=args.k)
            


if __name__ == "__main__":
    main()