"""Redraw docs/images/agent_structure_graph.png from the graph in core/graph.py.

Run from anywhere after changing the graph's nodes or edges:

    python utils/draw_agent_graph.py

The graph is built the way main.py builds it, but without tools, checkpointer or model
calls: the picture shows only nodes and edges, which do not depend on them. It is
rendered by mermaid.ink, so this needs an internet connection; only the node names and
edges are sent.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Same as main.py: embeddings use PyTorch only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")
# The model clients are created but never called, so placeholders do when .env lacks them.
for name, placeholder in (
    ("AZURE_AI_ENDPOINT", "https://placeholder.invalid/"),
    ("AZURE_AI_CREDENTIAL", "placeholder"),
):
    if not os.environ.get(name):
        os.environ[name] = placeholder

from core.graph import build_graph

OUTPUT_PATH = BASE_DIR / "docs" / "images" / "agent_structure_graph.png"
NO_TOOLS = {"communication": [], "planning": [], "content": [], "supervisor": []}


def main():
    graph = build_graph(NO_TOOLS, checkpointer=None)
    try:
        png = graph.get_graph().draw_mermaid_png()
    except Exception as e:
        sys.exit(f"Could not render the picture (mermaid.ink needs an internet connection): {e}")
    OUTPUT_PATH.write_bytes(png)
    print(f"Saved {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
