"""Rechnet die Messreihen neu und schreibt data/messreihen.json (rund 2 Minuten). Aufruf aus dem Demo-Ordner: python tools/make_messreihen.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fsp_tables as T  # noqa: E402


def main() -> None:
    data = T.build(progress=lambda *a: print("fertig:", *a, flush=True))
    T.DATA.parent.mkdir(exist_ok=True)
    T.DATA.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"geschrieben: {T.DATA} (Hash {data['meta']['source_hash'][:12]}, {data['meta']['build_seconds']} s)")


if __name__ == "__main__":
    main()
