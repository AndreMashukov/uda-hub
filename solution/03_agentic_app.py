"""Run UDA-Hub from the command line.

    cd solution
    python setup_external_db.py
    python setup_core_db.py
    python 03_agentic_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from setup_core_db import setup_udahub_db  # noqa: E402
from setup_external_db import setup_cultpass_db  # noqa: E402
from agentic.tools.paths import cultpass_db, udahub_db  # noqa: E402
from agentic.workflow import orchestrator  # noqa: E402
from utils import chat_interface  # noqa: E402


def ensure_databases() -> None:
    if not cultpass_db().exists():
        print("CultPass DB missing; running setup_external_db.")
        setup_cultpass_db()
    if not udahub_db().exists():
        print("UDA-Hub DB missing; running setup_core_db.")
        setup_udahub_db()


def main() -> None:
    ensure_databases()
    thread_id = os.environ.get("UDA_THREAD_ID", "demo-ticket")
    chat_interface(orchestrator, thread_id)


if __name__ == "__main__":
    main()
