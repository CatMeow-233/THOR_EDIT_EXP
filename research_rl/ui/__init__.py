"""Local channel search + edit tool for :mod:`research_rl.capability` (``CAPABILITY.md`` 1.13).

``python -m research_rl.ui --tiny-demo`` (or ``--model-dir DIR --store DIR``) starts a stdlib
HTTP server on 127.0.0.1 with a static browser front end. Every API request needs the session
token printed at start; one model worker thread executes all weight access; edits are declarative
:class:`~research_rl.capability.edits.EditRecord` objects committed through the capability gate
(or with a recorded override); the AUDIT split is not reachable; weights load from safetensors
only. Numbers shown come from the model loaded on the machine running the server; the tiny demo
model's numbers say nothing about real large models.
"""

from __future__ import annotations

from research_rl.ui.server import ServerOptions, UIApp, UIServer, create_app, main

__all__ = ["ServerOptions", "UIApp", "UIServer", "create_app", "main"]
