"""NadirClaw Python SDK — lightweight client for the NadirClaw LLM router.

Usage::

    from nadirclaw_sdk import NadirClient

    client = NadirClient()  # auto-detects localhost:8856
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}],
    )
    print(response.choices[0].message.content)
"""

from nadirclaw_sdk.client import NadirClient  # noqa: F401

__version__ = "0.1.0"
__all__ = ["NadirClient"]
