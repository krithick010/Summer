"""
plugins/base.py
---------------
Abstract Base Class that every plugin must inherit from.

Design contract
~~~~~~~~~~~~~~~
* ``name``              – machine-friendly identifier used by the LLM tool router
                          (e.g. ``"gmail_send"``)
* ``description``       – plain-English description injected into the LLM system
                          prompt so the model understands when to call this tool
* ``parameters_schema`` – JSON-Schema dict describing the keyword arguments
                          ``execute()`` accepts; used to validate LLM tool-calls
* ``requires_approval`` – when ``True`` the orchestrator PAUSES and asks the
                          Telegram user to approve before ``execute()`` is called
* ``execute(**kwargs)`` – the actual action; always returns a ``PluginResult``
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PluginResult:
    """Standardised return value from every plugin.

    Attributes
    ----------
    success:
        ``True`` if the action completed without errors.
    output:
        Human-readable string the bot will forward to the Telegram user.
    data:
        Optional structured payload for further processing by the orchestrator.
    error:
        Error message when ``success=False``.
    """

    success: bool
    output: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return self.output if self.success else f"[ERROR] {self.error}"


# ---------------------------------------------------------------------------
# Abstract base plugin
# ---------------------------------------------------------------------------


class PluginBase(abc.ABC):
    """Every plugin must inherit from this class and implement all abstract
    members.  The orchestrator will refuse to register a class that does not
    satisfy the interface.
    """

    # ------------------------------------------------------------------
    # Class-level attributes — override in the subclass, not in __init__
    # ------------------------------------------------------------------

    #: Machine-friendly unique identifier (snake_case)
    name: str = ""

    #: One-to-two sentence description injected into the LLM tool-prompt
    description: str = ""

    #: Optional short trigger rule used in the generated system prompt TOOLS section
    usage_hint: str = ""

    #: JSON-Schema object describing accepted keyword arguments.
    #: Follows the JSON Schema ``"object"`` type with ``"properties"`` and
    #: ``"required"`` keys so the LLM knows exactly what to pass.
    #:
    #: Example::
    #:
    #:     parameters_schema = {
    #:         "type": "object",
    #:         "properties": {
    #:             "to":      {"type": "string", "description": "Recipient e-mail"},
    #:             "subject": {"type": "string", "description": "Email subject line"},
    #:             "body":    {"type": "string", "description": "Plain-text body"},
    #:         },
    #:         "required": ["to", "subject", "body"],
    #:     }
    parameters_schema: dict[str, Any] = dataclasses.field(default_factory=dict)

    #: If ``True`` the orchestrator will request Telegram user approval before
    #: calling ``execute()``.  Set to ``True`` for any destructive or
    #: privacy-sensitive action (e.g. sending email, deleting files).
    requires_approval: bool = False

    #: Human-readable reason when the plugin is unavailable in the environment
    unavailable_reason: str = ""

    def is_available(self) -> bool:
        """Return True if the plugin is ready to use in the current environment."""
        return True

    # ------------------------------------------------------------------
    # Validation on subclass creation
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip validation for intermediate abstract classes
        if abc.ABC in cls.__bases__:
            return
        _required_attrs = ("name", "description", "parameters_schema")
        for attr in _required_attrs:
            value = getattr(cls, attr, None)
            if not value:
                raise TypeError(
                    f"Plugin '{cls.__name__}' must define a non-empty '{attr}' "
                    f"class attribute."
                )
        if not cls.name.replace("_", "").isalnum():
            raise TypeError(
                f"Plugin '{cls.__name__}'.name must be snake_case alphanumeric, "
                f"got: '{cls.name}'"
            )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> PluginResult:
        """Execute the plugin action.

        Parameters
        ----------
        **kwargs:
            Arguments parsed from the LLM tool-call, validated against
            ``parameters_schema`` by the orchestrator before this method
            is invoked.

        Returns
        -------
        PluginResult
            Standardised result container.
        """

    async def prepare_preview(self, kwargs: dict[str, Any]) -> str | None:
        """Optional hook to build a rich human-readable HITL preview before approval."""
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_tool_spec(self) -> dict[str, Any]:
        """Return the tool specification injected into the LLM system prompt.

        The format mirrors OpenAI-style function-calling so it is easy to
        switch LLM backends in the future.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    def __repr__(self) -> str:  # pragma: no cover
        flag = " [HITL]" if self.requires_approval else ""
        return f"<Plugin '{self.name}'{flag}>"

