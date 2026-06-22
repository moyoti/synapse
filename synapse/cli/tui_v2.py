"""
Full-screen TUI v2 — prompt_toolkit Application with self-contained scrollable
chat viewport, OpenCode-style independent terminal interface.

Layout:
  ┌─ Header: model / mode / memory stats ──────────────────────────┐
  │                                                                │
  │  Chat area (scrollable) — messages rendered via rich → ANSI    │
  │  • ↑↓ / PgUp/PgDn / mouse wheel to scroll                      │
  │  • Streaming tokens update in-place                             │
  │                                                                │
  ├────────────────────────────────────────────────────────────────┤
  │  › _                                          /help /mode ...  │
  └────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

from prompt_toolkit import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import has_focus
from prompt_toolkit.formatted_text import ANSI, HTML, FormattedText
from prompt_toolkit.history import FileHistory as PTFileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style as PTStyle

from rich.console import Console as RichConsole
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style as RichStyle
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text as RichText

from synapse import __version__
from synapse.cli.helpers import load_synapse_config, safe_ask
from synapse.config.loader import find_required_keys, save_config
from synapse.config.schema import SynapseConfig
from synapse.core.router import detect_mode
from synapse.models.registry import get_provider_for_model

# ── Constants ────────────────────────────────────────────────────────

_HISTORY_FILE = Path.home() / ".synapse" / ".chat_history"

_STYLE = PTStyle.from_dict({
    # Main background
    "": "#e6edf3",
    "window": "bg:#0d1117",
    # Header
    "header": "bg:#0d419d #ffffff bold",
    "header.dim": "bg:#0d419d #8b949e",
    # Chat
    "chat": "bg:#0d1117",
    "user": "#79c0ff bold",
    "ai": "#7ee787 bold",
    "system": "#8b949e italic",
    # Separator
    "separator": "#30363d",
    # Input area
    "input": "bg:#161b22",
    "input.prompt": "#58a6ff bold",
    "input.text": "#e6edf3",
    "input.text-area": "bg:#161b22 #e6edf3",
    "input.text-area.prompt": "#58a6ff bold",
    "hints": "#484f58",
    # Dropdown
    "completion-menu": "bg:#161b22 #e6edf3",
    "completion-menu.completion": "bg:#161b22 #58a6ff",
    "completion-menu.completion.current": "bg:#1f6feb #ffffff bold",
    "completion-menu.completion.meta": "bg:#161b22 #8b949e italic",
    "scrollbar.background": "bg:#30363d",
    "scrollbar.button": "bg:#58a6ff",
})

# ── Slash command list ───────────────────────────────────────────────

_COMMANDS_WITH_META = [
    ("/help", "Show all commands"),
    ("/clear", "Clear chat history"),
    ("/compact", "Compress conversation context"),
    ("/quit", "Exit chat"),
    ("/exit", "Exit chat"),
    ("/q", "Exit chat (shortcut)"),
    ("/mode single", "Single model mode"),
    ("/mode orchestrate", "Multi-agent orchestration"),
    ("/mode debate", "Multi-perspective debate"),
    ("/mode pipeline", "Sequential pipeline"),
    ("/mode auto", "Auto-detect best mode"),
    ("/model", "List or switch models"),
    ("/role", "List or switch roles"),
    ("/roles", "Manage role → model assignments"),
    ("/roles add", "Create a custom role"),
    ("/roles reassign", "Change a role's model"),
    ("/setup", "Add a new model (guided)"),
    ("/check", "Check API key status"),
    ("/config", "Show config summary"),
    ("/remember", "Save a fact to memory"),
    ("/recall", "Search memories"),
    ("/facts", "Show stored facts"),
    ("/stats", "Memory statistics"),
    ("/session save", "Save this conversation"),
    ("/session list", "List saved sessions"),
    ("/session load", "Resume a saved session"),
]

_COMMAND_DESCRIPTIONS = dict(_COMMANDS_WITH_META)


class SlashCompleter(Completer):
    """Hierarchical slash command completer with descriptions.

    - Typing / → shows all top-level commands
    - Typing /roles → shows /roles, /roles add, /roles reassign
    - Typing /roles → shows add, reassign (just the sub-parts)
    - Typing /session → shows save, list
    """

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        # Split into parts: e.g., "/roles add" → ["/roles", "add"]
        parts = text.split()
        base = parts[0] if parts else text

        # Detect if user typed a trailing space (wants sub-options)
        wants_sub = text.endswith(" ") and len(parts) >= 1

        if not wants_sub and len(parts) == 1:
            # User is still typing the base command — show matching commands
            for cmd, desc in _COMMANDS_WITH_META:
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )
            return

        # User has typed a space after the base command — show sub-options
        rest = text[len(base):].lstrip()

        for cmd, desc in _COMMANDS_WITH_META:
            if not cmd.startswith(base + " "):
                continue
            if cmd == base:
                continue
            # Extract the sub-command part: "/roles add" → "add"
            sub = cmd[len(base) + 1:]
            if sub.startswith(rest):
                yield Completion(
                    sub,
                    start_position=-len(rest),
                    display_meta=desc,
                )


# ── ANSI rendering helpers ───────────────────────────────────────────

def _get_terminal_width() -> int:
    """Get current terminal width, suitable for Rich rendering."""
    try:
        from prompt_toolkit.application.current import get_app
        app = get_app()
        if app and app.output:
            return app.output.get_size().columns
    except Exception:
        pass
    return shutil.get_terminal_size().columns


def _render_markdown(text: str, width: int | None = None) -> str:
    """Render markdown to ANSI string via rich."""
    w = width or _get_terminal_width()
    console = RichConsole(
        force_terminal=True,
        color_system="truecolor",
        width=w,
        file=StringIO(),
    )
    with console.capture() as capture:
        console.print(Markdown(text, code_theme="github-dark"))
    return capture.get().rstrip()


def _render_user_message(text: str, width: int | None = None) -> str:
    """Render a user message to ANSI string."""
    w = width or _get_terminal_width()
    console = RichConsole(force_terminal=True, color_system="truecolor", width=w)
    with console.capture() as capture:
        console.print(
            RichText("▸ You", style="bold #79c0ff"),
            RichText(f"  {text}", style="#e6edf3"),
        )
    return capture.get().rstrip()


def _render_system_message(text: str, width: int | None = None) -> str:
    """Render a system message to ANSI string."""
    w = width or _get_terminal_width()
    console = RichConsole(force_terminal=True, color_system="truecolor", width=w)
    with console.capture() as capture:
        console.print(RichText(text, style="italic #8b949e"))
    return capture.get().rstrip()


def _render_error(text: str, width: int | None = None) -> str:
    """Render an error message to ANSI string."""
    w = width or _get_terminal_width()
    console = RichConsole(force_terminal=True, color_system="truecolor", width=w)
    with console.capture() as capture:
        console.print(RichText(f"✗ {text}", style="bold #f78166"))
    return capture.get().rstrip()


def _render_panel(text: str, title: str = "", border_style: str = "#30363d",
                  width: int | None = None) -> str:
    """Render text in a rich Panel to ANSI."""
    w = width or _get_terminal_width()
    console = RichConsole(force_terminal=True, color_system="truecolor", width=w)
    with console.capture() as capture:
        console.print(Panel(Markdown(text), title=title,
                          border_style=border_style, padding=(0, 1)))
    return capture.get().rstrip()


# ── ChatSession (reused from original tui) ───────────────────────────

class ChatSession:
    """Holds mutable chat state: model, role, mode, messages."""

    def __init__(self, config: SynapseConfig, role: str, model: str, mode: str):
        self.config = config
        self.role_name = role
        self.model_name = model
        self.mode = mode
        self.messages: list[dict[str, str]] = []
        self._rebuild_provider()

    def _rebuild_provider(self):
        role_config = self.config.roles[self.role_name]
        model_config = self.config.models[self.model_name]
        if role_config.system_prompt:
            self.messages = [{"role": "system", "content": role_config.system_prompt}]
        else:
            self.messages = []
        self.provider = get_provider_for_model(model_config)
        dp = model_config.default_params
        self.temperature = dp.temperature
        self.max_tokens = dp.max_tokens

    def switch_model(self, name: str) -> bool:
        if name not in self.config.models:
            return False
        self.model_name = name
        self._rebuild_provider()
        return True

    def switch_role(self, name: str) -> bool:
        if name not in self.config.roles:
            return False
        self.role_name = name
        self.model_name = self.config.roles[name].model
        self._rebuild_provider()
        return True

    @property
    def model_config(self):
        return self.config.models[self.model_name]


# ── FullScreenTUI ────────────────────────────────────────────────────

class FullScreenTUI:
    """OpenCode-style full-screen terminal UI with self-contained scrollable chat."""

    def __init__(
        self,
        config: SynapseConfig,
        role: str = "default",
        model: str | None = None,
        mode: str = "auto",
    ):
        self.config = config
        model_name = model or config.roles[role].model
        self.session = ChatSession(config, role, model_name, mode)
        self.session_id = str(uuid.uuid4())

        # Chat messages: list of (role, raw_text, rendered_ansi)
        # role: "user", "assistant", "system"
        self._chat_lines: list[tuple[str, str]] = []  # (role, raw_text)
        # Streaming state
        self._streaming_text = ""
        self._streaming_reasoning = ""
        self._is_streaming = False

        # Input — manual Buffer + BufferControl for proper completion menu
        # (TextArea creates its own nested FloatContainer which conflicts
        #  with our outer FloatContainer, preventing the completion float
        #  from rendering)
        self._input_buffer = Buffer(
            completer=SlashCompleter(),
            complete_while_typing=True,
            history=PTFileHistory(str(_HISTORY_FILE)),
        )
        self._input_control = BufferControl(
            buffer=self._input_buffer,
            input_processors=[],
        )
        self._input_window = Window(
            content=self._input_control,
            height=1,
            style="class:input.text-area",
        )
        # Prompt window
        self._prompt_window = Window(
            content=FormattedTextControl(
                text=[("class:input.text-area.prompt", "› ")],
            ),
            width=2,
            height=1,
            style="class:input.text-area",
        )
        # Combined input line: prompt + input
        self._input_line = VSplit([
            self._prompt_window,
            self._input_window,
        ])

        # Key bindings
        self._kb = KeyBindings()

        @self._kb.add("enter")
        def _(event):
            """Submit input — processes in background, app stays open."""
            if self._is_streaming:
                return  # Don't accept input while AI is responding

            text = self._input_buffer.text.strip()
            if not text:
                return
            self._input_buffer.reset()
            self._pending_input = text

            # Quit commands exit immediately
            verb = text.split()[0] if text.startswith("/") else ""
            if verb in ("/quit", "/exit", "/q"):
                event.app.exit(result="quit")
                return
            # Suspend commands exit temporarily (need raw terminal)
            if verb in ("/setup", "/roles"):
                event.app.exit(result="suspend")
                return

            # Everything else: process in background, app stays open
            event.app.create_background_task(self._process_input(text))

        @self._kb.add("c-c")
        def _(event):
            """Ctrl+C cancels streaming or quits app."""
            if self._is_streaming:
                self._cancel_streaming = True
            else:
                event.app.exit(result="quit")

        @self._kb.add("c-d")
        def _(event):
            """Ctrl+D quits if input is empty."""
            if not self._input_buffer.text:
                event.app.exit(result="quit")

        @self._kb.add("escape")
        def _(event):
            """Escape focuses the chat window for scrolling."""
            event.app.layout.focus(self._chat_window)

        # ── Chat scrolling (only when chat window has focus) ──
        @self._kb.add("up", filter=has_focus(self._chat_window))
        def _(event):
            self._chat_window.vertical_scroll -= 1

        @self._kb.add("down", filter=has_focus(self._chat_window))
        def _(event):
            self._chat_window.vertical_scroll += 1

        @self._kb.add("pageup", filter=has_focus(self._chat_window))
        def _(event):
            self._chat_window.vertical_scroll -= (
                event.app.renderer.output.get_size().rows // 2
            )

        @self._kb.add("pagedown", filter=has_focus(self._chat_window))
        def _(event):
            self._chat_window.vertical_scroll += (
                event.app.renderer.output.get_size().rows // 2
            )

        @self._kb.add("home", filter=has_focus(self._chat_window))
        def _(event):
            self._chat_window.vertical_scroll = 0

        @self._kb.add("end", filter=has_focus(self._chat_window))
        def _(event):
            self._chat_window.vertical_scroll = 999_999  # scroll to bottom

        @self._kb.add("i", filter=has_focus(self._chat_window))
        def _(event):
            """Press 'i' to focus input from chat view."""
            event.app.layout.focus(self._input_line)

        @self._kb.add("/", filter=~has_focus(self._input_buffer))
        def _(event):
            """Press '/' to focus input and start typing a command."""
            self._input_buffer.text = "/"
            self._input_buffer.cursor_position = 1
            event.app.layout.focus(self._input_line)

        # Async state
        self._pending_input: str | None = None
        self._cancel_streaming = False
        self._running = True

        # Build layout
        self._build_layout()

    def _build_layout(self):
        """Build the prompt_toolkit Layout with header, chat, and input."""

        # ── Header ──
        def _get_header_text():
            mc = self.session.model_config
            mode_icon = {
                "single": "○", "orchestrate": "⬡", "debate": "◇",
                "pipeline": "→", "auto": "◎",
            }.get(self.session.mode, "?")
            return HTML(
                f" Synapse  ·  <b>{self.session.model_name}</b> ({mc.provider})  "
                f"·  {mode_icon} {self.session.mode}  "
                f"·  Role: {self.session.role_name}"
            )

        self._header_control = FormattedTextControl(
            text=_get_header_text,
            style="class:header",
        )
        header_window = Window(
            content=self._header_control,
            height=1,
            style="class:header",
        )

        # ── Chat area ──
        def _get_chat_text():
            """Build the full chat display text from stored lines + streaming."""
            # Render on-the-fly with current terminal width for responsiveness
            tw = _get_terminal_width()

            parts: list = []
            for role, raw in self._chat_lines:
                if role == "user":
                    ansi = _render_user_message(raw, width=tw)
                    parts.extend(ANSI(ansi).__pt_formatted_text__())
                elif role == "assistant":
                    ansi = _render_panel(raw, title=f"◉ {self.session.model_name}",
                                         border_style="#7ee787", width=tw)
                    parts.extend(ANSI(ansi).__pt_formatted_text__())
                elif role == "system":
                    ansi = _render_system_message(raw, width=tw)
                    parts.extend(ANSI(ansi).__pt_formatted_text__())
                else:
                    parts.append(("", raw))
                parts.append(("", "\n"))

            if self._is_streaming and (self._streaming_text or self._streaming_reasoning):
                # Live-streaming: render current accumulated text
                parts.append(("class:ai", f"◉ {self.session.model_name}\n"))
                if self._streaming_reasoning:
                    parts.append(("class:system", "🤔 Thinking...\n"))
                    reasoning_ansi = _render_markdown(self._streaming_reasoning, width=tw)
                    parts.extend(ANSI(reasoning_ansi).__pt_formatted_text__())
                    if self._streaming_text:
                        parts.append(("", "\n"))
                        parts.append(("class:separator", "─" * 40 + "\n"))
                if self._streaming_text:
                    content_ansi = _render_markdown(self._streaming_text, width=tw)
                    parts.extend(ANSI(content_ansi).__pt_formatted_text__())

            if not parts:
                # Show welcome message
                parts.append(("class:system",
                              "Welcome to Synapse! Type a message to start.\n"
                              "  /help — show commands  |  /setup — add models  |  /quit — exit"))

            return FormattedText(parts)

        self._chat_control = FormattedTextControl(
            text=_get_chat_text,
            style="class:chat",
            focusable=True,
        )
        self._chat_window = Window(
            content=self._chat_control,
            wrap_lines=True,
            always_hide_cursor=True,
            style="class:chat",
            allow_scroll_beyond_bottom=True,
        )

        # ── Separator ──
        separator = Window(
            height=1,
            char="─",
            style="class:separator",
        )

        # ── Hints bar ──
        def _get_hints_text():
            return HTML(
                "  <hints>/help</hints>  <hints>/mode</hints>  <hints>/model</hints>  "
                "<hints>/setup</hints>  <hints>/roles</hints>  "
                "<hints>/compact</hints>  <hints>/clear</hints>  "
                "<hints>/quit</hints> to exit"
            )

        hints_window = Window(
            content=FormattedTextControl(text=_get_hints_text, style="class:hints"),
            height=1,
            style="class:hints",
        )

        # ── Input (manual BufferControl + VSplit for prompt) ──

        # ── Full layout (wrapped in FloatContainer for completion menu) ──
        root_container = FloatContainer(
            content=HSplit([
                header_window,
                self._chat_window,
                separator,
                hints_window,
                self._input_line,
            ]),
            floats=[
                # Completion dropdown — positioned at cursor, shown only when
                # the focused buffer has completions (auto-filtered by CompletionsMenu)
                Float(
                    xcursor=True,
                    ycursor=True,
                    transparent=True,
                    content=CompletionsMenu(
                        max_height=12,
                        scroll_offset=1,
                        extra_filter=has_focus(self._input_buffer),
                    ),
                ),
            ],
        )

        self._layout = Layout(root_container, focused_element=self._input_line)

    # ── Rendering API ─────────────────────────────────────────────────

    def _add_user_message(self, text: str):
        """Append a user message to the chat."""
        self._chat_lines.append(("user", text))
        self._scroll_to_bottom()

    def _add_assistant_message(self, text: str):
        """Append a complete assistant message to the chat."""
        self._chat_lines.append(("assistant", text))
        self._scroll_to_bottom()

    def _add_system_message(self, text: str):
        """Append a system message to the chat."""
        self._chat_lines.append(("system", text))
        self._scroll_to_bottom()

    def _add_error(self, text: str):
        """Append an error message to the chat."""
        self._chat_lines.append(("system", text))
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Ensure the chat window shows the latest message."""
        # Force the window to scroll to bottom by moving cursor past end
        pass  # prompt_toolkit handles this via allow_scroll_beyond_bottom

    def _clear_chat(self):
        """Clear all visible messages (keep system prompt)."""
        self._chat_lines.clear()
        self.session.messages = [
            m for m in self.session.messages if m.get("role") == "system"
        ]

    # ── Streaming ─────────────────────────────────────────────────────

    async def _stream_response(self, user_input: str):
        """Stream AI response into the chat window in real-time.

        Handles reasoning/thinking tokens (DeepSeek R1, Claude thinking, etc.)
        by displaying them in a dimmed collapsible section before the response.
        """
        self._is_streaming = True
        self._streaming_text = ""
        self._streaming_reasoning = ""
        self._cancel_streaming = False

        app = get_app()

        try:
            stream = self.session.provider.chat_stream(
                messages=self.session.messages,
                temperature=self.session.temperature,
                max_tokens=self.session.max_tokens,
            )

            last_refresh = 0.0
            async for chunk in stream:
                if self._cancel_streaming:
                    self._add_system_message("[Streaming cancelled]")
                    break

                # Separate reasoning from content
                if chunk.reasoning:
                    self._streaming_reasoning += chunk.reasoning
                if chunk.content:
                    self._streaming_text += chunk.content

                # Throttle refreshes to ~15 fps
                import time
                now = time.monotonic()
                if now - last_refresh > 0.066:
                    app.invalidate()
                    last_refresh = now

        except Exception as e:
            self._add_error(f"Stream error: {e}")
        finally:
            self._is_streaming = False
            reasoning = self._streaming_reasoning
            full_text = self._streaming_text
            self._streaming_reasoning = ""
            self._streaming_text = ""

            if full_text:
                # If there was reasoning, prepend it as a dimmed block
                if reasoning:
                    full_text = (
                        f"<details><summary>🤔 Thinking</summary>\n\n"
                        f"{reasoning}\n\n</details>\n\n{full_text}"
                    )
                self.session.messages.append(
                    {"role": "assistant", "content": full_text}
                )
                self._add_assistant_message(full_text)

            app.invalidate()

    # ── Rich console capture (for multi-agent modes) ──────────────────

    def _make_capture_console(self) -> RichConsole:
        """Create a rich console that captures output to a string buffer."""
        buffer = StringIO()
        return RichConsole(
            force_terminal=True,
            color_system="truecolor",
            width=_get_terminal_width(),
            file=buffer,
            record=True,
        ), buffer

    # ── Input processing ──────────────────────────────────────────────

    async def _process_input(self, text: str):
        """Process user input as a background task while the app stays open.

        For slash commands, delegates to _handle_command.
        For chat messages, streams the AI response in real-time.
        After processing, invalidates the display to refresh the chat.
        """
        try:
            if text.startswith("/"):
                await self._handle_command(text)
            else:
                # Natural language detection for setup intent
                if self._is_setup_intent(text):
                    self._add_system_message("Tip: type /setup to configure models")
                    get_app().invalidate()
                    return

                self._add_user_message(text)
                self.session.messages.append({"role": "user", "content": text})

                effective_mode = self.session.mode
                if effective_mode == "auto":
                    effective_mode = detect_mode(text)

                if effective_mode == "single":
                    await self._stream_response(text)
                else:
                    await self._run_multi_agent(text)
        except Exception as e:
            self._add_error(f"Error: {e}")
        finally:
            try:
                get_app().invalidate()
            except Exception:
                pass

    # ── Command handling ──────────────────────────────────────────────

    async def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns False to quit the app."""
        parts = cmd.split(maxsplit=1)
        verb = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # ── Exit ──
        if verb in ("/quit", "/exit", "/q"):
            self._add_system_message("Goodbye! 👋")
            self._running = False
            return False

        # ── Clear ──
        if verb == "/clear":
            self._clear_chat()
            self._add_system_message("Chat cleared.")
            return True

        # ── Help ──
        if verb == "/help":
            lines = []
            for cmd, desc in _COMMANDS_WITH_META:
                lines.append(f"• **{cmd}** — {desc}")
            self._add_system_message("## Available Commands\n\n" + "\n".join(lines))
            return True

        # ── Mode ──
        if verb == "/mode":
            valid = ["single", "orchestrate", "debate", "pipeline", "auto"]
            if arg in valid:
                self.session.mode = arg
                self._add_system_message(f"Mode → **{arg}**")
            else:
                self._add_system_message(
                    f"Valid modes: {', '.join(valid)}\nCurrent: **{self.session.mode}**"
                )
            return True

        # ── Model ──
        if verb == "/model":
            if not arg:
                lines = ["## Available Models"]
                for name, mc in self.session.config.models.items():
                    cur = " ←" if name == self.session.model_name else ""
                    lines.append(f"• **{name}** ({mc.provider}) — {mc.model}{cur}")
                self._add_system_message("\n".join(lines))
            elif self.session.switch_model(arg):
                mc = self.session.config.models[arg]
                self._add_system_message(f"Model → **{arg}** ({mc.provider})")
            else:
                self._add_error(
                    f"Model '{arg}' not found. Available: {list(self.session.config.models.keys())}"
                )
            return True

        # ── Role ──
        if verb == "/role":
            if not arg:
                lines = ["## Available Roles"]
                for name, rc in self.session.config.roles.items():
                    cur = " ←" if name == self.session.role_name else ""
                    lines.append(f"• **{name}** → {rc.model} — {rc.description}{cur}")
                self._add_system_message("\n".join(lines))
            elif self.session.switch_role(arg):
                self._add_system_message(
                    f"Role → **{arg}** → Model: **{self.session.model_name}**"
                )
            else:
                self._add_error(
                    f"Role '{arg}' not found. Available: {list(self.session.config.roles.keys())}"
                )
            return True

        # ── Setup / Config ──
        if verb == "/setup":
            await self._do_setup()
            return True

        if verb == "/roles":
            await self._do_roles()
            return True

        if verb == "/compact":
            await self._do_compact()
            return None  # continue normally, chat was updated

        if verb == "/check":
            missing = find_required_keys(self.session.config)
            if missing:
                lines = ["**Missing API keys:**"]
                for k in missing:
                    lines.append(f"• `{k}`")
                self._add_system_message("\n".join(lines))
            else:
                self._add_system_message("✓ All API keys configured")
            return True

        if verb == "/config":
            lines = [
                f"**Models:** {list(self.session.config.models.keys())}",
                f"**Roles:** {list(self.session.config.roles.keys())}",
                f"**Memory store:** {self.session.config.memory.store_dir}",
                f"**Mode:** {self.session.mode}",
            ]
            self._add_system_message("\n".join(lines))
            return True

        # ── Memory ──
        if verb == "/remember":
            if not arg:
                self._add_error("Usage: /remember <content to remember>")
            else:
                try:
                    from synapse.memory import MemoryAgent, MemoryCategory
                    agent = MemoryAgent(self.session.config)
                    mid = await agent.remember(arg, category=MemoryCategory.FACT)
                    self._add_system_message(f"✓ Saved to memory ({mid})")
                except Exception as e:
                    self._add_error(str(e))
            return True

        if verb == "/recall":
            if not arg:
                self._add_error("Usage: /recall <search query>")
            else:
                try:
                    from synapse.memory import MemoryAgent
                    agent = MemoryAgent(self.session.config)
                    mems = await agent.recall(arg, top_k=5)
                    if mems:
                        lines = ["## Recalled Memories"]
                        for i, m in enumerate(mems, 1):
                            lines.append(f"{i}. [{m.category.value}] {m.content[:200]}")
                        self._add_system_message("\n".join(lines))
                    else:
                        self._add_system_message("No memories found.")
                except Exception as e:
                    self._add_error(str(e))
            return True

        if verb == "/facts":
            try:
                from synapse.memory import MemoryAgent
                agent = MemoryAgent(self.session.config)
                facts = agent.get_facts()
                if facts:
                    lines = ["## Stored Facts"]
                    for k, v in facts.items():
                        lines.append(f"• **{k}**: {v}")
                    self._add_system_message("\n".join(lines))
                else:
                    self._add_system_message("No facts stored yet.")
            except Exception as e:
                self._add_error(str(e))
            return True

        if verb == "/stats":
            try:
                from synapse.memory import MemoryAgent
                agent = MemoryAgent(self.session.config)
                s = agent.stats()
                self._add_system_message(
                    f"Memories: {s['total_memories']} | "
                    f"Sessions: {s['total_sessions']} | "
                    f"Facts: {s['total_facts']}"
                )
            except Exception as e:
                self._add_error(str(e))
            return True

        # ── Session ──
        if verb == "/session":
            sub_parts = arg.split(maxsplit=1)
            sub = sub_parts[0].lower() if sub_parts else ""
            sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

            if sub == "save":
                title = sub_arg or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                try:
                    from synapse.memory import MemoryAgent
                    agent = MemoryAgent(self.session.config)
                    await agent.compact(self.session_id, self.session.messages,
                                        title=title)
                    self._add_system_message(
                        f"✓ Session saved: **{title}** "
                        f"({len(self.session.messages)} messages)"
                    )
                except Exception as e:
                    self._add_error(str(e))
            elif sub == "list":
                try:
                    from synapse.memory import MemoryStore
                    store_dir = Path.home() / ".synapse"
                    store = MemoryStore(store_dir / "synapse.db")
                    sessions = store.list_sessions(limit=10)
                    if sessions:
                        lines = ["## Recent Sessions"]
                        for s in sessions:
                            lines.append(
                                f"• `{s.id[:8]}` {s.title or '(untitled)'} "
                                f"— {s.message_count} msgs — {s.created_at[:16]}"
                            )
                        self._add_system_message("\n".join(lines))
                    else:
                        self._add_system_message("No saved sessions.")
                except Exception as e:
                    self._add_error(str(e))
            elif sub == "load":
                session_prefix = sub_arg.strip() if sub_arg else ""
                if not session_prefix:
                    self._add_error(
                        "Usage: /session load <session_id_prefix>\n"
                        "Use /session list to see available sessions."
                    )
                else:
                    try:
                        from synapse.memory import MemoryStore
                        store_dir = Path.home() / ".synapse"
                        store = MemoryStore(store_dir / "synapse.db")

                        # Find session by prefix match
                        all_sessions = store.list_sessions(limit=100)
                        matched = [
                            s for s in all_sessions
                            if s.id.startswith(session_prefix)
                        ]
                        if not matched:
                            self._add_error(
                                f"No session found with prefix '{session_prefix}'.\n"
                                "Use /session list to see available sessions."
                            )
                        elif len(matched) > 1:
                            lines = ["## Multiple matches — be more specific:"]
                            for s in matched:
                                lines.append(
                                    f"• `{s.id[:8]}` {s.title or '(untitled)'} "
                                    f"— {s.created_at[:16]}"
                                )
                            self._add_system_message("\n".join(lines))
                        else:
                            session = matched[0]

                            # Build context from session summary
                            context_parts = [
                                f"[已恢复会话: {session.title or 'Untitled'}]",
                                f"创建时间: {session.created_at[:16]}",
                                f"消息数: {session.message_count}",
                            ]
                            if session.summary:
                                context_parts.append(
                                    f"\n--- 上次对话摘要 ---\n{session.summary}"
                                )

                            context = "\n".join(context_parts)

                            # Replace session messages with context
                            system_msgs = [
                                m for m in self.session.messages
                                if m.get("role") == "system"
                            ]
                            self.session.messages = system_msgs + [
                                {"role": "system", "content": context}
                            ]
                            self._clear_chat()
                            self._add_system_message(
                                f"✓ Session loaded: **{session.title or session.id[:8]}**\n"
                                f"  {session.message_count} messages compressed → summary context\n"
                                f"  Type anything to continue the conversation."
                            )
                    except Exception as e:
                        self._add_error(f"Session load error: {e}")
            else:
                self._add_system_message(
                    "**/session save** [title] — Save conversation\n"
                    "**/session list** — List saved sessions\n"
                    "**/session load** <id_prefix> — Resume a saved session"
                )
            return True

        # ── Unknown ──
        self._add_error(
            f"Unknown command: {verb}. Type /help for available commands."
        )
        return True

    # ── Interactive command implementations ────────────────────────────

    async def _do_setup(self):
        """Run the /setup guided flow in raw terminal."""
        try:
            from synapse.cli.onboarding import chat_setup
            await chat_setup(self.session.config)
            self.session.config = load_synapse_config()
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nSetup cancelled.")

    async def _do_roles(self):
        """Run the /roles interactive flow."""
        from rich.console import Console
        from rich.prompt import Prompt
        from synapse.config.loader import save_config

        console = Console()
        config = self.session.config
        models = list(config.models.keys())
        roles = list(config.roles.keys())

        # Show mapping
        table = Table(title="Role → Model Mapping")
        table.add_column("Role", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Purpose", style="dim")
        for name, rc in config.roles.items():
            table.add_row(name, rc.model, rc.description or "—")
        console.print(table)
        console.print()

        if len(models) < 2:
            console.print(
                "[yellow]You only have one model.[/yellow] "
                "Add more with [bold]/setup[/bold] first."
            )
            return

        while True:
            action = safe_ask(
                "What would you like to do?",
                choices=["reassign", "create", "done"],
                default="done",
            )
            if action == "done":
                save_config(config)
                self.session.config = load_synapse_config()
                console.print("\n[green]✓ Changes saved![/green]\n")
                break
            elif action == "reassign":
                role_name = safe_ask("Which role?", choices=roles, default=roles[0])
                rc = config.roles[role_name]
                choice = safe_ask(
                    f"New model for [cyan]{role_name}[/cyan]",
                    choices=models,
                    default=rc.model,
                )
                if choice and choice != rc.model:
                    rc.model = choice
                    console.print(f"  [green]✓[/green] {role_name} → {choice}")
            elif action == "create":
                from synapse.config.schema import RoleConfig
                name = Prompt.ask("Role name (lowercase, no spaces)")
                if not name:
                    continue
                name = name.strip().lower().replace(" ", "_")
                if name in config.roles:
                    console.print(f"[yellow]Role '{name}' already exists.[/yellow]")
                    continue
                desc = Prompt.ask("  Description", default="Custom role")
                model = safe_ask("  Model", choices=models, default=models[0])
                sp = Prompt.ask("  System prompt", default="You are a helpful AI assistant.")
                config.roles[name] = RoleConfig(
                    description=desc,
                    model=model,
                    system_prompt=sp,
                )
                console.print(f"\n[green]✓[/green] Created role [cyan]{name}[/cyan] → {model}")

    async def _do_compact(self):
        """Run context compression."""
        from synapse.memory.compactor import Compactor

        config = self.session.config
        conversation_msgs = [
            m for m in self.session.messages if m.get("role") != "system"
        ]

        if len(conversation_msgs) < 4:
            self._add_system_message(
                f"Not enough conversation to compress "
                f"({len(conversation_msgs)} messages — need at least 4)"
            )
            return

        original_count = len(conversation_msgs)
        total_chars = sum(len(m.get("content", "")) for m in conversation_msgs)

        provider = self.session.provider

        async def _compact_chat(messages, temperature, max_tokens):
            return await provider.chat(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )

        compactor = Compactor(provider=None)
        compactor._provider_fn = _compact_chat

        try:
            summary = await compactor.summarize(conversation_msgs)
        except Exception as e:
            self._add_error(f"Compression failed: {e}")
            return

        if not summary:
            self._add_error("Compression returned empty summary.")
            return

        # Auto-extract facts
        try:
            facts = await compactor.extract_facts(conversation_msgs, "default")
            if facts:
                from synapse.memory import MemoryAgent
                agent = MemoryAgent(self.session.config)
                for fd in facts:
                    await agent.upsert_fact(
                        key=fd.get("key", ""),
                        value=str(fd.get("value", "")),
                        namespace=fd.get("namespace", "global"),
                        confidence=fd.get("confidence", 1.0),
                    )
        except Exception:
            pass

        # Rebuild messages
        system_msgs = [m for m in self.session.messages if m.get("role") == "system"]
        context_block = (
            "[上下文摘要 — 以下是你与此用户之前对话的压缩摘要，"
            "请基于这些信息继续对话]\n\n" + summary
        )
        self.session.messages = system_msgs + [
            {"role": "assistant", "content": context_block}
        ]

        # Show result
        new_tokens = len(summary) // 4
        old_tokens = total_chars // 4
        reduction_pct = 100 - new_tokens * 100 // max(old_tokens, 1)

        self._clear_chat()
        self._add_system_message(
            f"✓ Context compressed! {original_count} messages → 1 summary | "
            f"~{old_tokens} tokens → ~{new_tokens} tokens ({reduction_pct}% reduction)"
        )
        # Redisplay system prompt + context
        self._chat_lines.clear()
        for msg in self.session.messages:
            if msg["role"] == "system":
                continue
            self._add_system_message(msg["content"])

    # ── Multi-agent mode handling ─────────────────────────────────────

    async def _run_multi_agent(self, user_input: str):
        """Run orchestrate/debate/pipeline mode and capture output."""
        effective_mode = self.session.mode
        if effective_mode == "auto":
            effective_mode = detect_mode(user_input)

        if effective_mode == "single":
            await self._stream_response(user_input)
            return

        # For multi-agent modes, capture rich output
        cap_console, cap_buffer = self._make_capture_console()

        result_text = ""
        try:
            if effective_mode == "debate":
                from synapse.cli.debate_ui import DebateUI
                result = await DebateUI(
                    self.session.config, console=cap_console
                ).run(user_input)
            elif effective_mode == "pipeline":
                from synapse.cli.pipeline_ui import PipelineUI
                result = await PipelineUI(
                    self.session.config, console=cap_console
                ).run(user_input)
            elif effective_mode == "orchestrate":
                from synapse.cli.orchestrate_ui import OrchestrationUI
                result = await OrchestrationUI(
                    self.session.config, console=cap_console
                ).run(user_input)
            else:
                await self._stream_response(user_input)
                return

            result_text = result
        except Exception as e:
            result_text = f"Error: {e}"

        # Get captured output
        captured = cap_buffer.getvalue()

        # Display in chat
        self._add_user_message(user_input)
        self.session.messages.append({"role": "user", "content": user_input})

        if result_text and result_text != user_input:
            self.session.messages.append({"role": "assistant", "content": result_text})
            self._add_assistant_message(result_text)

    # ── Main loop ─────────────────────────────────────────────────────

    async def run(self):
        """Main event loop — alternates between input and processing."""
        # First-time onboarding
        missing = find_required_keys(self.session.config)
        if missing and len(self.session.config.models) <= 1:
            print("No API keys found. Let's set up your first model.")
            print("Press Ctrl+C to skip setup.\n")
            try:
                from synapse.cli.onboarding import run_onboarding
                self.session.config = await run_onboarding(self.session.config)
                self.session = ChatSession(
                    self.session.config,
                    self.session.role_name,
                    self.session.model_name,
                    self.session.mode,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\nSkipping setup. You can run /setup anytime later.\n")

        self._running = True

        while self._running:
            # Run the prompt_toolkit application to get input
            app = Application(
                layout=self._layout,
                key_bindings=self._kb,
                style=_STYLE,
                full_screen=True,
                mouse_support=True,
            )

            try:
                result = await app.run_async()
            except Exception as exc:
                # Surface the error so the user can see what went wrong
                import traceback
                self._running = False
                print(f"\n✗ TUI crashed: {exc}")
                traceback.print_exc()
                print("\n(Report this bug at https://github.com/moyoti/synapse/issues)")
                break

            if result == "quit" or not self._running:
                break

            if result == "suspend":
                # App exited temporarily for /setup or /roles.
                # Process the command in raw terminal, then restart the app.
                user_input = self._pending_input
                self._pending_input = None
                if user_input:
                    await self._handle_command(user_input)
                continue

            # App exited for any other reason — just restart (shouldn't happen)
        # Cleanup
        self._save_history()

    def _is_setup_intent(self, text: str) -> bool:
        """Detect natural language requests for model setup."""
        text_lower = text.lower()
        patterns = [
            "怎么添加模型", "如何添加模型", "怎么接入", "如何接入",
            "add model", "add a model", "how to add", "connect model",
            "接入模型", "配置模型", "连接模型",
        ]
        return any(p in text_lower for p in patterns)

    def _save_history(self):
        """Save chat history to file."""
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            from prompt_toolkit.history import FileHistory
            # History is auto-saved by FileHistory
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────

async def run_chat_v2(
    mode: str = "auto",
    role: str | None = None,
    model: str | None = None,
):
    """Start interactive chat with the FullScreenTUI (v2)."""
    config = load_synapse_config()

    role_name = role or "default"
    if role_name not in config.roles:
        print(f"Role '{role_name}' not found. Available: {list(config.roles.keys())}")
        return

    model_name = model or config.roles[role_name].model
    if model_name not in config.models:
        print(
            f"Model '{model_name}' not found. Available: {list(config.models.keys())}"
        )
        return

    tui = FullScreenTUI(config, role=role_name, model=model_name, mode=mode)
    await tui.run()
