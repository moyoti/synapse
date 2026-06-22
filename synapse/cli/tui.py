"""
Full-screen TUI for Synapse chat — OpenCode-inspired terminal interface.

Layout:
  ┌─ Header bar: model / mode / memory stats ──────────────────────────┐
  │  Chat area: styled message bubbles with markdown, code highlighting │
  ├─ Input bar: prompt + command hints ─────────────────────────────────┤
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── prompt_toolkit for rich autocomplete ─────────────────────────────
try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory as PTFileHistory
    from prompt_toolkit.styles import Style as PTStyle
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.document import Document
    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False
    import readline as _fallback_readline

from rich.align import Align
from rich.box import Box, HEAVY, ROUNDED, SIMPLE
from rich.columns import Columns
from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from synapse import __version__
from synapse.cli.helpers import load_synapse_config
from synapse.config.loader import find_required_keys
from synapse.config.schema import SynapseConfig
from synapse.core.router import detect_mode
from synapse.models.registry import get_provider_for_model
from synapse.utils.streaming import stream_to_console

# ── Color palette ────────────────────────────────────────────────────

class Colors:
    """Synapse color system — warm, modern, accessible."""
    BG = "#0d1117"         # GitHub-dark background
    SURFACE = "#161b22"    # Card / panel surface
    BORDER = "#30363d"     # Subtle border
    ACCENT = "#58a6ff"     # Primary accent (blue)
    ACCENT2 = "#3fb950"    # Success (green)
    ACCENT3 = "#d2991d"    # Warning (gold)
    ACCENT4 = "#f78166"    # Error (orange-red)
    USER = "#79c0ff"       # User messages
    AI = "#7ee787"         # AI messages
    MUTED = "#8b949e"      # Muted / dim text
    HEADER_BG = "#0d419d"  # Header background

    # Styles
    HEADER = Style(color="white", bgcolor=HEADER_BG, bold=True)
    USER_STYLE = Style(color=USER, bold=True)
    AI_STYLE = Style(color=AI, bold=True)
    DIM = Style(color=MUTED, dim=True)
    ACCENT_STYLE = Style(color=ACCENT, bold=True)
    SUCCESS = Style(color=ACCENT2)
    WARNING = Style(color=ACCENT3)
    ERROR = Style(color=ACCENT4, bold=True)


# ── Command registry ──────────────────────────────────────────────────

# Full command list with descriptions for autocomplete
_COMMANDS_WITH_META = [
    ("/help", "Show all commands"),
    ("/clear", "Clear chat history"),
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
    ("/setup", "Add a new model (guided)"),
    ("/check", "Check API key status"),
    ("/config", "Show config summary"),
    ("/remember", "Save a fact to memory"),
    ("/recall", "Search memories"),
    ("/facts", "Show stored facts"),
    ("/stats", "Memory statistics"),
    ("/session save", "Save this conversation"),
    ("/session list", "List saved sessions"),
]

# Command names only (for fallback readline)
_COMMAND_NAMES = [c[0] for c in _COMMANDS_WITH_META]

# Nested structure for NestedCompleter-style typing (unused, kept for ref)
_COMMAND_DICT = {
    "/help": None, "/clear": None, "/quit": None, "/exit": None, "/q": None,
    "/mode": {"single": None, "orchestrate": None, "debate": None, "pipeline": None, "auto": None},
    "/model": None, "/role": None, "/setup": None, "/check": None, "/config": None,
    "/remember": None, "/recall": None, "/facts": None, "/stats": None,
    "/session": {"save": None, "list": None},
}

# Description map for fallback display
_COMMAND_DESCRIPTIONS = dict(_COMMANDS_WITH_META)

# History file
_HISTORY_FILE = Path.home() / ".synapse" / ".chat_history"

# ── prompt_toolkit autocomplete style ─────────────────────────────────

if _HAS_PROMPT_TOOLKIT:
    _PT_STYLE = PTStyle.from_dict({
        # Dropdown menu
        "completion-menu": "bg:#161b22 #e6edf3",
        "completion-menu.completion": "bg:#161b22 #58a6ff",
        "completion-menu.completion.current": "bg:#1f6feb #ffffff bold",
        # Scrollbar
        "scrollbar.background": "bg:#30363d",
        "scrollbar.button": "bg:#58a6ff",
        # Meta text (description shown next to completion)
        "completion-menu.completion.meta": "bg:#161b22 #8b949e italic",
    })
else:
    _PT_STYLE = None


def _build_completer():
    """Build a fuzzy-matching slash command completer with descriptions."""

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            for cmd, desc in _COMMANDS_WITH_META:
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )

    return SlashCompleter()


# ── Fallback readline setup (when prompt_toolkit not available) ──────

def _setup_readline_fallback():
    """Readline-based completion as fallback."""
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        _fallback_readline.read_history_file(str(_HISTORY_FILE))
    except (FileNotFoundError, OSError):
        pass

    def _completer(text: str, state: int) -> str | None:
        matches = [c for c in _COMMAND_NAMES if c.startswith(text)]
        try:
            return matches[state]
        except IndexError:
            return None

    _fallback_readline.set_completer(_completer)
    _fallback_readline.set_completer_delims(" \t\n")
    _fallback_readline.parse_and_bind("tab: complete")
    _fallback_readline.set_history_length(1000)


def _save_history_fallback():
    try:
        _fallback_readline.write_history_file(str(_HISTORY_FILE))
    except OSError:
        pass


# ── ChatSession ──────────────────────────────────────────────────────

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


# ── Main ChatTUI class ───────────────────────────────────────────────

class ChatTUI:
    """Full-screen chat interface with styled message bubbles and streaming."""

    def __init__(
        self,
        config: SynapseConfig,
        role: str = "default",
        model: str | None = None,
        mode: str = "auto",
    ):
        self.config = config
        self.console = Console()
        self.term_width = shutil.get_terminal_size().columns
        self.term_height = shutil.get_terminal_size().lines

        model_name = model or config.roles[role].model
        self.session = ChatSession(config, role, model_name, mode)
        self.session_id = f"ses_{uuid.uuid4().hex[:8]}"

        # Display message history (rendered Panels)
        self._rendered_messages: list[RenderableType] = []

        # Setup input method
        if _HAS_PROMPT_TOOLKIT:
            self._completer = _build_completer()
            self._history = PTFileHistory(str(_HISTORY_FILE))
        else:
            _setup_readline_fallback()

    # ── Prompt (input) ───────────────────────────────────────────────

    def _prompt(self) -> str:
        """Show the input prompt and return user input.

        Uses prompt_toolkit for live autocomplete dropdown with arrow-key
        navigation, falling back to readline if prompt_toolkit is unavailable.
        """
        # Build hint line
        hints = Text("  ", style=Colors.DIM)
        hints.append("/help", style=Style(color=Colors.MUTED))
        hints.append("  ", style=Colors.DIM)
        hints.append("/model", style=Style(color=Colors.MUTED))
        hints.append("  ", style=Colors.DIM)
        hints.append("/setup", style=Style(color=Colors.MUTED))
        hints.append("  ", style=Colors.DIM)
        hints.append("/clear", style=Style(color=Colors.MUTED))
        hints.append("  ", style=Colors.DIM)
        hints.append("/quit", style=Style(color=Colors.MUTED))
        hints.append(" to exit", style=Colors.DIM)

        self.console.print(hints)

        try:
            if _HAS_PROMPT_TOOLKIT:
                user_input = pt_prompt(
                    HTML("› "),  # prompt text
                    completer=self._completer,
                    history=self._history,
                    style=_PT_STYLE,
                    complete_while_typing=True,    # show dropdown as you type
                    complete_in_thread=True,        # non-blocking completion
                    reserve_space_for_menu=4,       # room for dropdown
                )
            else:
                prompt_str = "\033[38;2;88;166;255m› \033[0m"
                user_input = input(prompt_str)
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    # ── Rendering helpers ────────────────────────────────────────────

    def _render_header(self) -> Panel:
        """Render the top status bar."""
        mc = self.session.model_config
        mode_icon = {
            "single": "○", "orchestrate": "⬡", "debate": "◇",
            "pipeline": "→", "auto": "◎",
        }.get(self.session.mode, "?")

        segments = [
            ("🧠 Synapse", Colors.ACCENT_STYLE),
            ("  │", Colors.DIM),
            (f"  {mc.provider}/{self.session.model_name}", Style(color="white")),
            ("  │", Colors.DIM),
            (f"  {mode_icon} {self.session.mode}", Style(color=Colors.ACCENT2)),
            ("  │", Colors.DIM),
            (f"  role: {self.session.role_name}", Colors.DIM),
        ]

        text = Text()
        for s, style in segments:
            text.append(s, style=style)

        text.append(" " * 4, style=Colors.DIM)
        text.append(f"v{__version__}", style=Colors.DIM)

        return Panel(
            text,
            box=SIMPLE,
            style=Style(color=Colors.BORDER),
            padding=(0, 2),
        )

    def _render_user_message(self, content: str) -> Panel:
        return Panel(
            Markdown(content, code_theme="github-dark"),
            title="▶ You",
            title_align="left",
            border_style=Style(color=Colors.USER),
            box=ROUNDED,
            padding=(0, 1),
        )

    def _render_ai_message(self, content: str, model_name: str = "") -> Panel:
        title = f"◉ {model_name}" if model_name else "◉ Synapse"
        return Panel(
            Markdown(content, code_theme="github-dark"),
            title=title,
            title_align="left",
            border_style=Style(color=Colors.AI),
            box=ROUNDED,
            padding=(0, 1),
        )

    def _render_system_message(self, content: str) -> Panel:
        return Panel(
            content,
            border_style=Colors.DIM,
            box=SIMPLE,
            padding=(0, 1),
        )

    def _render_error(self, content: str) -> Panel:
        return Panel(
            f"[bold red]Error:[/bold red] {content}",
            border_style=Style(color=Colors.ACCENT4),
            box=SIMPLE,
            padding=(0, 1),
        )

    def _render_empty_state(self) -> Panel:
        welcome = Text()
        welcome.append("🧠  Welcome to ", style=Colors.DIM)
        welcome.append("Synapse", style=Colors.ACCENT_STYLE)
        welcome.append("\n\n", style=Colors.DIM)

        shortcuts = [
            ("/help", "Show all commands"),
            ("/model <name>", "Switch AI model"),
            ("/setup", "Add a new model (guided)"),
            ("/mode <name>", "Change mode (single/debate/pipeline)"),
            ("/remember <text>", "Save facts to memory"),
            ("/recall <query>", "Search memories"),
            ("/quit, /exit", "Exit chat"),
        ]

        for key, desc in shortcuts:
            welcome.append(f"  {key:<18}", style=Colors.ACCENT_STYLE)
            welcome.append(f"{desc}\n", style=Colors.DIM)

        welcome.append("\n", style=Colors.DIM)
        welcome.append("💡 Just type ", style=Colors.DIM)
        welcome.append("/setup", style=Style(color=Colors.ACCENT2, bold=True))
        welcome.append(" to connect a new model — I'll guide you step by step.", style=Colors.DIM)

        return Panel(
            welcome,
            title="[bold]Getting Started[/bold]",
            border_style=Style(color=Colors.BORDER),
            box=ROUNDED,
            padding=(1, 2),
        )

    # ── Display helpers ──────────────────────────────────────────────

    def _print_header_and_history(self):
        self.console.print(self._render_header())

        if not self._rendered_messages:
            self.console.print()
            self.console.print(Align.center(self._render_empty_state()))
            self.console.print()

        for msg in self._rendered_messages:
            self.console.print(msg)
            self.console.print()

    def _print_user_input(self, text: str):
        panel = self._render_user_message(text)
        self._rendered_messages.append(panel)
        self.console.print(panel)
        self.console.print()

    def _print_ai_response(self, response: str):
        panel = self._render_ai_message(response, self.session.model_name)
        self._rendered_messages.append(panel)
        self.console.print(panel)
        self.console.print()

    def _print_system(self, text: str):
        panel = self._render_system_message(text)
        self.console.print(panel)
        self.console.print()

    @staticmethod
    def _detect_add_model_intent(text: str) -> bool:
        text_lower = text.lower().strip().rstrip("?.!。？！")
        patterns = [
            "add model", "new model", "add a model", "add another model",
            "connect model", "configure model", "setup model", "set up model",
            "how to add", "how do i add", "how can i add",
            "add provider", "add llm", "add ai",
            "添加模型", "新增模型", "加模型", "增加模型",
            "怎么添加", "如何添加", "怎样添加", "如何接入",
            "添加一个模型", "再加一个模型",
            "接入模型", "配置模型", "连接模型",
        ]
        for p in patterns:
            if p in text_lower:
                return True
        return False

    # ── Streaming ────────────────────────────────────────────────────

    async def _stream_response(self, user_input: str) -> str:
        self.session.messages.append({"role": "user", "content": user_input})

        stream = self.session.provider.chat_stream(
            messages=self.session.messages,
            temperature=self.session.temperature,
            max_tokens=self.session.max_tokens,
        )

        full_text = ""
        model_name = self.session.model_name

        with Live(
            console=self.console,
            refresh_per_second=15,
            transient=False,
            vertical_overflow="visible",
        ) as live:
            async for token in stream:
                full_text += token
                live.update(
                    Panel(
                        Markdown(full_text, code_theme="github-dark"),
                        title=f"◉ {model_name}",
                        title_align="left",
                        border_style=Style(color=Colors.AI),
                        box=ROUNDED,
                        padding=(0, 1),
                    )
                )

        self.session.messages.append({"role": "assistant", "content": full_text})
        return full_text

    # ── Command handling ─────────────────────────────────────────────

    async def _handle_command(self, cmd: str) -> str | None:
        parts = cmd.split(maxsplit=1)
        verb = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # ── Exit ──
        if verb in ("/quit", "/exit", "/q"):
            self._print_system("Goodbye! 👋")
            self.close()
            return "quit"

        # ── Clear ──
        if verb == "/clear":
            self.session._rebuild_provider()
            self._rendered_messages.clear()
            return "clear"

        # ── Help ──
        if verb == "/help":
            self._print_help()
            return None

        # ── Mode ──
        if verb == "/mode":
            valid = ["single", "orchestrate", "debate", "pipeline", "auto"]
            if arg in valid:
                self.session.mode = arg
                self._print_system(f"Mode → [bold]{arg}[/bold]")
            else:
                self._print_system(
                    f"[red]Valid modes:[/red] {', '.join(valid)}\n"
                    f"  [dim]Current: {self.session.mode}[/dim]"
                )
            return None

        # ── Model ──
        if verb == "/model":
            if not arg:
                self._list_models()
            elif self.session.switch_model(arg):
                mc = self.session.config.models[arg]
                self._print_system(f"Model → [bold]{arg}[/bold] ({mc.provider})")
            else:
                self.console.print(self._render_error(
                    f"Model '{arg}' not found. Available: {list(self.session.config.models.keys())}"
                ))
            return None

        # ── Role ──
        if verb == "/role":
            if not arg:
                self._list_roles()
            elif self.session.switch_role(arg):
                self._print_system(f"Role → [bold]{arg}[/bold] → Model: [bold]{self.session.model_name}[/bold]")
            else:
                self.console.print(self._render_error(
                    f"Role '{arg}' not found. Available: {list(self.session.config.roles.keys())}"
                ))
            return None

        # ── Memory ──
        if verb == "/remember":
            if not arg:
                self.console.print(self._render_error("Usage: /remember <content to remember>"))
            else:
                try:
                    from synapse.memory import MemoryAgent, MemoryCategory
                    agent = MemoryAgent(self.session.config)
                    mid = await agent.remember(arg, category=MemoryCategory.FACT)
                    self._print_system(f"[green]✓[/green] Saved to memory [dim]({mid})[/dim]")
                except Exception as e:
                    self.console.print(self._render_error(str(e)))
            return None

        if verb == "/recall":
            if not arg:
                self.console.print(self._render_error("Usage: /recall <search query>"))
            else:
                try:
                    from synapse.memory import MemoryAgent
                    agent = MemoryAgent(self.session.config)
                    mems = await agent.recall(arg, top_k=5)
                    if mems:
                        lines = []
                        for i, m in enumerate(mems, 1):
                            lines.append(f"[cyan]{i}.[/cyan] [{m.category.value}] {m.content[:200]}")
                        self._print_system("\n".join(lines))
                    else:
                        self._print_system("[dim]No memories found.[/dim]")
                except Exception as e:
                    self.console.print(self._render_error(str(e)))
            return None

        if verb == "/facts":
            try:
                from synapse.memory import MemoryAgent
                agent = MemoryAgent(self.session.config)
                facts = agent.get_facts()
                if facts:
                    lines = ["[bold]Stored Facts:[/bold]"]
                    for k, v in facts.items():
                        lines.append(f"  • [cyan]{k}[/cyan]: {v}")
                    self._print_system("\n".join(lines))
                else:
                    self._print_system("[dim]No facts stored yet.[/dim]")
            except Exception as e:
                self.console.print(self._render_error(str(e)))
            return None

        if verb == "/stats":
            try:
                from synapse.memory import MemoryAgent
                agent = MemoryAgent(self.session.config)
                s = agent.stats()
                self._print_system(
                    f"Memories: {s['total_memories']} | "
                    f"Sessions: {s['total_sessions']} | "
                    f"Facts: {s['total_facts']}"
                )
            except Exception as e:
                self.console.print(self._render_error(str(e)))
            return None

        # ── Setup / Config ──
        if verb == "/setup":
            try:
                from synapse.cli.onboarding import chat_setup
                await chat_setup(self.session.config)
                self.session.config = load_synapse_config()
            except (KeyboardInterrupt, asyncio.CancelledError):
                self.console.print()
                self._print_system("[dim]Setup cancelled.[/dim]")
            return None

        if verb == "/check":
            missing = find_required_keys(self.session.config)
            if missing:
                lines = ["[yellow]Missing API keys:[/yellow]"]
                for k in missing:
                    lines.append(f"  • [cyan]{k}[/cyan] — set with: synapse config set-key {k} YOUR_KEY")
                self._print_system("\n".join(lines))
            else:
                self._print_system("[green]✓ All API keys configured[/green]")
            return None

        if verb == "/config":
            lines = [
                f"Models: {list(self.session.config.models.keys())}",
                f"Roles: {list(self.session.config.roles.keys())}",
                f"Memory store: {self.session.config.memory.store_dir}",
            ]
            self._print_system("\n".join(lines))
            return None

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
                    await agent.compact(self.session_id, self.session.messages, title=title)
                    self._print_system(
                        f"[green]✓[/green] Session saved: [bold]{title}[/bold] "
                        f"({len(self.session.messages)} messages)"
                    )
                except Exception as e:
                    self.console.print(self._render_error(str(e)))

            elif sub == "list":
                try:
                    from synapse.memory import MemoryStore
                    store_dir = Path.home() / ".synapse"
                    store = MemoryStore(store_dir / "synapse.db")
                    sessions = store.list_sessions(limit=10)
                    if sessions:
                        lines = ["[bold]Recent Sessions:[/bold]"]
                        for s in sessions:
                            lines.append(
                                f"  [dim]{s.id[:8]}[/dim] {s.title or '(untitled)'} "
                                f"— {s.message_count} msgs — {s.created_at[:16]}"
                            )
                        self._print_system("\n".join(lines))
                    else:
                        self._print_system("[dim]No saved sessions.[/dim]")
                except Exception as e:
                    self.console.print(self._render_error(str(e)))
            else:
                self._print_system(
                    "  /session save [title] — Save this conversation\n"
                    "  /session list         — List saved sessions"
                )
            return None

        # ── Unknown ──
        self.console.print(self._render_error(
            f"Unknown command: {verb}. Type [bold]/help[/bold] for available commands."
        ))
        return None

    def _print_help(self):
        help_table = Table(show_header=False, box=SIMPLE, padding=(0, 2))
        help_table.add_column("Command", style=Colors.ACCENT_STYLE)
        help_table.add_column("Description", style=Colors.DIM)

        help_table.add_row("[bold]Conversation[/bold]", "")
        help_table.add_row("/clear", "Clear chat history")
        help_table.add_row("/mode <name>", "Switch mode: single|orchestrate|debate|pipeline|auto")
        help_table.add_row("", "")
        help_table.add_row("[bold]Model & Role[/bold]", "")
        help_table.add_row("/model [name]", "List or switch models")
        help_table.add_row("/role [name]", "List or switch roles")
        help_table.add_row("", "")
        help_table.add_row("[bold]Memory[/bold]", "")
        help_table.add_row("/remember <text>", "Save a fact to memory")
        help_table.add_row("/recall <query>", "Search memories")
        help_table.add_row("/facts", "Show stored facts")
        help_table.add_row("/stats", "Memory statistics")
        help_table.add_row("", "")
        help_table.add_row("[bold]Session[/bold]", "")
        help_table.add_row("/session save [title]", "Save conversation")
        help_table.add_row("/session list", "List saved sessions")
        help_table.add_row("", "")
        help_table.add_row("[bold]Config[/bold]", "")
        help_table.add_row("/setup", "Add a new model")
        help_table.add_row("/check", "Check API keys")
        help_table.add_row("/config", "Show config summary")
        help_table.add_row("", "")
        help_table.add_row("[bold]Other[/bold]", "")
        help_table.add_row("/quit, /exit, /q", "Exit chat")

        self.console.print(
            Panel(help_table, title="[bold]Commands[/bold]", border_style=Style(color=Colors.BORDER), box=ROUNDED)
        )
        self.console.print()

    def _list_models(self):
        table = Table(title="Available Models", box=SIMPLE)
        table.add_column("Name", style=Colors.ACCENT_STYLE)
        table.add_column("Provider", style=Style(color=Colors.AI))
        table.add_column("Model ID", style=Colors.DIM)
        table.add_column("", style=Colors.SUCCESS, width=3)

        for name, mc in self.session.config.models.items():
            cur = "←" if name == self.session.model_name else ""
            table.add_row(name, mc.provider, mc.model, cur)

        self.console.print(table)
        self.console.print()

    def _list_roles(self):
        table = Table(title="Available Roles", box=SIMPLE)
        table.add_column("Name", style=Colors.ACCENT_STYLE)
        table.add_column("Model", style=Style(color=Colors.AI))
        table.add_column("Description", style=Colors.DIM)
        table.add_column("", style=Colors.SUCCESS, width=3)

        for name, rc in self.session.config.roles.items():
            cur = "←" if name == self.session.role_name else ""
            table.add_row(name, rc.model, rc.description, cur)

        self.console.print(table)
        self.console.print()

    # ── Main loop ────────────────────────────────────────────────────

    async def run(self):
        """Main chat loop with full TUI rendering."""
        # First-time onboarding
        missing = find_required_keys(self.session.config)
        if missing and len(self.session.config.models) <= 1:
            self._print_system("[yellow]No API keys found. Let's set up your first model.[/yellow]")
            self._print_system("[dim]Press Ctrl+C to skip setup[/dim]")
            try:
                from synapse.cli.onboarding import run_onboarding
                self.session.config = await run_onboarding(self.session.config)
                self.session = ChatSession(
                    load_synapse_config(),
                    self.session.role_name,
                    self.session.model_name,
                    self.session.mode,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                self.console.print()
                self._print_system("[dim]Skipping setup. You can run /setup anytime later.[/dim]")
                self.console.print()

        # Initial display
        self.console.clear()
        self._print_header_and_history()

        while True:
            try:
                user_input = self._prompt()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                self._print_system("Goodbye! 👋")
                self.close()
                break

            if not user_input:
                continue

            try:
                # Slash command
                if user_input.startswith("/"):
                    result = await self._handle_command(user_input)
                    if result == "quit":
                        break
                    elif result == "clear":
                        self.console.clear()
                        self._print_header_and_history()
                    continue

                # Smart suggestion
                if self._detect_add_model_intent(user_input):
                    self._print_system(
                        "💡 [bold]You can add a new model right here![/bold]\n"
                        "Just type [bold green]/setup[/bold green] and I'll guide you through it — "
                        "pick a provider, enter your API key, and you're ready to go.\n"
                        "[dim]Type /setup now, or just ask me a question.[/dim]"
                    )
                    continue

                # Normal chat
                self._print_user_input(user_input)

                effective_mode = self.session.mode
                if effective_mode == "auto":
                    effective_mode = detect_mode(user_input).value

                if effective_mode == "debate":
                    from synapse.cli.debate_ui import DebateUI
                    result = await DebateUI(self.session.config, console=self.console).run(user_input)
                    self._print_ai_response(result)
                elif effective_mode == "pipeline":
                    from synapse.cli.pipeline_ui import PipelineUI
                    result = await PipelineUI(self.session.config, console=self.console).run(user_input)
                    self._print_ai_response(result)
                elif effective_mode == "orchestrate":
                    from synapse.cli.orchestrate_ui import OrchestrationUI
                    result = await OrchestrationUI(self.session.config, console=self.console).run(user_input)
                    self._print_ai_response(result)
                else:
                    response = await self._stream_response(user_input)
                    panel = self._render_ai_message(response, self.session.model_name)
                    self._rendered_messages.append(panel)
                    self.console.print()

            except (KeyboardInterrupt, asyncio.CancelledError):
                self.console.print()
                self._print_system("[dim]Cancelled.[/dim]")
            except Exception as e:
                self.console.print(self._render_error(str(e)))
                self.console.print()

    # ── Cleanup ──────────────────────────────────────────────────────

    def close(self):
        """Cleanup resources."""
        if not _HAS_PROMPT_TOOLKIT:
            _save_history_fallback()
