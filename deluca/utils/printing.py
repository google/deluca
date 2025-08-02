import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

# ---------------------------------------------------------------------------
# Environment detection (Jupyter vs. regular terminal)
# ---------------------------------------------------------------------------
try:
    # Detailed environment check: we are in a notebook *only* when the running
    # IPython shell is a ZMQInteractiveShell (used by Jupyter kernels).
    from IPython import get_ipython  # type: ignore
    from IPython.display import clear_output  # type: ignore

    _ip = get_ipython()
    # Detect whether we're running inside an IPython-powered notebook (Jupyter, Colab, etc.).
    # 1. If no IPython shell is present -> definitely not a notebook.
    # 2. If the shell class is *TerminalInteractiveShell* we are in a classic terminal.
    #    Anything else (ZMQInteractiveShell, InteractiveShell, etc.) is a notebook-like env.
    # 3. As an additional safeguard, explicitly look for the google.colab module which is
    #    always imported in Colab notebooks.
    if _ip is None:
        _IN_JUPYTER = False
    else:
        shell_name = _ip.__class__.__name__
        _IN_JUPYTER = shell_name != "TerminalInteractiveShell"

    # Explicit Colab check – covers edge-cases where the shell name heuristic fails.
    if not _IN_JUPYTER and "google.colab" in sys.modules:
        _IN_JUPYTER = True
except Exception:  # pragma: no cover – IPython not available at runtime
    _IN_JUPYTER = False

    # Define a stub so references are always valid even when IPython is absent.
    def clear_output(*_args, **_kwargs):  # type: ignore
        pass


_DEBUG = True

class _ProgressManager:
    """A singleton class to manage the rendering of all progress bars."""

    def __init__(self):
        self.root_tasks = []  # type: list[Task]
        self._lock = threading.Lock()
        self._render_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        self._lines_rendered = 0
        self.SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.SPINNER_INTERVAL = 0.1
        self.spinner_index = 0

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def add_task(self, task: "Task") -> None:
        with self._lock:
            self.root_tasks.append(task)
            if not self._running:
                self._start()

    def remove_task(self, task: "Task") -> None:
        # Mark task and all children as done before removing
        task._mark_tree_done()
        if task in self.root_tasks:
            if len(self.root_tasks) == 1:
                self._stop()
            self.root_tasks.remove(task)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()  # Ensure the event is clear when starting
        # Hide cursor in classic terminals. In Jupyter this escape sequence
        # is ignored, so it is harmless.
        sys.stdout.write("\033[?25l")
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def _stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stop_event.set()  # Signal the thread to stop

        if self._render_thread:
            self._render_thread.join()
            # A final render to show everything as completed.
            self._render(final=True)
            # Show cursor again (no-op in notebooks)
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    def _render_loop(self) -> None:
        if not _DEBUG:
            return

        while not self._stop_event.is_set():
            self._render()
            self._stop_event.wait(self.SPINNER_INTERVAL)

    # ---------------------------------------------------------------------
    # The heart of the progress-bar rendering logic
    # ---------------------------------------------------------------------
    def _render(self, final: bool = False) -> None:
        with self._lock:
            # ----------------------------------------------------------------
            # Notebook handling – we clear the entire cell output and redraw
            # the progress hierarchy from scratch.  Cursor-movement escape
            # sequences are not reliably supported by Jupyter.
            # ----------------------------------------------------------------
            if _IN_JUPYTER:
                clear_output(wait=True)
                # Since we cleared everything, reset the rendered-line counter
                self._lines_rendered = 0
            # ----------------------------------------------------------------
            # Classic terminal handling – move the cursor back to the start of
            # the block so we can redraw in-place.
            # ----------------------------------------------------------------
            if not _IN_JUPYTER and self._lines_rendered > 0:
                sys.stdout.write(f"\033[{self._lines_rendered}A")

            # ----------------------------------------------------------------
            # Build the visual representation for every root task.
            # ----------------------------------------------------------------
            lines: list[str] = []
            if not final:
                self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_CHARS)
                spinner_char = self.SPINNER_CHARS[self.spinner_index]
            else:
                spinner_char = "\033[92m✔\033[0m"

            for task in self.root_tasks:
                task._collect_lines(lines, spinner_char)

            # ----------------------------------------------------------------
            # Emit the lines.
            # ----------------------------------------------------------------
            for line in lines:
                sys.stdout.write(line)
                # Clear any characters that may remain from a previous render
                sys.stdout.write("\033[K")
                sys.stdout.write("\n")

            # ----------------------------------------------------------------
            # Clean-up of any leftover lines from the previous frame.
            # Only necessary in classic terminals.
            # ----------------------------------------------------------------
            if not _IN_JUPYTER:
                lines_to_clear = self._lines_rendered - len(lines)
                if lines_to_clear > 0:
                    for _ in range(lines_to_clear):
                        sys.stdout.write("\033[K\n")
                    # Move cursor back up so the next render starts in the
                    # correct place.
                    sys.stdout.write(f"\033[{lines_to_clear}A")

            self._lines_rendered = len(lines)
            sys.stdout.flush()


# Global singleton used by every Task
_manager = _ProgressManager()


class Task:
    """A class to create and manage a hierarchical progress bar."""

    _task_stack = deque()
    COLOR_SCHEMES = [
        ("\033[91m", "\033[94m", "\033[94m"),
        ("\033[38;5;208m", "\033[93m", "\033[93m"),
        ("\033[92m", "\033[95m", "\033[95m"),
    ]
    RESET_COLOR = "\033[0m"

    def __init__(self, name: str, total: int):
        self.name = name
        self.total = total
        self.current = 0
        self.done = False
        self.children: list[Task] = []
        self.parent: Optional[Task] = Task._task_stack[-1] if Task._task_stack else None
        self.level = self.parent.level + 1 if self.parent else 0
        self.update_text: Optional[str] = None

        if self.parent:
            self.parent.children.append(self)
        else:
            _manager.add_task(self)

        Task._task_stack.append(self)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def update(self, increment: int = 1, *, text: Optional[str] = None) -> None:
        if self.done:
            return
        self.current = min(self.total, self.current + increment)
        # Remove children that have completed so they no longer render.
        self.children = [c for c in self.children if not c.done]
        if text is not None:
            self.update_text = text

    def add_subtask(self, name: str, total: int) -> "Task":
        return Task(name, total)

    def _complete(self) -> None:
        if self.done:
            return

        if Task._task_stack and Task._task_stack[-1] is self:
            Task._task_stack.pop()

        # Mark this task and all its descendants as done
        self._mark_tree_done()

        if not self.parent:
            _manager.remove_task(self)

    def _mark_tree_done(self) -> None:
        """Recursively mark this task and all children as done."""
        self.done = True
        self.current = self.total
        for child in self.children:
            child._mark_tree_done()

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "Task":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._complete()

    def complete(self) -> None:
        self._complete()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_colors(self):
        return self.COLOR_SCHEMES[self.level % len(self.COLOR_SCHEMES)]

    def _collect_lines(self, lines: list[str], spinner_char: str) -> None:
        indent = "  " * self.level

        if self.done:
            status_char = "\033[92m✔\033[0m"
            bracket_color, bar_color, count_color = ("\033[92m",) * 3
        else:
            status_char = spinner_char
            bracket_color, bar_color, count_color = self._get_colors()

        bar_length = max(15, 30 - 5 * self.level)
        progress = self.current / self.total if self.total > 0 else 1.0
        filled_length = int(bar_length * progress)
        bar_str = "█" * filled_length + "░" * (bar_length - filled_length)

        progress_bar = (
            f"{bracket_color}[{bar_color}{bar_str}{bracket_color}]{self.RESET_COLOR}"
        )

        update_text_str = (
            f" ({bar_color}{self.update_text}{self.RESET_COLOR})" if self.update_text else ""
        )

        # Special-case: root task finishes – write a final clean line and skip
        # further rendering for this frame.
        if self.done and self.level == 0:
            count_str = f"{bracket_color}[Completed {self.total}]{self.RESET_COLOR}"
            sys.stdout.write(f"{status_char} {self.name} {progress_bar} {count_str}")
            sys.stdout.write("\033[K\n")  # Clear the rest of the line, then newline
            sys.stdout.flush()
            lines.clear()  # Prevent parent loop from reprinting this task now.
            return
        else:
            count_str = (
                f"{bracket_color}[{count_color}{self.current}/{self.total}{bracket_color}]"
                f"{self.RESET_COLOR}"
            )

        line = f"{indent}{status_char} {self.name}{update_text_str} {progress_bar} {count_str}"
        lines.append(line)

        if not self.done:
            for child in self.children:
                child._collect_lines(lines, spinner_char)


# ---------------------------------------------------------------------------
# tqdm-style helper – automatically creates and updates a Task while iterating
# ---------------------------------------------------------------------------
T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    name: str = "Progress",
    total: Optional[int] = None,
    update_text_fn: Optional[Callable[[T], str]] = None,
) -> Iterator[T]:
    """Iterator wrapper that displays a Task-based progress bar.

    Example
    -------
    >>> for i in progress(range(100), "Training"):
    ...     ...
    >>> for idx, item in progress(enumerate(items), "Enumerate"):
    ...     ...
    """
    # Infer *total* if not explicitly provided.
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            # Some iterators like ``enumerate`` expose ``__length_hint__``.
            total = getattr(iterable, "__length_hint__", lambda: None)()

    if total is None:
        raise ValueError(
            "Unable to determine *total* automatically; please pass it explicitly."
        )

    # The Task context ensures clean up even if the loop exits early.
    with Task(name, total) as _task:
        for item in iterable:
            yield item
            postfix = None
            if update_text_fn is not None:
                try:
                    postfix = update_text_fn(item)
                except Exception:
                    # Never let the progress bar crash user code.
                    postfix = None
            _task.update(text=postfix)
        # Task will be marked complete by the context manager.


# ---------------------------------------------------------------------------
# Self-test – executed only when run directly, ignored when imported.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick demo of nested tasks + progress helper.
    import random

    with Task("Outer", 2) as outer:
        for _ in range(2):
            with Task("Inner", 5) as inner:
                for _ in range(5):
                    time.sleep(0.05)
                    inner.update()
            outer.update()

    for i in progress(range(20), "Progress helper"):
        time.sleep(0.02)
