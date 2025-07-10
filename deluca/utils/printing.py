import sys
import threading
import time
from collections import deque

_DEBUG = True

class _ProgressManager:
    """A singleton class to manage the rendering of all progress bars."""

    def __init__(self):
        self.root_tasks = []
        self._lock = threading.Lock()
        self._render_thread = None
        self._running = False
        self._stop_event = threading.Event()
        self._lines_rendered = 0
        self.SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.SPINNER_INTERVAL = 0.1
        self.spinner_index = 0

    def add_task(self, task):
        with self._lock:
            self.root_tasks.append(task)
            if not self._running:
                self._start()

    def remove_task(self, task):
        # Mark task and all children as done before removing
        task._mark_tree_done()
        if task in self.root_tasks:
            if len(self.root_tasks) == 1:
                self._stop()

            self.root_tasks.remove(task)

    def _start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()  # Ensure the event is clear when starting
        # Hide cursor
        sys.stdout.write("\033[?25l")
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def _stop(self):
        if not self._running:
            return

        self._running = False
        self._stop_event.set()  # Signal the thread to stop

        if self._render_thread:
            self._render_thread.join()
            #    A final render to show everything as completed.
            self._render(final=True)
            # Show cursor again
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    def _render_loop(self):
        if not _DEBUG:
            return
        
        while not self._stop_event.is_set():
            self._render()
            self._stop_event.wait(self.SPINNER_INTERVAL)

    def _render(self, final=False):
        with self._lock:
            # Move cursor up to the start of the progress block
            if self._lines_rendered > 0:
                sys.stdout.write(f"\033[{self._lines_rendered}A")

            lines = []
            if not final:
                self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_CHARS)
                spinner_char = self.SPINNER_CHARS[self.spinner_index]
            else:
                spinner_char = "\033[92m✔\033[0m"

            for task in self.root_tasks:
                task._collect_lines(lines, spinner_char)

            for line in lines:
                sys.stdout.write(line)
                sys.stdout.write("\033[K")  # Clear the rest of the line
                sys.stdout.write("\n")

            # Clear any remaining lines from the previous render
            lines_to_clear = self._lines_rendered - len(lines)
            if lines_to_clear > 0:
                for _ in range(lines_to_clear):
                    sys.stdout.write("\033[K\n")
                # Move cursor back up
                sys.stdout.write(f"\033[{lines_to_clear}A")

            self._lines_rendered = len(lines)
            sys.stdout.flush()


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
    
    def __init__(self, name, total):
        self.name = name
        self.total = total
        self.current = 0
        self.done = False
        self.children = []
        self.parent = Task._task_stack[-1] if Task._task_stack else None
        self.level = self.parent.level + 1 if self.parent else 0
        self.update_text = None

        if self.parent:
            self.parent.children.append(self)
        else:
            _manager.add_task(self)

        Task._task_stack.append(self)

    def update(self, increment=1, text=None):
        if self.done:
            return
        self.current = min(self.total, self.current + increment)
        self.children = [c for c in self.children if not c.done]
        if text is not None:
            self.update_text = text

    def add_subtask(self, name, total):
        return Task(name, total)

    def _complete(self):
        if self.done:
            return

        if Task._task_stack and Task._task_stack[-1] is self:
            Task._task_stack.pop()

        # Mark this task and all its descendants as done
        self._mark_tree_done()

        if not self.parent:
            _manager.remove_task(self)

    def _mark_tree_done(self):
        """Recursively mark this task and all children as done."""
        self.done = True
        self.current = self.total
        for child in self.children:
            child._mark_tree_done()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._complete()

    def complete(self):
        self._complete()

    def _get_colors(self):
        return self.COLOR_SCHEMES[self.level % len(self.COLOR_SCHEMES)]

    def _collect_lines(self, lines, spinner_char):
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

        update_text_str = f" ({bar_color}{self.update_text}{self.RESET_COLOR})" if self.update_text else ""

        if self.done and self.level == 0:
            count_str = f"{bracket_color}[Completed {self.total}]{self.RESET_COLOR}"
            print(f"{status_char} {self.name} {progress_bar} {count_str}")
            lines = []
            return
        else:
            count_str = f"{bracket_color}[{count_color}{self.current}/{self.total}{bracket_color}]{self.RESET_COLOR}"

        line = f"{indent}{status_char} {self.name}{update_text_str} {progress_bar} {count_str}"
        lines.append(line)

        if not self.done:
            for child in self.children:
                child._collect_lines(lines, spinner_char)


if __name__ == "__main__":
    with Task("Test learner", 3) as main_task:

        def test_function():
            with Task("Generating trajectories", 25) as task:
                for i in range(25):
                    task.update()
                    if i == 20:
                        with Task("Generating Trajectory 3/100", 30) as subtask:
                            for j in range(30):
                                subtask.update()
                                if j == 10:
                                    with Task("Sub-subtask", 10) as subsubtask:
                                        for k in range(10):
                                            subsubtask.update()
                                            time.sleep(0.01)
                                time.sleep(0.01)
                    time.sleep(0.01)

        test_function()
        main_task.update()

        test_function()
        main_task.update()

        test_function()
        main_task.update()

    with Task("Another Task", 5) as task2:
        for i in range(5):
            task2.update()
            time.sleep(0.1)
