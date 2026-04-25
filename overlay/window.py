"""
Transparent overlay window.

Manages the Tkinter root, canvas, Win32 click-through, control panel,
the thread-safe shot queue, and the 60 Hz render loop.

All drawing is delegated to renderer.py — this file owns the window,
not the pixels.
"""
import ctypes
import queue
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional

from models import Shot
from overlay.renderer import Renderer

_GWL_EXSTYLE       = -20
_WS_EX_TRANSPARENT = 0x00000020
_CHROMA            = "black"   # color used as the transparency key


class OverlayWindow:

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x      = x;     self.y      = y
        self.width  = width; self.height = height

        # Thread-safe queue: pipeline thread writes, UI thread reads
        self._q:    queue.Queue    = queue.Queue(maxsize=3)
        self._on:   bool           = True    # overlay visible?
        self._tick: int            = 0       # animation tick (0–39)

        self.root:      Optional[tk.Tk]     = None
        self.canvas:    Optional[tk.Canvas] = None
        self._renderer: Optional[Renderer]  = None

        # Wired by main.py after construction
        self.on_type_change: Optional[Callable[[str], None]] = None

    # ── window setup ──────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(
            f"{self.width}x{self.height}+{max(0, self.x)}+{max(0, self.y)}")
        self.root.configure(bg=_CHROMA)

        self.canvas = tk.Canvas(
            self.root, bg=_CHROMA, highlightthickness=0,
            width=self.width, height=self.height)
        self.canvas.pack(fill="both", expand=True)

        self.root.update_idletasks()
        self.root.attributes("-transparentcolor", _CHROMA)
        self.root.attributes("-topmost", True)
        self.root.lift()

        self._renderer = Renderer(self.canvas)

        # Click-through must be applied after the window is fully shown
        self.root.after(300, self._apply_clickthrough)
        self._build_panel()
        self.root.after(16, self._loop)   # start 60 Hz render loop

    def run(self) -> None:
        """Block until the overlay is closed. Call from the main thread."""
        self.root.mainloop()

    # ── thread-safe shot queue ────────────────────────────────────────────────

    def push_shot(self, shot: Optional[Shot]) -> None:
        """
        Called from the pipeline thread — never blocks.
        Drops the oldest item if the queue is full (we only care about
        the most recent shot, not a backlog).
        """
        try:
            self._q.put_nowait(shot)
        except queue.Full:
            try:    self._q.get_nowait()
            except queue.Empty: pass
            try:    self._q.put_nowait(shot)
            except queue.Full:  pass

    # ── Win32 click-through ───────────────────────────────────────────────────

    def _apply_clickthrough(self) -> None:
        """
        Make the overlay window completely transparent to mouse clicks.
        WS_EX_TRANSPARENT causes Windows to pass all mouse events to
        whatever window is beneath the overlay.
        """
        try:
            hwnd  = self.root.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, _GWL_EXSTYLE, style | _WS_EX_TRANSPARENT)
            print("[Overlay] click-through applied")
        except Exception as e:
            print(f"[Overlay] click-through failed: {e}")

    # ── control panel ─────────────────────────────────────────────────────────

    def _build_panel(self) -> None:
        """
        Small always-on-top control panel in the top-right corner.
        Separate Toplevel window so it receives mouse events (unlike the
        click-through canvas).
        """
        p = tk.Toplevel(self.root)
        p.title("")
        p.geometry(f"220x95+{max(10, self.x + self.width - 230)}+{max(10, self.y + 10)}")
        p.attributes("-topmost", True)
        p.resizable(False, False)
        p.configure(bg="#0d1b2a")
        p.protocol("WM_DELETE_WINDOW", self.root.destroy)
        p.overrideredirect(True)

        f  = tkfont.Font(family="Consolas", size=9,  weight="bold")
        fs = tkfont.Font(family="Consolas", size=8)

        tk.Label(p, text="8BP  AI", bg="#0d1b2a", fg="#00ffaa", font=f).place(x=8, y=6)

        tk.Button(p, text="✕  Close",  command=self.root.destroy,
                  bg="#c0392b", fg="white", font=f,
                  relief="flat", bd=0, padx=8, pady=3).place(x=8,   y=30)
        tk.Button(p, text="⊙  Toggle", command=self._toggle,
                  bg="#1a6fa0", fg="white", font=f,
                  relief="flat", bd=0, padx=8, pady=3).place(x=110, y=30)

        tk.Label(p, text="My balls:", bg="#0d1b2a", fg="#aaaaaa", font=fs).place(x=8, y=63)
        tk.Button(p, text="Solid",  command=lambda: self._set_type("solid"),
                  bg="#005577", fg="white", font=fs,
                  relief="flat", bd=0, padx=6, pady=2).place(x=72,  y=61)
        tk.Button(p, text="Stripe", command=lambda: self._set_type("stripe"),
                  bg="#774400", fg="white", font=fs,
                  relief="flat", bd=0, padx=6, pady=2).place(x=135, y=61)

        self._type_lbl = tk.Label(p, text="not set",
                                  bg="#0d1b2a", fg="#666666", font=fs)
        self._type_lbl.place(x=8, y=77)

    def _set_type(self, t: str) -> None:
        col = "#00aaff" if t == "solid" else "#ffaa00"
        self._type_lbl.config(text=f"mine: {t}s", fg=col)
        if self.on_type_change:
            self.on_type_change(t)

    def _toggle(self) -> None:
        self._on = not self._on
        if not self._on:
            self.canvas.delete("all")

    # ── render loop ───────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """
        Drain the queue (keep only the latest shot), advance animation tick,
        and ask the renderer to draw. Schedules itself every 16 ms (~60 Hz).
        """
        shot = None
        while True:
            try:    shot = self._q.get_nowait()
            except queue.Empty: break

        self._tick = (self._tick + 1) % 40

        if self._on and self._renderer:
            self._renderer.draw(shot, self._tick)

        self.root.after(16, self._loop)
