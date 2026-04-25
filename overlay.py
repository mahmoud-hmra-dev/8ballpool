"""
Transparent click-through overlay  —  shot visualisation
=========================================================

Drawing layers (bottom → top)
  1  Aim line       cue → ghost → extended  (white dashed)
  2  Cue path       ghost → wall bounces    (segments, blue→cyan fading)
  3  Target path    target → pocket         (colored arrow)
  4  Pocket ring    pulsing circle
  5  Ghost circle   + crosshair
  6  Cue highlight  yellow ring
  7  Target ring    color-coded ring
  8  HUD            difficulty / miss / angle / source
"""
import math
import ctypes
import queue
import tkinter as tk
from tkinter import font as tkfont

_GWL_EXSTYLE       = -20
_WS_EX_TRANSPARENT = 0x00000020
_CHROMA            = "black"


# ── colour helpers ─────────────────────────────────────────────────────────────

def _shot_col(miss, angle):
    if miss < 12 and angle < 30: return "#00FF88"   # easy  – green
    if miss < 35 and angle < 55: return "#FFD700"   # medium – gold
    return "#FF4444"                                  # hard  – red

def _difficulty(miss, angle):
    if miss < 12 and angle < 30: return "EASY"
    if miss < 35 and angle < 55: return "MEDIUM"
    return "HARD"

# Segment colours for cue-ball wall bounces (fades from bright to dim)
_BOUNCE_COLS = ["#44AAFF", "#3399EE", "#2277CC", "#1155AA", "#003388"]


class TransparentOverlay:
    def __init__(self, x, y, width, height):
        self.x = x;  self.y = y
        self.width = width;  self.height = height
        self._q   = queue.Queue(maxsize=3)
        self.root = None
        self.canvas = None
        self._on  = True
        self._tick = 0
        self.on_type_change = None   # callback(t) wired by main.py → det.set_my_type

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(
            f"{self.width}x{self.height}+{max(0,self.x)}+{max(0,self.y)}")
        self.root.configure(bg=_CHROMA)

        self.canvas = tk.Canvas(
            self.root, bg=_CHROMA, highlightthickness=0,
            width=self.width, height=self.height)
        self.canvas.pack(fill="both", expand=True)

        self.root.update_idletasks()
        self.root.attributes("-transparentcolor", _CHROMA)
        self.root.attributes("-topmost", True)
        self.root.lift()

        self.root.after(300, self._clickthrough)
        self._panel()
        self.root.after(16, self._loop)

    def push_shot(self, shot):
        try:
            self._q.put_nowait(shot)
        except queue.Full:
            try:    self._q.get_nowait()
            except queue.Empty: pass
            try:    self._q.put_nowait(shot)
            except queue.Full:  pass

    def run(self):
        self.root.mainloop()

    # ── Win32 click-through ───────────────────────────────────────────────────

    def _clickthrough(self):
        try:
            hwnd  = self.root.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, _GWL_EXSTYLE, style | _WS_EX_TRANSPARENT)
            print("[Overlay] click-through OK")
        except Exception as e:
            print(f"[Overlay] {e}")

    # ── control panel ─────────────────────────────────────────────────────────

    def _panel(self):
        p = tk.Toplevel(self.root)
        p.title("")
        p.geometry(f"220x95+{max(10,self.x+self.width-230)}+{max(10,self.y+10)}")
        p.attributes("-topmost", True)
        p.resizable(False, False)
        p.configure(bg="#0d1b2a")
        p.protocol("WM_DELETE_WINDOW", self.root.destroy)
        p.overrideredirect(True)
        f  = tkfont.Font(family="Consolas", size=9,  weight="bold")
        fs = tkfont.Font(family="Consolas", size=8)
        tk.Label(p, text="8BP  AI", bg="#0d1b2a", fg="#00ffaa", font=f
                 ).place(x=8, y=6)
        tk.Button(p, text="✕  Close", command=self.root.destroy,
                  bg="#c0392b", fg="white", font=f,
                  relief="flat", bd=0, padx=8, pady=3).place(x=8,  y=30)
        tk.Button(p, text="⊙  Toggle", command=self._toggle,
                  bg="#1a6fa0", fg="white", font=f,
                  relief="flat", bd=0, padx=8, pady=3).place(x=110, y=30)
        # ── My balls ──────────────────────────────────────────────────────────
        tk.Label(p, text="My balls:", bg="#0d1b2a", fg="#aaaaaa", font=fs
                 ).place(x=8, y=63)
        tk.Button(p, text="Solid",  command=lambda: self._set_type("solid"),
                  bg="#005577", fg="white", font=fs,
                  relief="flat", bd=0, padx=6, pady=2).place(x=72, y=61)
        tk.Button(p, text="Stripe", command=lambda: self._set_type("stripe"),
                  bg="#774400", fg="white", font=fs,
                  relief="flat", bd=0, padx=6, pady=2).place(x=135, y=61)
        self._type_lbl = tk.Label(p, text="not set",
                                   bg="#0d1b2a", fg="#666666", font=fs)
        self._type_lbl.place(x=8, y=77)

    def _set_type(self, t: str):
        col = "#00aaff" if t == "solid" else "#ffaa00"
        self._type_lbl.config(text=f"mine: {t}s", fg=col)
        if self.on_type_change:
            self.on_type_change(t)

    def _toggle(self):
        self._on = not self._on
        if not self._on:
            self.canvas.delete("all")

    # ── draw loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        shot = None
        while True:
            try:    shot = self._q.get_nowait()
            except queue.Empty: break
        self._tick = (self._tick + 1) % 40
        if self._on:
            self._draw(shot)
        self.root.after(16, self._loop)

    # ═══════════════════════════════════════════════════════════════════════════
    # RENDERING
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw(self, shot):
        c = self.canvas
        c.delete("all")
        if not shot:
            return

        cue    = shot.get("cue_pos")
        ghost  = shot.get("ghost_pos")
        target = shot.get("target_pos")
        pocket = shot.get("pocket")
        cpath  = shot.get("cue_path", [])
        R      = max(8, shot.get("ball_radius", 13))
        miss   = shot.get("miss_px",   999)
        angle  = shot.get("cut_angle",  90)
        score  = shot.get("score",       0)
        source = shot.get("source",   "?")

        col  = _shot_col(miss, angle)
        diff = _difficulty(miss, angle)

        # ── 1. AIM LINE  cue → ghost → extended ──────────────────────────────
        if cue and ghost:
            dx, dy = ghost[0]-cue[0], ghost[1]-cue[1]
            m  = max(1.0, math.hypot(dx, dy))
            nx, ny = dx/m, dy/m

            # Line from just in front of cue ball all the way past ghost
            sx = int(cue[0] + nx*(R+3))
            sy = int(cue[1] + ny*(R+3))
            # Extended 900 px past ghost
            ex = int(ghost[0] + nx*900)
            ey = int(ghost[1] + ny*900)
            # Draw in two segments for visual clarity:
            # a) cue → ghost: solid white (path the cue ball travels)
            c.create_line(sx, sy, int(ghost[0]), int(ghost[1]),
                          fill="white", width=2)
            # b) ghost → extended: dashed white (aim direction beyond)
            c.create_line(int(ghost[0]), int(ghost[1]), ex, ey,
                          fill="white", width=1, dash=(10, 7))

        # ── 2. CUE-BALL PATH AFTER IMPACT  (wall bounces) ────────────────────
        if cpath and len(cpath) >= 2:
            for i in range(len(cpath) - 1):
                p0 = cpath[i];   p1 = cpath[i+1]
                seg_col = _BOUNCE_COLS[min(i, len(_BOUNCE_COLS)-1)]
                width   = max(1, 3 - i)
                dash    = (8, 4) if i == 0 else (5, 5)
                c.create_line(int(p0[0]), int(p0[1]),
                              int(p1[0]), int(p1[1]),
                              fill=seg_col, width=width, dash=dash)
            # Arrow on the last segment
            if len(cpath) >= 2:
                p0 = cpath[-2];  p1 = cpath[-1]
                c.create_line(int(p0[0]), int(p0[1]),
                              int(p1[0]), int(p1[1]),
                              fill=_BOUNCE_COLS[0], width=2,
                              arrow=tk.LAST, arrowshape=(10,12,4))

        # ── 3. TARGET BALL PATH → POCKET  (coloured arrow) ───────────────────
        if target and pocket:
            c.create_line(int(target[0]), int(target[1]),
                          int(pocket[0]),  int(pocket[1]),
                          fill=col, width=3, dash=(10,5),
                          arrow=tk.LAST, arrowshape=(15,18,6))

        # ── 4. POCKET RING  (pulsing) ─────────────────────────────────────────
        if pocket:
            px, py  = int(pocket[0]), int(pocket[1])
            pr      = 16 + int(5 * math.sin(self._tick * math.pi / 20))
            # Outer glow
            c.create_oval(px-pr-3, py-pr-3, px+pr+3, py+pr+3,
                          outline=col, width=1)
            # Main ring
            c.create_oval(px-pr, py-pr, px+pr, py+pr,
                          outline=col, width=3)
            # Inner dot
            c.create_oval(px-4, py-4, px+4, py+4, fill=col, outline="")

        # ── 5. GHOST BALL  circle + crosshair ────────────────────────────────
        if ghost:
            gx, gy = int(ghost[0]), int(ghost[1])
            # Outer dashed outline
            c.create_oval(gx-R, gy-R, gx+R, gy+R,
                          outline="white", width=2, dash=(4,3))
            # Crosshair
            c.create_line(gx-R+2, gy, gx+R-2, gy, fill="white", width=1)
            c.create_line(gx, gy-R+2, gx, gy+R-2, fill="white", width=1)

        # ── 6. CUE BALL HIGHLIGHT  yellow ring ────────────────────────────────
        if cue:
            cx2, cy2 = int(cue[0]), int(cue[1])
            c.create_oval(cx2-R-5, cy2-R-5, cx2+R+5, cy2+R+5,
                          outline="#FFFF00", width=3)

        # ── 7. TARGET BALL HIGHLIGHT  colour-coded ring ───────────────────────
        if target:
            tx2, ty2 = int(target[0]), int(target[1])
            c.create_oval(tx2-R-5, ty2-R-5, tx2+R+5, ty2+R+5,
                          outline=col, width=3)

        # ── 8. HUD ────────────────────────────────────────────────────────────
        src_lbl = "AI" if source == "yolo" else "CALC"
        hud = (f"  [{src_lbl}]  {diff}   "
               f"miss={miss:.0f}px   cut={angle:.0f}°   "
               f"score={score:.3f}")
        c.create_rectangle(5, 5, 400, 29, fill="#111111", outline=col, width=1)
        c.create_text(12, 17, anchor="w", text=hud,
                      fill=col, font=("Consolas", 11, "bold"))
