# models/render_params.py

from __future__ import annotations
from typing import List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel
from .moviepy_params import ProjectSpec, AudioTrack, Transition, Transform

class RenderClipSpec(BaseModel):
    """Defines a single visual clip with precise coordinates."""
    path: Path
    asset_type: str
    start_at: float
    duration: float
    in_offset: float = 0.0
    size: Optional[Tuple[int, int]] = None  # (width, height)
    transform: Optional[Transform] = None
    transition_to_next: Optional[Transition] = None

class RenderOverlaySpec(BaseModel):
    """Defines an overlay with precise coordinates and sizes."""
    type: str
    content: Optional[str] = None
    path: Optional[Path] = None
    start_at: float
    end_at: float
    position: Tuple[int, int]  # (x, y) coordinates
    size: Optional[Tuple[int, int]] = None  # (width, height)
    font_size: Optional[int] = None
    font_color: Optional[str] = "white"

class FinalRenderParams(BaseModel):
    """The final, renderer-ready timeline specification."""
    project: ProjectSpec
    video_clips: List[RenderClipSpec]
    audio_clips: List[AudioTrack]
    overlays: List[RenderOverlaySpec]
    subtitle_file: Optional[Path] = None