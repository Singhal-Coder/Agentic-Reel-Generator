# models/moviepy_params.py

from __future__ import annotations
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field
from pathlib import Path

class Transform(BaseModel):
    """Defines a visual transformation like a zoom or pan."""
    effect: Literal["KenBurns", "CropAndZoom"]
    start_zoom: Optional[float] = 1.0
    end_zoom: Optional[float] = 1.1
    start_pos: Optional[List[float]] = Field(default_factory=lambda: [0.5, 0.5])
    end_pos: Optional[List[float]] = Field(default_factory=lambda: [0.5, 0.5])

class Transition(BaseModel):
    """Defines a transition between clips."""
    type: Literal["fade", "crossfade", "slide_in"]
    duration_sec: float = 0.5

class ClipSpec(BaseModel):
    """Defines a single visual clip on the timeline."""
    asset_id: str
    asset_type: Literal["video", "image"]
    path: Path
    start_at: float
    end_at: float
    duration: float
    in_offset: float = 0.0
    width: Optional[int] = None
    height: Optional[int] = None
    transform: Optional[Transform] = None
    transition_to_next: Optional[Transition] = None

class VideoTrack(BaseModel):
    """A track containing visual clips."""
    track_id: str = "main_video"
    clips: List[ClipSpec]

class AudioTrack(BaseModel):
    """A track containing audio clips."""
    track_id: str
    path: Path
    start_at: float
    volume: float = 1.0

class OverlaySpec(BaseModel):
    """Defines an overlay (text caption or sticker)."""
    type: Literal["text", "sticker"]
    content: Optional[str] = None # For text
    path: Optional[Path] = None # For stickers
    start_at: float
    end_at: float
    position: str = "center" # e.g., "center", "top_left"
    size: Optional[Literal["small", "medium", "large"]] = "medium"
    font_size: Optional[int] = 50
    font_color: str = "white"

class ProjectSpec(BaseModel):
    """Defines overall project settings."""
    aspect_ratio: str
    duration_sec: float
    fps: int = 24
    resolution: List[int] # [width, height]

class MoviePyParams(BaseModel):
    """The final, complete timeline specification for the renderer."""
    project: ProjectSpec
    video_tracks: List[VideoTrack]
    audio_tracks: List[AudioTrack]
    overlays: List[OverlaySpec]
    subtitle_file: Optional[Path] = None