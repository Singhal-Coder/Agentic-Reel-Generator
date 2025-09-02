# agents/moviepy_agent.py
import uuid
from typing import Dict, Any, Tuple, List
import numpy as np
from PIL import Image

from moviepy import (VideoFileClip, ImageClip, 
ColorClip, CompositeVideoClip, AudioFileClip, 
CompositeAudioClip, TextClip)
from moviepy.video.VideoClip import VideoClip as MPVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.audio.fx import MultiplyVolume, AudioFadeOut

from ..models.moviepy_params import MoviePyParams
from ..models.render_params import (
    FinalRenderParams, RenderClipSpec, RenderOverlaySpec
)
from ..utils.logging import logger
from ..config.settings import settings

# Maps abstract sizes to font sizes in pixels
FONT_SIZE_MAP = {
    "small": 40,
    "medium": 60,
    "large": 90
}

# Maps abstract sizes to sticker sizes in pixels (width, height)
STICKER_SIZE_MAP = {
    "small": (150, 150),
    "medium": (250, 250),
    "large": (400, 400)
}

# --- Helper Functions for Translation ---

def _calculate_position(
    position_str: str,
    project_resolution: Tuple[int, int],
    element_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Translates a descriptive position string into absolute (x, y) coordinates.
    """
    proj_w, proj_h = project_resolution
    elem_w, elem_h = element_size

    # Horizontal alignment
    if "left" in position_str:
        x = int(proj_w * 0.05)  # 5% margin
    elif "right" in position_str:
        x = int(proj_w * 0.95 - elem_w) # 5% margin
    else: # center
        x = (proj_w - elem_w) // 2

    # Vertical alignment
    if "top" in position_str:
        y = int(proj_h * 0.1) # 10% margin
    elif "bottom" in position_str:
        y = int(proj_h * 0.9 - elem_h) # 10% margin
    else: # center
        y = (proj_h - elem_h) // 2
        
    return (x, y)


# --- Helper for Ken Burns Effect ---
def create_ken_burns_effect(clip, duration, start_zoom, end_zoom, start_pos, end_pos):
    """
    Apply a Ken Burns effect to a MoviePy clip (video or image).
    - clip: source clip (ImageClip or VideoFileClip)
    - duration: total duration of the produced clip (seconds)
    - start_zoom/end_zoom: zoom factors (e.g., 1.0->no zoom, 1.2->zoomed in)
    - start_pos/end_pos: center positions as ratios (x_ratio, y_ratio) in [0..1]
    """
    def make_frame(t):
        # Get source frame at time t (works for ImageClip and Video clips)
        frame = clip.get_frame(t)
        h, w = frame.shape[:2]

        # Linear interpolation for zoom and position
        zoom = start_zoom + (end_zoom - start_zoom) * (t / duration)
        center_x = w * (start_pos[0] + (end_pos[0] - start_pos[0]) * (t / duration))
        center_y = h * (start_pos[1] + (end_pos[1] - start_pos[1]) * (t / duration))

        # Calculate crop box
        new_w, new_h = int(round(w / zoom)), int(round(h / zoom))
        x1 = max(0, int(round(center_x - new_w / 2)))
        y1 = max(0, int(round(center_y - new_h / 2)))
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        cropped_frame = frame[y1:y2, x1:x2]

        # If crop smaller than expected (edges), pad with black
        ch, cw = cropped_frame.shape[:2]
        if (ch != new_h) or (cw != new_w):
            pad_h = max(0, new_h - ch)
            pad_w = max(0, new_w - cw)
            if frame.ndim == 3:
                cropped_frame = np.pad(
                    cropped_frame,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                cropped_frame = np.pad(
                    cropped_frame,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

        # Resize back to original dimensions with high quality resampling
        img = Image.fromarray(cropped_frame)
        resized_img = img.resize((w, h), Image.Resampling.LANCZOS)
        return np.array(resized_img)

    # Create a new VideoClip from the frame generator
    new_clip = MPVideoClip(make_frame, duration=duration)

    # Preserve audio if present
    if hasattr(clip, "audio") and clip.audio is not None:
        new_clip = new_clip.with_audio(clip.audio)

    # Preserve fps if source provides it (helps write_videofile defaults)
    if hasattr(clip, "fps") and getattr(clip, "fps", None) is not None:
        try:
            new_clip = new_clip.with_fps(clip.fps)
        except Exception:
            new_clip.fps = clip.fps

    return new_clip

# --- Main Rendering Logic ---
def render_video(data: FinalRenderParams):
    project = data.project
    resolution = tuple(project.resolution)
    fps = project.fps
    total_duration = project.duration_sec

    # --- Video Clips ---
    video_clips = []
    for clip_data in data.video_clips:
        path = str(clip_data.path)
        if clip_data.asset_type == 'video':
            # .subclipped is the correct method per docs [cite: 83]
            clip = VideoFileClip(path).subclipped(clip_data.in_offset)
        else: # image
            clip = ImageClip(path)

        # CORRECTED: Consistently use .with_duration() [cite: 83]
        clip = clip.with_duration(clip_data.duration)

        if clip_data.transform and clip_data.transform.effect == 'KenBurns':
            transform = clip_data.transform
            clip = create_ken_burns_effect(
                clip, clip.duration,
                transform.start_zoom, transform.end_zoom,
                transform.start_pos, transform.end_pos
            )
        
        # Chain .with_start() and .resize() for clarity 
        video_clips.append(clip.with_start(clip_data.start_at).resized(resolution))

    # --- Audio Clips ---
    audio_clips = []
    for audio_data in data.audio_clips:
        clip = AudioFileClip(str(audio_data.path))

        if "background_music" in audio_data.track_id:
            clip = clip.subclipped(0, total_duration).with_effects([AudioFadeOut(1.5)])


        clip = clip.with_effects([
            MultiplyVolume(audio_data.volume)
        ])
        
        audio_clips.append(clip.with_start(audio_data.start_at))
    
    final_audio = CompositeAudioClip(audio_clips) if audio_clips else None

    # --- Overlays ---
    overlay_clips = []
    for overlay_data in data.overlays:
        if overlay_data.type == 'text':
            clip = TextClip(
                text=overlay_data.content, 
                font_size=overlay_data.font_size,
                color=overlay_data.font_color, size=(int(resolution[0] * 0.85), None),
                method='caption', text_align='center'
            )
        else:
            clip = ImageClip(str(overlay_data.path), transparent=True).resized(width=overlay_data.size[0])

        clip = clip.with_position(tuple(overlay_data.position))
        clip = clip.with_start(overlay_data.start_at)
        clip = clip.with_duration(overlay_data.end_at - overlay_data.start_at)
        overlay_clips.append(clip)
        
    # --- Subtitles ---
    if data.subtitle_file:
        generator = lambda txt: TextClip(text=txt, font_size=48, color='white', stroke_color='black', stroke_width=2)
        # .with_position() is the modern equivalent for subtitles [cite: 2]
        subtitles = SubtitlesClip(str(data.subtitle_file), make_textclip=generator).with_position(('center', 0.8), relative=True)
        overlay_clips.append(subtitles)

    # --- Final Composition ---
    base_clip = ColorClip(size=resolution, color=(0, 0, 0), duration=total_duration)
    final_clip = CompositeVideoClip([base_clip] + video_clips + overlay_clips, size=resolution)
    
    if final_audio:
        # .with_audio() is the correct modern method [cite: 74]
        final_clip = final_clip.with_audio(final_audio)
        
    final_clip = final_clip.with_duration(total_duration)

    # --- Render and Save ---
    output_path = settings.OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Rendering video to: {output_path}")
    final_clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    logger.info("Rendering complete!")
    return output_path


# --- Main Agent Logic ---

def moviepy_agent_logic(state: Dict[str, Any]) -> Dict[str, FinalRenderParams]:
    """
    Translates the high-level MoviePyParams into a low-level,
    renderer-ready FinalRenderParams object.
    """
    logger.info("--- Executing: MoviePyAgent (Translator) ---")
    timeline: MoviePyParams = state["timeline_json"]
    project_res = (timeline.project.resolution[0], timeline.project.resolution[1])

    # 1. Translate Video Clips (mostly a direct mapping)
    render_video_clips = [
        RenderClipSpec(
            path=clip.path,
            asset_type=clip.asset_type,
            start_at=clip.start_at,
            duration=clip.duration,
            in_offset=clip.in_offset,
            size=(clip.width, clip.height) if clip.width and clip.height else None,
            transform=clip.transform
        ) for track in timeline.video_tracks for clip in track.clips
    ]

    # 2. Translate Overlays (this is where the main translation happens)
    render_overlays: List[RenderOverlaySpec] = []
    for overlay in timeline.overlays:
        if overlay.type == "text":
            font_size = FONT_SIZE_MAP.get(overlay.size, FONT_SIZE_MAP["medium"])
            # Estimate element size for positioning - a real renderer might need a more
            # accurate way to measure text, but this is a good approximation.
            elem_size_estimate = (project_res[0] * 0.8, font_size * 2) # Assume 80% width, 2 lines high
            
            render_overlays.append(RenderOverlaySpec(
                type="text",
                content=overlay.content,
                start_at=overlay.start_at,
                end_at=overlay.end_at,
                position=_calculate_position(overlay.position, project_res, elem_size_estimate),
                font_size=font_size,
                font_color=overlay.font_color
            ))
        elif overlay.type == "sticker":
            sticker_size = STICKER_SIZE_MAP.get(overlay.size, STICKER_SIZE_MAP["medium"])
            render_overlays.append(RenderOverlaySpec(
                type="sticker",
                path=overlay.path,
                start_at=overlay.start_at,
                end_at=overlay.end_at,
                position=_calculate_position(overlay.position, project_res, sticker_size),
                size=sticker_size
            ))

    # 3. Assemble the Final Render Params
    final_params = FinalRenderParams(
        project=timeline.project,
        video_clips=render_video_clips,
        audio_clips=timeline.audio_tracks,
        overlays=render_overlays,
        subtitle_file=timeline.subtitle_file
    )
    logger.info("--- MoviePyAgent: Successfully translated to FinalRenderParams ---")

    output_path = render_video(final_params)
    
    return {"final_json": final_params, "output_path": output_path}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state('final_state.json')
    state.update(moviepy_agent_logic(state))
    save_app_state(state, 'final_state.json')