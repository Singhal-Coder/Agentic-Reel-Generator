# agents/timeline_builder_agent.py

from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio

from ..models.specs import CanonicalSpec, ShotEntry, AssetCandidate
from ..models.moviepy_params import (
    MoviePyParams, ProjectSpec, VideoTrack, AudioTrack, 
    ClipSpec, OverlaySpec, Transform
)
from ..utils.logging import logger
from ..utils.cache_manager import AssetCacheManager

async def _download_assets(state: Dict[str, Any]) -> Dict[str, Path]:
    """
    (Internal Helper) Takes all selected asset URLs and downloads them using
    the AssetCacheManager. This runs concurrently for speed.
    """
    logger.info("--- Downloading all required assets ---")
    local_paths = {}
    
    async with AssetCacheManager() as cache:
        tasks = []
        
        urls_to_download = {}
        
        # Visual assets
        visual_assets = state.get("visual_assets", {})
        for shot_id, candidates in visual_assets.items():
            if candidates: urls_to_download[candidates[0].id] = (candidates[0].path, candidates[0].type)
        
        # Music asset
        music_asset = state.get("music_asset")
        if music_asset: urls_to_download[music_asset.id] = (music_asset.path, music_asset.type)
        
        # Sticker assets
        sticker_assets = state.get("sticker_assets", {})

        for shot_id, stickers in sticker_assets.items():
            for sticker in stickers: 
                urls_to_download[sticker.id] = (sticker.path, sticker.type)

        # Create download tasks
        async def download_task(asset_id, url, content_type):
            path = await cache.get_asset(url, content_type)
            if path:
                local_paths[asset_id] = path
                
        for asset_id, (url, content_type) in urls_to_download.items():
            if url: tasks.append(download_task(asset_id, url, content_type))
        
        await asyncio.gather(*tasks)

    logger.info(f"--- Successfully downloaded {len(local_paths)} assets ---")
    return local_paths

def _select_final_visual(shot: ShotEntry, candidates: List[AssetCandidate]) -> Optional[AssetCandidate]:
    """(Internal Helper) Selects the best asset with fallback logic."""
    shot_duration = shot.end_sec - shot.start_sec
    # Try to find a suitable video first
    for asset in candidates:
        if asset.type == "video" and asset.duration_sec and asset.duration_sec >= shot_duration:
            return asset
    # If no suitable video, fall back to the best image
    for asset in candidates:
        if asset.type == "image":
            return asset
    return None # No suitable asset found

def timeline_builder_logic(state: Dict[str, Any]) -> Dict[str, MoviePyParams]:
    """
    The main deterministic agent logic for assembling the final timeline.
    """
    logger.info("--- Executing: TimelineBuilderAgent ---")
    
    # Run the async download function
    local_asset_paths = asyncio.run(_download_assets(state))
    
    spec: CanonicalSpec = state["spec"]
    shot_list: List[ShotEntry] = state["shot_list"]
    tts_asset: AssetCandidate = state.get("tts_file")
    # 1. Build Project Spec
    requested_duration = spec.duration_sec.value if spec.duration_sec else 30.0
    tts_duration = (tts_asset.duration_sec+1) if tts_asset else 0.0
    final_duration = max(requested_duration, tts_duration)
    aspect_ratio = spec.aspect_ratio.value if spec.aspect_ratio else "9:16"
    width, height = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)
    project_spec = ProjectSpec(
        aspect_ratio=aspect_ratio,
        duration_sec=final_duration,
        resolution=[width, height]
    )

    # 2. Build Video Track
    video_clips: List[ClipSpec] = []
    for shot in shot_list:
        candidates = state.get("visual_assets", {}).get(shot.shot_id, [])
        final_asset = _select_final_visual(shot, candidates)
        
        if not final_asset or final_asset.id not in local_asset_paths:
            logger.warning(f"No suitable or downloaded asset for shot {shot.shot_id}. Skipping.")
            continue

        clip = ClipSpec(
            asset_id=final_asset.id,
            asset_type=final_asset.type,
            path=local_asset_paths[final_asset.id],
            start_at=shot.start_sec,
            end_at=shot.end_sec,
            duration=shot.end_sec - shot.start_sec,
            width=final_asset.width,
            height=final_asset.height, 
            transform=Transform(effect="KenBurns") if final_asset.type == "image" else None
        )
        video_clips.append(clip)
        
    video_track = VideoTrack(clips=video_clips)

    # 3. Build Audio Tracks
    audio_tracks: List[AudioTrack] = []
    # Music
    music_asset = state.get("music_asset")
    if music_asset and music_asset.id in local_asset_paths:
        audio_tracks.append(AudioTrack(
            track_id="background_music",
            path=local_asset_paths[music_asset.id],
            start_at=0.0,
            volume=0.5 # Default volume
        ))
    # TTS
    tts_asset = state.get("tts_file")
    if tts_asset:
        audio_tracks.append(AudioTrack(
            track_id="voice_over",
            path=tts_asset.path,
            start_at=0.0,
            volume=1.0
        ))

    # 4. Build Overlays
    overlays: List[OverlaySpec] = []
    for shot in shot_list:
        # A. Always add the main caption for the shot
        if shot.caption:
            overlays.append(OverlaySpec(
                type="text",
                content=shot.caption,
                start_at=shot.start_sec,
                end_at=shot.end_sec,
                position=shot.caption_position or "bottom",
                size="medium"
            ))
    
        # B. Add any supplemental overlays like stickers or title cards
        if shot.desired_overlays:
            sticker_assets_for_shot = state.get("sticker_assets", {}).get(shot.shot_id, [])
            sticker_index = 0
    
            for overlay_req in shot.desired_overlays:
                if overlay_req.type == "stickers":
                    if sticker_index < len(sticker_assets_for_shot):
                        sticker = sticker_assets_for_shot[sticker_index]
                        if sticker.id in local_asset_paths:
                            overlays.append(OverlaySpec(
                                type="sticker",
                                path=local_asset_paths[sticker.id],
                                start_at=shot.start_sec,
                                end_at=shot.end_sec,
                                position=overlay_req.position or "center",
                                size=overlay_req.size or "medium"
                            ))
                        sticker_index += 1
                # Handle other text-based overlays like title_card or hashtags
                else:
                    content = " ".join(overlay_req.search_tags)
                    overlays.append(OverlaySpec(
                        type="text",
                        content=content,
                        start_at=shot.start_sec,
                        end_at=shot.end_sec,
                        position=overlay_req.position or "center",
                        size=overlay_req.size or "large"
                    ))

    # 5. Assemble Final Params
    timeline_json = MoviePyParams(
        project=project_spec,
        video_tracks=[video_track],
        audio_tracks=audio_tracks,
        overlays=overlays,
        subtitle_file=state.get("subtitle_file")
    )
    
    logger.info("--- TimelineBuilderAgent: Successfully assembled MoviePyParams JSON ---")
    return {"timeline_json": timeline_json}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state('reel_on_coding_and_ai.json')
    state.update(timeline_builder_logic(state))
    save_app_state(state, 'reel_on_coding_and_ai.json')