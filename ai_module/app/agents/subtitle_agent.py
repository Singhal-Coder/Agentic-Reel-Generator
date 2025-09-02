from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

from ..models.specs import ShotEntry
from ..utils.logging import logger
from ..config.settings import settings

def _generate_srt_from_shot_list(shot_list: List[ShotEntry], output_dir: Path) -> Optional[Path]:
    """
    Generates a .srt subtitle file from the script captions and shot timings.
    """
    if not shot_list:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    srt_file_path = output_dir / f"{uuid.uuid4().hex}.srt"
    
    srt_content = []
    for i, shot in enumerate(shot_list):
        if not shot.caption:
            continue
            
        start_time = f"{int(shot.start_sec // 3600):02}:{int((shot.start_sec % 3600) // 60):02}:{int(shot.start_sec % 60):02},{int((shot.start_sec * 1000) % 1000):03}"
        end_time = f"{int(shot.end_sec // 3600):02}:{int((shot.end_sec % 3600) // 60):02}:{int(shot.end_sec % 60):02},{int((shot.end_sec * 1000) % 1000):03}"
        
        srt_block = f"{i + 1}\n{start_time} --> {end_time}\n{shot.caption}\n"
        srt_content.append(srt_block)

    try:
        with open(srt_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))
        logger.info(f"Successfully generated subtitle file: {srt_file_path}")
        return srt_file_path
    except Exception as e:
        logger.error(f"Failed to write SRT file: {e}")
        return None

def subtitle_logic(state: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    """
    The main agent function for generating subtitles.
    """
    logger.info("--- Executing: SubtitleAgent ---")
    shot_list: List[ShotEntry] = state["shot_list"]
    
    subtitle_output_dir = settings.CACHE_DIR / "subtitles"
    
    subtitle_file = _generate_srt_from_shot_list(shot_list, subtitle_output_dir)
    
    return {"subtitle_file": subtitle_file}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state.update(subtitle_logic(state))
    save_app_state(state)