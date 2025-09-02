from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid
import os

from ..models.specs import CanonicalSpec, ScriptLine, AssetCandidate
from ..utils.logging import logger
from ..config.settings import settings

from elevenlabs.client import ElevenLabs
from mutagen.mp3 import MP3

def _get_voice_id(voice_style: str) -> str:
    voices = {
        "energetic_male": "15CVCzDByBinCIoCblXo",
        "calm_male": "PGoKnSD4gKn2aS99wOR2",
        "energetic_female": "ecp3DWciuUyW7BYM7II1",
        "calm_female": "19STyYD15bswVz51nqLf"
    }
    return voices.get(voice_style, None)

def _elevenlabs_tts_api(
    paragraph: List[str],
    voice_id: str,
    tts_id: str,
    output_dir: Path
) -> Optional[Path]:
    """
    Convert paragraphs to speech and save to a single file.
    Returns Path to saved file on success, or None on failure.
    """
    try:
        elevenlabs = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        request_ids = []
        chosen_ext = ".mp3" 

        file_dir = Path(output_dir)
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / tts_id
        if settings.ELEVENLABS_MODEL_NAME == "eleven_v3":
            with open(file_path, "wb") as f:
                for i, line in enumerate(paragraph):
                    audio = elevenlabs.text_to_speech.convert(
                        text=line,
                        voice_id=voice_id,
                        model_id=settings.ELEVENLABS_MODEL_NAME,
                    )
                    for chunk in audio:
                        if chunk:
                            f.write(chunk)
        else:
            with open (file_path, "wb") as f:
                for i, line in enumerate(paragraph):
                    with elevenlabs.text_to_speech.with_raw_response.convert( 
                        text=line, 
                        voice_id=voice_id, 
                        model_id=settings.ELEVENLABS_MODEL_NAME, 
                        previous_request_ids=request_ids[-3:] 
                    ) as response:
                        rid = response._response.headers.get("request-id")
                        if rid:
                            request_ids.append(rid)
                        else:
                            logger.warning("No request-id found in response headers")
                            return None
                        if i==0:
                            content_type = response._response.headers.get("content-type", "").lower()
                            if "mpeg" in content_type or "mp3" in content_type:
                                chosen_ext = ".mp3"
                            elif "wav" in content_type:
                                chosen_ext = ".wav"
                            elif "ogg" in content_type:
                                chosen_ext = ".ogg"
                        for chunk in response.data:
                            if chunk:
                                f.write(chunk)
        os.rename(file_path, file_path.with_suffix(chosen_ext))
        file_path = file_path.with_suffix(chosen_ext)
        return file_path

    except Exception as exc:
        logger.exception("Failed to save TTS audio: %s", exc)
        return None

def tts_logic(state: Dict[str, Any]) -> Dict[str, List[AssetCandidate]]:
    """
    Generates audio for each line in the script using a TTS service.
    """
    logger.info("--- Executing: TTSAgent ---")
    script: List[ScriptLine] = state["script"]
    spec: CanonicalSpec = state["spec"]
    
    if spec.voice_style and spec.voice_style.value:
        voice_id = _get_voice_id(spec.voice_style.value)
    else:
        voice_id = None
    
    if not voice_id:
        return {"tts_files": []}
    
    tts_output_dir = settings.CACHE_DIR / "tts"
    

    all_lines = [script_line.line for script_line in script]

    tts_id = uuid.uuid4().hex
    audio_file_path = _elevenlabs_tts_api(all_lines, voice_id, tts_id, tts_output_dir)

    logger.info(f"TTS audio file generated at {audio_file_path}")
    if not audio_file_path:
        logger.error("TTS audio file generation failed.")
        return {"tts_file": None}

    path = str(audio_file_path)
    try:
        audio = MP3(path)
        duration = audio.info.length
        tts_file = AssetCandidate(id=f"tts_{tts_id}", type="audio", path=path, duration_sec=duration, score=1.0)
    except Exception as e:
        logger.error(f"Could not read duration from audio file {path}: {e}")
        tts_file = AssetCandidate(id=f"tts_{tts_id}", type="audio", path=path, duration_sec=sum([script_line.estimated_sec for script_line in script]), score=1.0)

    return {"tts_file": tts_file}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state.update(tts_logic(state))
    save_app_state(state)