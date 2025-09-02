# agents/music_retriever_agent.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, model_validator
import requests
import os

from .llm_utils import call_llm_with_json_output
from ..providers.loader import providers
from ..models.specs import CanonicalSpec, AssetCandidate
from ..utils.logging import logger

# --- Configuration ---
# Assumes these are in your .env file or environment variables
MUSIC_API_BASE_URL = os.getenv("MUSIC_API_BASE_URL", "http://127.0.0.1:8000")
MUSIC_API_KEY = os.getenv("MUSIC_API_KEY")

# --- 1. Pydantic Models for API Interaction ---

class MusicAttributes(BaseModel):
    genres: List[str]
    moods: List[str]
    instruments: List[str]

attributes: Optional[MusicAttributes] = None

class MappedAttributes(BaseModel):
    genre: Optional[str] = Field(None, description="The single best genre from the provided list that matches the creative style.")
    mood: Optional[str] = Field(None, description="The single best mood from the provided list that matches the creative style.")

    @model_validator(mode="after")
    def validate_genre_and_mood(self):
        if self.genre is not None and self.genre not in attributes.genres:
            raise ValueError("Genre is required.")
        if self.mood is not None and self.mood not in attributes.moods:
            raise ValueError("Mood is required.")
        return self

    @model_validator(mode="before")
    def upper_genre_and_mood(self):
        if self.get('genre') is not None:
            self['genre'] = self.get('genre').upper()
        if self.get('mood') is not None:
            self['mood'] = self.get('mood').upper()
        return self

# --- 2. API Interaction Helpers ---

def _get_music_attributes() -> Optional[MusicAttributes]:
    """Fetches the available genres, moods, and instruments from the API."""
    try:
        response = requests.get(f"{MUSIC_API_BASE_URL}/attributes")
        response.raise_for_status()
        return MusicAttributes(**response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not fetch music attributes from API: {e}")
        return None

def _search_tracks(genre: Optional[str], mood: Optional[str]) -> List[Dict[str, Any]]:
    """Searches for tracks using the music service API."""
    if not MUSIC_API_KEY:
        logger.error("MUSIC_API_KEY is set. Cannot search for music.")
        return []
    
    headers = {
        "X-API-Key": MUSIC_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "attributes": {
            "genre": genre,
            "mood": mood
        },
        "use_or_logic": False # We want both mood and genre to match
    }
    try:
        response = requests.post(f"{MUSIC_API_BASE_URL}/tracks/search", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching for music tracks: {e}")
        return []

# --- 3. LLM-Powered Mapping Logic ---

MAPPING_SYSTEM_PROMPT = """
You are an expert Music Supervisor. Your task is to select the best `genre` and `mood` from a predefined list of valid options to match a creative request.

**Your Rules:**
1.  **Analyze the Request:** Read the user's desired `music_style` and the overall `mood` for the video.
2.  **Choose the Best Fit:** From the provided lists of valid `genres` and `moods`, choose the ONE `genre` and ONE `mood` that are the closest match to the user's request.
3.  **Strictly Adhere to Lists:** You MUST only choose values that are present in the provided lists. Do not invent new ones.
"""

def _get_mapped_attributes(music_style: str, mood: str, attributes: MusicAttributes) -> MappedAttributes:
    """Uses an LLM to map our spec's style to the API's attributes."""
    
    
    
    llm_user_prompt = f"""
Please choose the best `genre` and `mood` from the valid options below to match the creative request.

**Creative Request:**
* **Music Style:** "{music_style}"
* **Overall Reel Mood:** "{mood}"

**Valid Options (Choose from these lists):**
* **Valid Genres:** {attributes.genres}
* **Valid Moods:** {attributes.moods}
"""

    llm_output = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=MAPPING_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        output_model=MappedAttributes
    )
    
    return MappedAttributes(**llm_output)

# --- 4. Main Agent Logic ---

def music_retriever_logic(state: Dict[str, Any]) -> Dict[str, Optional[AssetCandidate]]:
    """
    The main agent logic for finding and selecting a music track.
    """
    global attributes
    logger.info("--- Executing: MusicRetrieverAgent ---")
    spec: CanonicalSpec = state["spec"]
    
    if not spec.music_style or not spec.music_style.value:
        logger.info("No music style specified in the spec. Skipping music retrieval.")
        return {"music_asset": None}

    # Step 1: Get available attributes from the API
    attributes = _get_music_attributes()
    if not attributes:
        return {"music_asset": None} # Exit if API call fails

    # Step 2: Use LLM to map our spec to the API's attributes
    logger.info("--- Mapping creative style to Genre and Mood ---")
    music_style = spec.music_style.value if spec.music_style else "any"
    mood = spec.mood.value if spec.mood else "any"
    mapped_attrs = _get_mapped_attributes(music_style, mood, attributes)
    logger.info(f"  - Mapped creative style to Genre: '{mapped_attrs.genre}', Mood: '{mapped_attrs.mood}'")

    # Step 3: Search for tracks with the mapped attributes
    tracks = _search_tracks(mapped_attrs.genre, mapped_attrs.mood)
    if not tracks:
        logger.warning("No tracks found matching the criteria.")
        return {"music_asset": None}

    # Step 4: Rank and select the best track.
    # For now, we'll simply select the first result as the "best" one.
    # A more advanced ranker could be added here later (e.g., to match duration).
    best_track = tracks[0]
    logger.info(f"  - Selected music track: '{best_track['title']}' by {best_track.get('artist',{}).get('name','Unknown Artist')}")
    download_path = f"{MUSIC_API_BASE_URL}/tracks/{best_track['trackId']}/download"
    music_asset = AssetCandidate(
        id=f"youtube-{best_track['trackId']}",
        type="audio",
        path=download_path,
        duration_sec=best_track.get("duration",{}).get("seconds",None),
        score=1.0
    )

    return {"music_asset": music_asset}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state = music_retriever_logic(state)
    save_app_state(state)