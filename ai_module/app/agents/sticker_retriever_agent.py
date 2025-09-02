from typing import Dict, Any, List
import requests

from ..models.specs import ShotEntry, AssetCandidate
from ..utils.logging import logger
from ..config.settings import settings

GIPHY_API_KEY = settings.GIPHY_API_KEY
GIPHY_STICKER_SEARCH_URL = "https://api.giphy.com/v1/stickers/search"

def _search_giphy_stickers(query: str) -> List[Dict[str, Any]]:
    """(Internal Helper) Performs a search against the GIPHY Sticker Search API."""
    if not GIPHY_API_KEY:
        logger.warning("GIPHY_API_KEY not found in environment variables. Skipping sticker search.")
        return []

    params = {
        "api_key": GIPHY_API_KEY,
        "q": query,
        "limit": 3,
        "rating": "g"
    }

    try:
        response = requests.get(GIPHY_STICKER_SEARCH_URL, params=params)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling GIPHY API for query '{query}': {e}")
        return []

def _parse_giphy_results(results: List[Dict[str, Any]]) -> List[AssetCandidate]:
    """(Internal Helper) Parses GIPHY API results into our AssetCandidate model."""
    candidates = []
    for result in results:
        try:
            image_data = result.get("images", {}).get("fixed_height")
            if not image_data or not image_data.get("webp"):
                continue

            candidate = AssetCandidate(
                id=f"giphy-{result['id']}",
                type="sticker",
                path=image_data["webp"],
                width=int(image_data.get("width", 0)),
                height=int(image_data.get("height", 0)),
                score=0.0
            )
            candidates.append(candidate)
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing GIPHY result: {e} - Skipping item.")
            continue
    return candidates

def sticker_retriever_logic(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The main agent logic for retrieving sticker assets based on the shot list.
    """
    logger.info("--- Executing: StickerRetrieverAgent ---")
    shot_list: List[ShotEntry] = state["shot_list"]
    
    final_sticker_assets = {}

    for shot in shot_list:
        sticker_requests = [
            overlay for overlay in (shot.desired_overlays or []) 
            if overlay.type == "stickers"
        ]
        if not sticker_requests:
            continue
        shot_stickers = []
        for request in sticker_requests:
            if not request.search_tags:
                continue
            
            query = " ".join(request.search_tags)
            logger.info(f"  - Searching for sticker for shot {shot.shot_id}: '{query}'")
            
            search_results = _search_giphy_stickers(query)
            sticker_candidates = _parse_giphy_results(search_results)
            
            if sticker_candidates:
                shot_stickers.append(sticker_candidates[0])
        
        if shot_stickers:
            final_sticker_assets[shot.shot_id] = shot_stickers

    return {"sticker_assets": final_sticker_assets}

if __name__ == "__main__":
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state.update(sticker_retriever_logic(state))
    save_app_state(state)