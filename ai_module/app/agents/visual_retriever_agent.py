# agents/visual_retriever_agent.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import requests
import re
import math

from .llm_utils import call_llm_with_json_output
from ..providers.loader import providers
from ..models.specs import CanonicalSpec, ShotEntry, OverlayRequest, AssetCandidate
from ..utils.logging import logger
from ..config.settings import settings

# --- 1. Define Pydantic Model for Query Generation Output ---
class GeneratedQueries(BaseModel):
    queries: List[str] = Field(..., description="A list of 4-6 diverse, specific, and high-quality search queries optimized for a stock media API like Pexels.")

# --- 2. Create the System Prompt ---
QUERY_GENERATION_SYSTEM_PROMPT = """
You are an expert creative director and stock footage researcher. You have a deep understanding of how to find the perfect cinematic visuals by crafting expert-level search queries.

**Your Task:**
Take a high-level shot description (`ShotEntry`) and expand it into a list of 4-6 diverse and specific search queries. These queries will be used to search for both photos and videos on the Pexels API.

**Your Process:**
1.  **Analyze the Input:** Carefully read the `desired_asset_tags`, `mood`, `pacing`, and other details from the provided shot description and overall spec.
2.  **Brainstorm Diverse Angles:** Think of different ways to visually represent the core idea. Do not just rephrase the tags; add cinematic and technical terms to find professional-quality assets.
3.  **Craft High-Quality Queries:** Each query should be a short phrase (3-7 words). Include keywords related to:
    * **Subject & Action:** (e.g., "woman jogging", "water pouring")
    * **Cinematic Style:** (e.g., "cinematic", "slow motion", "dolly shot")
    * **Lighting & Mood:** (e.g., "golden hour", "dramatic lighting", "soft focus")
    * **Composition:** (e.g., "close up", "wide angle", "over the shoulder")
    * **Quality:** (e.g., "4k", "professional footage")
4.  **Ensure Diversity:** The list of queries should be varied to maximize the chance of finding a great asset.

Your output must be a JSON object conforming to the `GeneratedQueries` schema.
"""

# --- 3. The Main Logic for this Step ---
def _generate_queries_for_shot(shot: ShotEntry, spec: CanonicalSpec) -> List[str]:
    """
    (Internal Helper) Takes a single shot and generates a list of search queries.
    This is the "Query Generation Agent" part of the workflow.
    """
    logger.info(f"--- Generating queries for shot: {shot.shot_id} ---")
    
    # We provide the LLM with both the specific shot and the overall spec for context
    llm_user_prompt = f"""
Please generate search queries for the following shot, keeping the overall project spec in mind.

**Overall Project Spec (for context):**
---
{spec.model_dump_json(indent=2)}
---

**Specific Shot to Generate Queries For:**
---
{shot.model_dump_json(indent=2)}
---
"""

    llm_output = call_llm_with_json_output(
        llm_provider=providers.llm,
        system_prompt=QUERY_GENERATION_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        output_model=GeneratedQueries
    )

    queries = llm_output.get("queries", [])
    logger.info(f"  - Generated queries: {queries}")
    return queries

PEXELS_API_KEY = settings.PEXEL_API_KEY
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

def _search_pexels(query: str, is_media_video: bool, spec: CanonicalSpec) -> List[Dict[str, Any]]:
    """(Internal Helper) Performs a search against the Pexels API."""
    if not PEXELS_API_KEY:
        logger.error("PEXELS_API_KEY not found in environment variables.")
        return []

    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": 3,
        "orientation": "portrait" if spec.aspect_ratio and spec.aspect_ratio.value == "9:16" else "landscape"
    }
    
    url = PEXELS_VIDEO_URL if is_media_video else PEXELS_PHOTO_URL
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("videos", []) if is_media_video else data.get("photos", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Pexels API: {e}")
        return []

def _execute_searches(queries: List[str], spec: CanonicalSpec) -> Dict[str, List[Dict[str, Any]]]:
    """
    (Internal Helper) Takes a list of queries, searches Pexels for both
    photos and videos, and returns a list of AssetCandidate objects.
    This is the "Pexels Tool Executor" part of the workflow.
    """
    logger.info("--- Executing Pexels searches ---")
    all_videos = {} 
    all_photos = {}

    for query in queries:
        logger.info(f"  - Searching for: '{query}'")
        # Search for videos
        video_results = _search_pexels(query, True, spec)
        
        # Search for photos
        photo_results = _search_pexels(query, False, spec)

        for video in video_results:
            all_videos[video['id']] = video
        for photo in photo_results:
            all_photos[photo['id']] = photo
        
    
    logger.info(f"Photos: {len(all_photos)}, Videos: {len(all_videos)}")
    return {
        "videos": list(all_videos.values()),
        "photos": list(all_photos.values())
    }



def rank_shot_assets(
    shot: ShotEntry,
    all_candidates: Dict[str, List[Dict[str, Any]]],
    top_k: Optional[int] = None,
    desired_width: Optional[int] = None,
    desired_height: Optional[int] = None,
    prefer_hls: bool = True,
    reel_mood: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Rank assets (images + videos) for a single ShotEntry and pick best video_file for videos.
    Updated to use the provided ShotEntry fields (shot_type, setting_description, time_of_day,
    desired_overlays). No explicit_asset_id used (per your update).
    Parameters:
      - shot: ShotEntry object
      - all_candidates: Dict[str, List[Dict[str, Any]]]
      - top_k: limit results
      - desired_width/desired_height: preferred playback/export resolution (optional)
      - prefer_hls: prefer .m3u8 when picking video_file
      - reel_mood: optional global mood for the reel (affects scoring marginally)
    Returns: sorted list of dicts:
      {
        "asset": <original asset dict>,
        "asset_id": <id or url>,
        "type": "video"|"image",
        "score": float,
        "reason": str,
        "chosen_video_file": { ... } | None
      }
    """

    # ---- helpers ----
    def _safe_lower(x):
        """Return lower-case string for x, or empty string for None / non-str safely."""
        if isinstance(x, str):
            return x.lower()
        if x is None:
            return ""
        try:
            return str(x).lower()
        except Exception:
            return ""

    def tokenize(text: Optional[str]):
        if not text:
            return set()
        t = text.lower()
        t = re.sub(r"[^\w\s#-]", " ", t)
        return set(tok for tok in t.split() if tok)

    def jaccard_list(a: List[str], b: List[str]):
        sa = set(x.lower() for x in (a or []))
        sb = set(x.lower() for x in (b or []))
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def token_overlap_score(text1: Optional[str], text2: Optional[str]):
        t1 = tokenize(text1)
        t2 = tokenize(text2)
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1)

    def duration_fit_score(shot_len: float, asset_duration: Optional[float]):
        if asset_duration is None or asset_duration <= 0:
            return 0.0
        ratio = min(shot_len, asset_duration) / max(shot_len, asset_duration)
        return 0.5 + 0.5 * ratio

    def motion_compatibility(desired_motion: Optional[str], asset_type: str):
        if not desired_motion:
            return 0.8
        dm = desired_motion.lower()
        if dm == "static":
            return 0.8 if asset_type == "image" else 0.2
        if dm in ("pan", "zoom"):
            return 1.0 if asset_type == "video" else 0.85
        if dm == "slow_motion":
            return 1.0 if asset_type == "video" else 0.0
        return 0.5

    def resolution_score(asset_w: Optional[int], asset_h: Optional[int]):
        if asset_w and asset_h:
            area = asset_w * asset_h
            max_area = 4096 * 2160
            score = math.log1p(area) / math.log1p(max_area)
            return max(0.0, min(1.0, score))
        return 0.5

    # map time_of_day into likely tags to match against asset tags/alt text
    TIME_OF_DAY_TAGS = {
        "day": ["day", "daytime", "sunny"],
        "night": ["night", "nighttime", "dark", "lights"],
        "golden_hour": ["golden", "golden hour", "sunset", "warm"],
        "blue_hour": ["blue hour", "twilight", "cold"],
        "dawn": ["dawn", "sunrise", "early morning"],
        "dusk": ["dusk", "sunset", "evening"],
    }

    # shot_type biasing (weights modifiers)
    SHOT_TYPE_WEIGHT_PRESETS = {
        "close_up": {"tag": 0.25, "motion": 0.15, "duration": 0.15, "caption": 0.20, "resolution": 0.25},
        "medium_shot": {"tag": 0.35, "motion": 0.20, "duration": 0.20, "caption": 0.10, "resolution": 0.15},
        "wide_shot": {"tag": 0.45, "motion": 0.15, "duration": 0.20, "caption": 0.05, "resolution": 0.15},
        "drone_shot": {"tag": 0.50, "motion": 0.10, "duration": 0.15, "caption": 0.05, "resolution": 0.20},
        "pov": {"tag": 0.35, "motion": 0.25, "duration": 0.20, "caption": 0.10, "resolution": 0.10},
        None: {"tag": 0.40, "motion": 0.20, "duration": 0.20, "caption": 0.10, "resolution": 0.10},
    }

    def choose_video_file(video_asset: Dict[str, Any], target_w: Optional[int], target_h: Optional[int],
                          prefer_hls_local: bool, required_min_fps: Optional[int] = None):
        files = video_asset.get("video_files") or []
        if not files:
            return None

        # Prefer explicit HLS entries when requested
        if prefer_hls_local:
            for f in files:
                link = _safe_lower(f.get("link"))
                if link.endswith(".m3u8") or (_safe_lower(f.get("file_type")).find("m3u8") >= 0) or _safe_lower(f.get("quality")) == "hls":
                    return f

        # consider mp4 candidates first
        mp4s = [f for f in files if _safe_lower(f.get("file_type")).endswith("mp4") or _safe_lower(f.get("link")).endswith(".mp4")]
        candidates = mp4s if mp4s else list(files)

        # filter by fps if slow_motion is required (prefer those >= required_min_fps)
        if required_min_fps:
            high_fps = [f for f in candidates if (f.get("fps") or 0) >= required_min_fps]
            if high_fps:
                candidates = high_fps

        # scoring: prefer files that meet/exceed target resolution (closest above),
        # else prefer the largest area available
        def file_score(f):
            fw = f.get("width")
            fh = f.get("height")
            fps = f.get("fps") or 0
            if fw and fh and target_w and target_h:
                # prefer >=target with minimal oversize; if smaller, penalize but not disqualify
                if fw >= target_w and fh >= target_h:
                    # negative absolute oversize (smaller is better)
                    return 1_000_000 - ((fw - target_w) ** 2 + (fh - target_h) ** 2) - int(1000 / max(1, fps))
                else:
                    # penalize undersize but keep viable
                    return -((target_w - fw) ** 2 + (target_h - fh) ** 2) - int(2000 / max(1, fps))
            # fallback: use area
            if fw and fh:
                return fw * fh + int(fps * 10)
            # if dimensions missing, fallback modest score based on fps only
            return int(fps * 10)

        chosen = sorted(candidates, key=file_score, reverse=True)[0]
        return chosen

    # ---- main ----
    shot_len = max(0.01, (shot.end_sec or 0) - (shot.start_sec or 0))
    desired_tags: List[str] = shot.desired_asset_tags or []
    desired_motion: Optional[str] = shot.desired_motion
    caption: str = shot.caption or ""
    shot_type = shot.shot_type
    setting_description: Optional[str] = shot.setting_description
    time_of_day: Optional[str] = shot.time_of_day
    overlays: List[OverlayRequest] = shot.desired_overlays or []

    # determine base weights based on shot_type
    weights = SHOT_TYPE_WEIGHT_PRESETS.get(shot_type, SHOT_TYPE_WEIGHT_PRESETS[None])

    results: List[Dict[str, Any]] = []

    # set min_fps if slow_motion requested; else a gentle preference for >=24
    required_min_fps = 60 if (desired_motion and desired_motion.lower() == "slow_motion") else None

    # precompute time_of_day tag set
    tod_tags = []
    if time_of_day:
        tod_tags = TIME_OF_DAY_TAGS.get(time_of_day, [])

    # overlay search tags aggregated (if overlays provided)
    overlay_search_terms = []
    for ov in overlays:
        if isinstance(ov.search_tags, list):
            overlay_search_terms.extend([t.lower() for t in ov.search_tags])

    
    candidate_videos = all_candidates.get("videos") or []
    candidate_photos = all_candidates.get("photos") or []

    def process_asset(asset: Dict[str, Any], asset_type: str):
        alt_text = asset.get("alt") or asset.get("description") or asset.get("url") or ""
        asset_tags: List[str] = []
        raw_tags = asset.get("tags") or []
        if isinstance(raw_tags, list):
            processed = []
            for t in raw_tags:
                if isinstance(t, dict):
                    processed.append(t.get("title") or t.get("name") or "")
                else:
                    processed.append(str(t))
            asset_tags = [x for x in processed if x]
        else:
            # fallback to tokenized alt_text
            asset_tags = list(tokenize(alt_text))

        # dims and duration
        aw = asset.get("width")
        ah = asset.get("height")
        try:
            aw = int(aw) if aw else None
            ah = int(ah) if ah else None
        except Exception:
            aw = None
            ah = None
        duration = asset.get("duration") if asset_type == "video" else None

        # sub-scores
        tag_s = jaccard_list(desired_tags, asset_tags)
        motion_s = motion_compatibility(desired_motion, asset_type)
        dur_s = (0.5 if asset_type == "image" else duration_fit_score(shot_len, duration))
        cap_s = token_overlap_score(caption, alt_text + " " + " ".join(asset_tags))
        res_s = resolution_score(aw, ah)

        # setting description match (token overlap)
        setting_s = token_overlap_score(setting_description or "", alt_text + " " + " ".join(asset_tags))

        # time_of_day match: measure overlap with mapped tags
        time_s = 0.0
        if tod_tags:
            combined_text_tokens = set(tokenize(alt_text)) | set([t.lower() for t in asset_tags])
            time_s = len(combined_text_tokens & set(tod_tags)) / max(1, len(set(tod_tags)))

        # overlay compatibility: prefer assets whose tags don't conflict with overlay (very soft boost)
        # Simple heuristic: if overlay search tags present in asset_tags, give small boost (assets semantically related)
        overlay_s = 0.0
        if overlay_search_terms:
            aset = set([t.lower() for t in asset_tags]) | tokenize(alt_text)
            overlay_matches = aset & set(overlay_search_terms)
            overlay_s = len(overlay_matches) / max(1, len(set(overlay_search_terms)))

        # mood match (reel_mood) - optional small boost if reel_mood token exists in asset tags/alt
        mood_s = 0.0
        if reel_mood:
            aset = set([t.lower() for t in asset_tags]) | tokenize(alt_text)
            mood_s = 1.0 if reel_mood.lower() in aset else 0.0

        # combine with weights (extend weights to include setting/time/overlay/mood with small multipliers)
        extra_weights = {"setting": 0.07, "time_of_day": 0.06, "overlay": 0.03, "mood": 0.02}
        total_score = (
            weights["tag"] * tag_s +
            weights["motion"] * motion_s +
            weights["duration"] * dur_s +
            weights["caption"] * cap_s +
            weights["resolution"] * res_s +
            extra_weights["setting"] * setting_s +
            extra_weights["time_of_day"] * time_s +
            extra_weights["overlay"] * overlay_s +
            extra_weights["mood"] * mood_s
        )

        # slight hard-bias for drone_shot tags (if shot_type == 'drone_shot' and asset_tags include 'drone'/'aerial')
        if shot_type == "drone_shot":
            if any(t in ("drone", "aerial", "birds-eye", "topdown", "landscape") for t in [x.lower() for x in asset_tags]):
                total_score += 0.05

        # choose video file if needed; for slow motion require higher fps preference
        chosen_file = None
        if asset_type == "video":
            chosen_file = choose_video_file(asset, desired_width, desired_height, prefer_hls, required_min_fps)
            # penalize if slow_motion required but chosen_file lacks fps info or fps too low
            if required_min_fps:
                file_fps = (chosen_file.get("fps") if chosen_file else None) or 0
                if file_fps < required_min_fps:
                    # reduce score (can't provide true slow-motion)
                    total_score *= 0.6

        reason_parts = [
            f"tag:{tag_s:.2f}",
            f"motion:{motion_s:.2f}",
            f"dur:{dur_s:.2f}",
            f"cap:{cap_s:.2f}",
            f"res:{res_s:.2f}",
            f"setting:{setting_s:.2f}",
            f"time:{time_s:.2f}",
            f"overlay:{overlay_s:.2f}"
        ]
        reason = ", ".join(reason_parts)

        results.append({
            "asset": asset,
            "asset_id": asset.get("id") or asset.get("url"),
            "type": asset_type,
            "score": float(total_score),
            "reason": reason,
            "chosen_video_file": chosen_file
        })

    for v in candidate_videos:
        process_asset(v, "video")
    for p in candidate_photos:
        process_asset(p, "image")

    # sort and return
    results.sort(key=lambda x: x["score"], reverse=True)

    if top_k:
        results = results[:top_k]
    return results


def visual_retriever_logic(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The main orchestrator for the VisualRetrieverAgent. It loops through each
    shot, generates queries, fetches assets, and ranks them.
    """
    logger.info("--- Executing: VisualRetrieverAgent ---")
    shot_list: List[ShotEntry] = state["shot_list"]
    spec: CanonicalSpec = state["spec"]
    
    final_shot_assets = {}
    for shot in shot_list:
        # Step 1: Generate Queries for the current shot
        queries = _generate_queries_for_shot(shot, spec)
        if not queries:
            logger.warning(f"No queries generated for shot {shot.shot_id}. Skipping.")
            final_shot_assets[shot.shot_id] = []
            continue

        # Step 2: Execute the searches on Pexels
        all_candidates = _execute_searches(queries, spec)

        # Step 3: Rank the found assets
        ranked_assets = rank_shot_assets(
            shot=shot,
            all_candidates=all_candidates,
            top_k=5,
            reel_mood=spec.mood.value if spec.mood else None
        )

        asset_candidates = []
        for asset in ranked_assets:
            asset_candidate = AssetCandidate(
                id=f"pexels-{asset["asset_id"]}",
                type=asset["type"],
                path=asset["chosen_video_file"]["link"] if asset["type"] == "video" else asset["asset"]["src"]["original"],
                width=asset['asset'].get("width"),
                height=asset['asset'].get("height"),
                duration_sec=asset["asset"].get("duration",None),
                score=asset["score"]
            )
            asset_candidates.append(asset_candidate)

        logger.info(f"--- Top ranked asset for shot {shot.shot_id} (Score: {ranked_assets[0]['score']:.2f}) ---")
        final_shot_assets[shot.shot_id] = asset_candidates


    return {"visual_assets": final_shot_assets}



if __name__ == "__main__":
    ranked_assets = [{'asset': {'id': 8869366, 'width': 4000, 'height': 6000, 'url': 'https://www.pexels.com/photo/close-up-shot-of-a-man-drinking-a-bottle-of-water-8869366/', 'photographer': 'MART  PRODUCTION', 'photographer_url': 'https://www.pexels.com/@mart-production', 'photographer_id': 30039676, 'avg_color': '#85776C', 'src': {'original': 'https://images.pexels.com/photos/8869366/pexels-photo-8869366.jpeg', 'large2x': 'https://images.pexels.com/photos/8869366/pexels-photo-8869366.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', 'large': 'https://images.pexels.com/photos/8869366/pexels-photo-8869366.jpeg?auto=compress&cs=tinysrgb&h=650&w=940', 'medium': 'https://images.pexels.com/photos/8869366/pexels-photo-8869366.jpeg?auto=compress&cs=tinysrgb&h=350', 'small': 'https://images.pexels.com/photos/8869366/pexels-photo-8869366.jpeg?auto=compress&cs=tinysrgb&h=130', 'portrait': 'https://images.pexels.com/photos/8869366/pexels-photo-8869366.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=1200&w=800', 'landscape': 'https://images.pexels.com/photos/8869366/pexels-photo-8869366.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=627&w=1200', 'tiny': 'https://images.pexels.com/photos/8869366/pexels-photo-8869366.jpeg?auto=compress&cs=tinysrgb&dpr=1&fit=crop&h=200&w=280'}, 'liked': False, 'alt': 'Close-up of a man drinking water from a bottle in an outdoor setting, focusing on hydration.'}, 'asset_id': 8869366, 'type': 'image', 'score': 0.5890000000000001, 'reason': 'tag:0.00, motion:1.00, dur:0.90, cap:0.20, res:1.00, setting:0.20, time:0.00, overlay:0.00', 'chosen_video_file': None}, {'asset': {'id': 13700906, 'width': 3803, 'height': 5716, 'url': 'https://www.pexels.com/photo/green-trees-on-mountain-under-white-clouds-13700906/', 'photographer': 'Zetong Li', 'photographer_url': 'https://www.pexels.com/@zetong-li-880728', 'photographer_id': 880728, 'avg_color': '#A1856D', 'src': {'original': 'https://images.pexels.com/photos/13700906/pexels-photo-13700906.jpeg', 'large2x': 'https://images.pexels.com/photos/13700906/pexels-photo-13700906.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', 'large': 'https://images.pexels.com/photos/13700906/pexels-photo-13700906.jpeg?auto=compress&cs=tinysrgb&h=650&w=940', 'medium': 'https://images.pexels.com/photos/13700906/pexels-photo-13700906.jpeg?auto=compress&cs=tinysrgb&h=350', 'small': 'https://images.pexels.com/photos/13700906/pexels-photo-13700906.jpeg?auto=compress&cs=tinysrgb&h=130', 'portrait': 'https://images.pexels.com/photos/13700906/pexels-photo-13700906.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=1200&w=800', 'landscape': 'https://images.pexels.com/photos/13700906/pexels-photo-13700906.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=627&w=1200', 'tiny': 'https://images.pexels.com/photos/13700906/pexels-photo-13700906.jpeg?auto=compress&cs=tinysrgb&dpr=1&fit=crop&h=200&w=280'}, 'liked': False, 'alt': 'Sunrise casts golden rays over rolling fog and forested hills, creating a dramatic landscape.'}, 'asset_id': 13700906, 'type': 'image', 'score': 0.5830000000000001, 'reason': 'tag:0.00, motion:1.00, dur:0.90, cap:0.00, res:1.00, setting:0.40, time:0.33, overlay:0.00', 'chosen_video_file': None}, {'asset': {'id': 3124674, 'width': 2160, 'height': 2700, 'url': 'https://www.pexels.com/photo/woman-drinking-water-from-glass-bottle-3124674/', 'photographer': 'Arnie Watkins', 'photographer_url': 'https://www.pexels.com/@arnie-watkins-1337313', 'photographer_id': 1337313, 'avg_color': '#A1A5BF', 'src': {'original': 'https://images.pexels.com/photos/3124674/pexels-photo-3124674.jpeg', 'large2x': 'https://images.pexels.com/photos/3124674/pexels-photo-3124674.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', 'large': 'https://images.pexels.com/photos/3124674/pexels-photo-3124674.jpeg?auto=compress&cs=tinysrgb&h=650&w=940', 'medium': 'https://images.pexels.com/photos/3124674/pexels-photo-3124674.jpeg?auto=compress&cs=tinysrgb&h=350', 'small': 'https://images.pexels.com/photos/3124674/pexels-photo-3124674.jpeg?auto=compress&cs=tinysrgb&h=130', 'portrait': 'https://images.pexels.com/photos/3124674/pexels-photo-3124674.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=1200&w=800', 'landscape': 'https://images.pexels.com/photos/3124674/pexels-photo-3124674.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=627&w=1200', 'tiny': 'https://images.pexels.com/photos/3124674/pexels-photo-3124674.jpeg?auto=compress&cs=tinysrgb&dpr=1&fit=crop&h=200&w=280'}, 'liked': False, 'alt': 'A woman drinking water from a glass bottle against a clear blue sky, emphasizing hydration.'}, 'asset_id': 3124674, 'type': 'image', 'score': 0.5824863606183667, 'reason': 'tag:0.00, motion:1.00, dur:0.90, cap:0.20, res:0.97, setting:0.20, time:0.00, overlay:0.00', 'chosen_video_file': None}, {'asset': {'id': 7034779, 'width': 4000, 'height': 6000, 'url': 'https://www.pexels.com/photo/a-woman-drinking-coffee-while-reading-a-book-7034779/', 'photographer': 'George Milton', 'photographer_url': 'https://www.pexels.com/@george-milton', 'photographer_id': 19162611, 'avg_color': '#A7A3A0', 'src': {'original': 'https://images.pexels.com/photos/7034779/pexels-photo-7034779.jpeg', 'large2x': 'https://images.pexels.com/photos/7034779/pexels-photo-7034779.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', 'large': 'https://images.pexels.com/photos/7034779/pexels-photo-7034779.jpeg?auto=compress&cs=tinysrgb&h=650&w=940', 'medium': 'https://images.pexels.com/photos/7034779/pexels-photo-7034779.jpeg?auto=compress&cs=tinysrgb&h=350', 'small': 'https://images.pexels.com/photos/7034779/pexels-photo-7034779.jpeg?auto=compress&cs=tinysrgb&h=130', 'portrait': 'https://images.pexels.com/photos/7034779/pexels-photo-7034779.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=1200&w=800', 'landscape': 'https://images.pexels.com/photos/7034779/pexels-photo-7034779.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=627&w=1200', 'tiny': 'https://images.pexels.com/photos/7034779/pexels-photo-7034779.jpeg?auto=compress&cs=tinysrgb&dpr=1&fit=crop&h=200&w=280'}, 'liked': False, 'alt': 'Casual woman enjoying a peaceful moment with coffee and a book in a modern kitchen setting.'}, 'asset_id': 7034779, 'type': 'image', 'score': 0.5770000000000001, 'reason': 'tag:0.00, motion:1.00, dur:0.90, cap:0.00, res:1.00, setting:0.60, time:0.00, overlay:0.00', 'chosen_video_file': None}, {'asset': {'id': 8018978, 'width': 3889, 'height': 5826, 'url': 'https://www.pexels.com/photo/back-view-of-a-person-working-out-near-the-sea-8018978/', 'photographer': 'ROMAN ODINTSOV', 'photographer_url': 'https://www.pexels.com/@roman-odintsov', 'photographer_id': 2678846, 'avg_color': '#938B80', 'src': {'original': 'https://images.pexels.com/photos/8018978/pexels-photo-8018978.jpeg', 'large2x': 'https://images.pexels.com/photos/8018978/pexels-photo-8018978.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', 'large': 'https://images.pexels.com/photos/8018978/pexels-photo-8018978.jpeg?auto=compress&cs=tinysrgb&h=650&w=940', 'medium': 'https://images.pexels.com/photos/8018978/pexels-photo-8018978.jpeg?auto=compress&cs=tinysrgb&h=350', 'small': 'https://images.pexels.com/photos/8018978/pexels-photo-8018978.jpeg?auto=compress&cs=tinysrgb&h=130', 'portrait': 'https://images.pexels.com/photos/8018978/pexels-photo-8018978.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=1200&w=800', 'landscape': 'https://images.pexels.com/photos/8018978/pexels-photo-8018978.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=627&w=1200', 'tiny': 'https://images.pexels.com/photos/8018978/pexels-photo-8018978.jpeg?auto=compress&cs=tinysrgb&dpr=1&fit=crop&h=200&w=280'}, 'liked': False, 'alt': 'Woman practicing yoga on a mat by the ocean at sunrise.'}, 'asset_id': 8018978, 'type': 'image', 'score': 0.5690000000000001, 'reason': 'tag:0.00, motion:1.00, dur:0.90, cap:0.00, res:1.00, setting:0.20, time:0.33, overlay:0.00', 'chosen_video_file': None}]
    from ...test.state_loader import load_app_state, save_app_state
    state = load_app_state()
    state.update(visual_retriever_logic(state))
    save_app_state(state)
    