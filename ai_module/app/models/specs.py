# models/specs.py

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

OverlayElementType = Literal["title_card", "stickers", "hashtags"]

class FieldWithExplicit(BaseModel):
    """Wraps any blueprint field with a boolean to indicate if it came from user input."""
    value: Optional[object] = Field(None, description="The value of the parameter, either supplied by the user or inferred by an agent.")
    explicit: bool = Field(
        default=False, 
        description="A boolean flag set to 'true' if the 'value' was explicitly provided by the user in the initial prompt. This prevents creative agents from overriding it."
    )

class CanonicalSpec(BaseModel):
    """Single source of truth for the job; used by all subsequent agents to guide generation."""
    title: Optional[FieldWithExplicit] = Field(None, description="The title of the reel, which can be used for title cards or social media captions.")
    duration_sec: Optional[FieldWithExplicit] = Field(default=FieldWithExplicit(value=30, explicit=False), description="The target duration of the final video in seconds.")
    aspect_ratio: Optional[FieldWithExplicit] = Field(default=FieldWithExplicit(value="9:16", explicit=False), description="The aspect ratio of the video.")
    mood: Optional[FieldWithExplicit] = Field(None, description="The desired overall mood (as a descriptive string).")
    pacing: Optional[FieldWithExplicit] = Field(None, description="The desired pacing of the edits (as a descriptive string).")
    voice_style: Optional[FieldWithExplicit] = Field(None, description="The style for the text-to-speech voice.")
    music_style: Optional[FieldWithExplicit] = Field(None, description="The style of background music.")
    overlay_elements: Optional[FieldWithExplicit] = Field(default=FieldWithExplicit(value=[], explicit=False), description="A list of overlay elements to include.")
    denies: Optional[FieldWithExplicit] = Field(None, description="A list of elements or topics to explicitly avoid.")
    extra_instructions: Optional[FieldWithExplicit] = Field(None, description="Any other creative instructions from the user.")

class ScriptLine(BaseModel):
    line: str = Field(..., description="The voice-over text for this segment of the script.")
    caption: str = Field(..., description="A short, punchy on-screen caption that corresponds to the voice-over line (max 5-6 words).")
    estimated_sec: float = Field(..., description="The estimated speaking duration of the voice-over line in seconds.")

class OverlayRequest(BaseModel):
    """Defines a request for a specific overlay element within a shot."""
    type: Literal["titles", "text overlays", "stickers", "hashtags"] = Field(..., description="The category of the overlay, e.g., 'sticker' or 'title'.")
    search_tags: List[str] = Field(..., description="Descriptive tags for the overlay content. For a sticker, this would be search terms like ['thumbs up', 'success']. For a title, it could be the text itself.")
    position: Optional[str] = Field(None, description="A hint for where to place the overlay, e.g., 'top_right'.")
    size: Optional[Literal["small", "medium", "large"]] = Field(None, description="A hint for the size of the overlay.")

class ShotEntry(BaseModel):
    """Atomic scene / edit unit used to build the video timeline."""
    shot_id: str = Field(..., description="A unique identifier for this shot (e.g., 's1', 's2').")
    start_sec: float = Field(..., description="The absolute start time of the shot in seconds from the beginning of the video.")
    end_sec: float = Field(..., description="The absolute end time of the shot in seconds.")
    desired_asset_tags: Optional[List[str]] = Field(None, description="A list of descriptive tags for the visual content needed for this shot (e.g., ['sunset', 'beach', 'close-up']).")
    desired_motion: Optional[Literal["static", "pan", "zoom", "slow_motion", "drone_shot"]] = Field(None, description="The desired camera motion for the asset in this shot.")
    caption: Optional[str] = Field(None, description="The on-screen text caption for this specific shot.")
    caption_position: Optional[Literal["top", "bottom", "center"]] = Field(None, description="The desired position of the caption on the screen.")
    shot_type: Optional[Literal["close_up", "medium_shot", "wide_shot", "drone_shot", "pov"]] = Field(None, description="The type of camera framing for the shot.")
    setting_description: Optional[str] = Field(None, description="A brief description of the shot's location or setting (e.g., 'a busy city street at night').")
    time_of_day: Optional[Literal["day", "night", "golden_hour", "blue_hour", "dawn", "dusk"]] = Field(None, description="The specific time of day to enhance the mood and lighting.")
    desired_overlays: Optional[List[OverlayRequest]] = Field(None, description="A list of specific overlay elements (like stickers or titles) to be added to this shot.")

class AssetCandidate(BaseModel):
    """Represents a potential asset (video, image, etc.) for a shot."""
    id: str
    type: Literal["video", "image", "audio", "sticker"]
    path: Optional[str]
    height: Optional[int]=None
    width: Optional[int]=None
    duration_sec: Optional[float]=None
    score: Optional[float] = None

class QuestionAnswer(BaseModel):
    """The output model for the question and answer step."""
    question: str = Field(..., description="A question related to the main topic.")
    answer: str = Field(..., description="A factual answer to the question.")

class FactualResearch(BaseModel):
    """The final, consolidated research output of the agent."""
    summary: str = Field(..., description="A concise, synthesized summary of the key facts found during research.")
    qa_pairs: List[QuestionAnswer] = Field(..., description="A list of question-and-answer pairs based on the research.")
    
