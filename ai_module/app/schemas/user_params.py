from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

VoiceStyleType = Literal["energetic_male", "calm_male", "energetic_female", "calm_female", "none"]
MusicStyleType = Literal["upbeat_electronic", "cinematic_orchestral", "lofi_hiphop", "acoustic_folk", "ambient"]
AspectRatioType = Literal["9:16", "1:1", "4:5"]

class UserParams(BaseModel):
    vduration_sec: Optional[int] = Field(None, description="The duration of the video in seconds. Maximum duration is 60 seconds and minimum duration is 5 seconds.")
    aspect_ratio: AspectRatioType = Field(None, description="The aspect ratio.")
    voice_style: Optional[VoiceStyleType] = Field(None, description="The desired voice style.")
    music_style: Optional[MusicStyleType] = Field(None, description="The desired music style.")

    @field_validator('vduration_sec', mode='after')
    @classmethod
    def cap_duration(cls, v):
        if v is None:
            return v
        return max(min(v, 60), 5)