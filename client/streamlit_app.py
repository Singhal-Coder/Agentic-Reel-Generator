import requests
import streamlit as st

API_URL = "https://agentic-reel-generator.onrender.com/api/generate"

st.set_page_config(page_title="Agentic Reel Generator", page_icon="🎬", layout="centered")

st.title("🎬 Agentic Reel Generator")
st.write("Enter a prompt and choose options to generate a short reel.")

with st.form("reel_form"):
    prompt = st.text_area("Prompt", placeholder="e.g., Strength of Lord Hanuman", height=140)

    col1, col2 = st.columns(2)
    with col1:
        vduration_sec = st.number_input("Duration (seconds)", min_value=5, max_value=60, value=30, step=1)
        aspect_ratio = st.selectbox("Aspect ratio", options=["9:16", "1:1", "4:5"], index=0)
    with col2:
        voice_style = st.selectbox(
            "Voice style",
            options=["none", "energetic_male", "calm_male", "energetic_female", "calm_female"],
            index=0,
        )
        music_style = st.selectbox(
            "Music style",
            options=["upbeat_electronic", "cinematic_orchestral", "lofi_hiphop", "acoustic_folk", "ambient"],
            index=0,
        )

    submitted = st.form_submit_button("Generate Reel")

if submitted:
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Generating video. This may take a few minutes..."):
            payload = {
                "prompt": prompt,
                "params": {
                    "vduration_sec": vduration_sec,
                    "aspect_ratio": aspect_ratio,
                    "voice_style": voice_style,
                    "music_style": music_style,
                },
            }
            try:
                resp = requests.post(API_URL, json=payload, timeout=None)
                resp.raise_for_status()
                video_bytes = resp.content
                if not video_bytes:
                    st.error("No video bytes returned from API.")
                else:
                    st.success("Video generated!")
                    st.video(video_bytes, format='video/mp4')
                    st.download_button(
                        label="Download video",
                        data=video_bytes,
                        file_name="generated_reel.mp4",
                        mime="video/mp4",
                    )
            except requests.RequestException as e:
                st.error(f"API error: {e}")


