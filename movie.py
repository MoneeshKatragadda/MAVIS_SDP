import json
import os
import logging
import textwrap
from moviepy.editor import ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, ColorClip, CompositeVideoClip
from moviepy.audio.fx.all import audio_loop
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [MOVIE] %(message)s")
logger = logging.getLogger("MAVIS_MOVIE")

# --- PIL font (no ImageMagick required) ---
_FONT_PATH_BOLD   = "C:/Windows/Fonts/arialbd.ttf"
_FONT_PATH_NORMAL = "C:/Windows/Fonts/arial.ttf"

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

_FONT_SUBTITLE = _load_font(_FONT_PATH_BOLD,   14)
_FONT_SPEAKER  = _load_font(_FONT_PATH_NORMAL, 16)

def _draw_subtitles(frame, text, wrap_width=55):
    """Burn wrapped subtitle text onto the bottom of a numpy RGB frame."""
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    lines = textwrap.wrap(text, width=wrap_width)
    line_h = 34
    total_h = len(lines) * line_h + 16
    y_start = h - total_h - 20

    # Semi-transparent black background bar
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([0, y_start - 8, w, h - 10], fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=_FONT_SUBTITLE)
        tw = bbox[2] - bbox[0]
        x  = (w - tw) // 2
        y  = y_start + i * line_h
        # shadow
        draw.text((x + 2, y + 2), line, font=_FONT_SUBTITLE, fill=(0, 0, 0, 255))
        draw.text((x, y),         line, font=_FONT_SUBTITLE, fill=(255, 255, 255, 255))

    return np.array(img)

def _draw_speaker_label(frame, speaker):
    """Burn a small speaker badge in the top-left corner."""
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    label = f"  {speaker}  "
    bbox  = draw.textbbox((0, 0), label, font=_FONT_SPEAKER)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    x0, y0 = 18, 18
    x1, y1 = x0 + tw + pad * 2, y0 + th + pad * 2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=(255, 255, 255, 220))
    draw.text((x0 + pad, y0 + pad), label, font=_FONT_SPEAKER, fill=(20, 20, 20))
    return np.array(img)

EVENTS_FILE = "output/events.json"
OUTPUT_VIDEO = "output/movie.mp4"
IMG_DIR = "output/images"
AUDIO_DIR = "output/audio"
BGM_DIR = os.path.join(AUDIO_DIR, "bgm")
SFX_DIR = os.path.join(AUDIO_DIR, "sfx")

def generate_movie(events_path=EVENTS_FILE, output_file=OUTPUT_VIDEO):
    if not os.path.exists(events_path):
        logger.error(f"Events file not found at {events_path}")
        return

    with open(events_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timeline = data.get("timeline", [])
    clips = []

    logger.info("--- ASSEMBLING MOVIE ---")

    for scene in timeline:
        scene_id = scene.get("id")
        logger.info(f"Processing Scene: {scene_id}")

        for beat in scene.get("beats", []):
            beat_id = beat.get("sub_scene_id")
            text = beat.get("text")
            duration = beat.get("duration", 2.0)

            # 1. VISUAL
            img_path = os.path.join(IMG_DIR, f"{beat_id}.png")
            if os.path.exists(img_path):
                clip = ImageClip(img_path).set_duration(duration)
            else:
                logger.warning(f"  Missing Image: {beat_id}. Using Black Placeholder.")
                clip = ColorClip(size=(1024, 1024), color=(0, 0, 0), duration=duration)

            # 2. PRIMARY AUDIO (Dialogue / Narration)
            audio_path = os.path.join(AUDIO_DIR, f"{beat_id}.wav")
            primary_audio = None
            if os.path.exists(audio_path):
                primary_audio = AudioFileClip(audio_path)
                # Extend visual to match audio if audio is longer than estimated duration
                if primary_audio.duration > duration:
                    clip = clip.set_duration(primary_audio.duration)
            else:
                logger.warning(f"  Missing Audio: {beat_id}")

            # 3. COMPOSITE AUDIO (Primary + BGM + SFX)
            audio_layers = []
            if primary_audio:
                audio_layers.append(primary_audio)

            # --- BGM ---
            prod = beat.get("production", {})
            bgm_info = prod.get("bgm", {})
            style = bgm_info.get("style", "None")
            vol = bgm_info.get("volume", 0.1) * 0.8  # Defaulting background to 8% to prevent dialogue drowning

            if style and style.lower() not in ["none", "silence"]:
                key = style.replace(" ", "_").replace("/", "-").lower()
                bgm_path = os.path.join(BGM_DIR, f"{key}.wav")
                if os.path.exists(bgm_path):
                    bgm_clip = AudioFileClip(bgm_path)
                    if bgm_clip.duration < clip.duration:
                        bgm_clip = audio_loop(bgm_clip, duration=clip.duration)
                    else:
                        bgm_clip = bgm_clip.subclip(0, clip.duration)
                    bgm_clip = bgm_clip.volumex(vol)
                    audio_layers.append(bgm_clip)

            # --- SFX ---
            sfx_list = prod.get("sfx", [])
            for s in sfx_list:
                name = s.get("name", "None")
                if name and name.lower() != "none":
                    key = name.replace(" ", "_").replace("/", "-").lower()
                    sfx_path = os.path.join(SFX_DIR, f"{key}.wav")
                    if os.path.exists(sfx_path):
                        sfx_clip = AudioFileClip(sfx_path).volumex(0.2)
                        # Relative start offset (0.0–1.0) → absolute seconds
                        start_offset = 0.0
                        timing = s.get("timing", {})
                        if "start" in timing:
                            start_offset = timing["start"] * clip.duration
                        sfx_clip = sfx_clip.set_start(start_offset)
                        audio_layers.append(sfx_clip)

            # Combine audio layers
            if audio_layers:
                final_audio = CompositeAudioClip(audio_layers)
                final_audio = final_audio.set_duration(clip.duration)
                clip = clip.set_audio(final_audio)

            # 4. SUBTITLES
            if text:
                try:
                    clip = clip.fl_image(lambda f, t=text: _draw_subtitles(f, t))
                except Exception as e:
                    logger.warning(f"  Subtitle Error {beat_id}: {e}")

            # 5. SPEAKER INDICATOR
            speaker = beat.get("speaker", "Narrator")
            if not speaker or speaker == "Unknown":
                speaker = "Narrator"

            try:
                clip = clip.fl_image(lambda f, s=speaker: _draw_speaker_label(f, s))
            except Exception as e:
                logger.warning(f"  Speaker Indicator Error {beat_id}: {e}")

            clips.append(clip)

    if not clips:
        logger.error("No clips generated!")
        return

    logger.info(f"Concatenating {len(clips)} beats...")
    final_video = concatenate_videoclips(clips, method="compose")

    logger.info(f"Writing video to {output_file}...")
    final_video.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac")
    logger.info("Movie Generation Complete.")


if __name__ == "__main__":
    import sys
    events = sys.argv[1] if len(sys.argv) > 1 else EVENTS_FILE
    generate_movie(events)
