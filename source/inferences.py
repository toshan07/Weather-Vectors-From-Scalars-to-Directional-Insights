import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image, ImageSequence
import io
import base64
import numpy as np
import re

#---------------------Configuring API key---------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

#--------------------- Prompts---------------------------------------------------------------
prompt_pollution="""
    You are analyzing a GIF that shows monthly pollution (PM2.5) gradient maps for the year over Northern India (Jammu & Kashmir, Himachal Pradesh, Punjab, Haryana, Uttarakhand, and nearby regions).
    Each frame represents a month, with colors showing pollution intensity (green = low, red = high) and arrows showing movement/dispersion.

    Please provide 2–4 concise, high-level insights that cover:
    1) Seasonal pollution patterns (winter, summer, monsoon, post-monsoon).
    2) Regional differences (north vs. south, plains vs. hills).
    3) Changes in pollution intensity and movement trends.

    Guidelines:
    - Write like dynamic field notes (not too polished, not GPT-fluffy).

    - Keep it short and observational.
    - Use simple language, as if explaining to a student
    - Avoid over-interpretation; just describe what you see.
    - make it concise 4-6 bullet points.
    - just return the bullet points no other text.
    - Each bullet should be small and not too long.

    Example style: (dont use * use - for bullets )
    -   Winter/Post-Monsoon Pollution Peak: Late autumn and winter (October-February) consistently show the highest pollution, with deep red colors concentrating in the southern plains.
    -   Monsoon Relief: Pollution dramatically decreases across the entire region during the monsoon months (July-August), shifting to predominantly green and light yellow.
    -   Plains vs. Hills Gradient: The southern plains consistently experience higher pollution levels compared to the northern hilly regions, with this difference being most pronounced during high-pollution seasons.
    -   Northward Dispersion: A dominant trend shows pollution moving northward or northwestward from the southern plains towards the Himalayan foothills, especially during the high-pollution periods.
              
    Keep each point easy for a professor to understand. Avoid technical jargon.
"""

prompt_wind="""
    You are an atmospheric scientist. 
    I will provide you with a GIF showing daily wind vectors over a region for a month in some year. 
    Your job is to watch the animation and summarize the wind behavior in a few bullet points.

    - Write like dynamic field notes (not too polished, not GPT-fluffy).
    - Focus on patterns: which regions show stronger winds, calmer areas, directional shifts, and any anomalies.
    - Mention approximate speeds when noticeable (e.g., 2–3 m/s normally, 4.5+ m/s bursts).
    - Keep it short and observational.
    - Use simple language, as if explaining to a student
    - Avoid over-interpretation; just describe what you see.
    - make it concise 4 bullet points.
    - just return the bullet points no other text.
    - dont use too techinal stuff and dont mention cyclone or any other weather phenomenom
    - Each bullet should be small and not too long.

    Example style:
    - Winds generally strengthen in the north-eastern region around mid-July (yellow/green patches show higher speeds).
    - Early July has calmer central zones (blue arrows), with stronger winds only near the eastern edge.
    - Wind speeds are mostly 2–3 m/s, but short bursts exceed 4.5 m/s along the eastern border.

    Now, analyze the attached GIF and give me 8–12 similar points.
"""
#---------------------Extract frames--------------------------------------------------------------------------------
def extract_key_frames(gif_path, num_frames=2):
    frames = []
    with Image.open(gif_path) as im:
        all_frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(im)]

    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    for idx in indices:
        buffer = io.BytesIO()
        all_frames[idx].save(buffer, format="PNG")
        frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return frames


def analyze_gif(gif_path,parameter):
    key_frames = extract_key_frames(gif_path, num_frames=30) 

    if parameter=="wind":
        prompt = (prompt_wind)
    else:
        prompt = (prompt_pollution)

    model = genai.GenerativeModel("gemini-2.5-flash")

    contents = [
        {
            "role": "user",
            "parts": [{"text": prompt}]
        }
    ]

    for frame_b64 in key_frames:
        contents[0]["parts"].append({
            "inline_data": {"mime_type": "image/png", "data": frame_b64}
        })
    response = model.generate_content(contents)
    return response.text

def parse_bullets(s):
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    items = []
    for ln in lines:
        clean = re.sub(r'^[\-\u2022\*\u2023\•\s]+', '', ln)
        items.append(clean)
    return items

def generate_infernces_wind(gif_path):
    result=analyze_gif(gif_path,"wind")
    inf = parse_bullets(result)
    return inf

def generate_infernces_pollution(gif_path):
    result=analyze_gif(gif_path,"pollution")
    inf = parse_bullets(result)
    return inf