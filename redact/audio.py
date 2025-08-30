from pydub import AudioSegment
from pydub.generators import Sine

def apply_audio_actions(audio_path: str, actions: list, out_path: str):
    """
    actions: [{start_ms,end_ms, action: 'bleep'|'mute'}]
    """
    au = AudioSegment.from_file(audio_path)
    for a in actions:
        seg = au[a["start_ms"]:a["end_ms"]]
        if a["action"] == "bleep":
            tone = Sine(1000).to_audio_segment(duration=len(seg)).apply_gain(+6)
            au = au[:a["start_ms"]] + tone + au[a["end_ms"]:]
        elif a["action"] == "mute":
            silence = AudioSegment.silent(duration=len(seg))
            au = au[:a["start_ms"]] + silence + au[a["end_ms"]:]
    au.export(out_path, format="wav")
