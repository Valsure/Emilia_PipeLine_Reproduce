from pydub import AudioSegment

# Load your audio file
audio = AudioSegment.from_file("audio1.mp3")

# Resample to a constant sample rate (e.g., 16kHz)
audio = audio.set_frame_rate(16000)

# Export the resampled audio as .wav (which works better for many pipelines)
audio.export("resampled_audio.wav", format="wav")
