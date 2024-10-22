from pydub import AudioSegment

audio = AudioSegment.from_file("audio_samples/audio1.mp3")

audio = audio.set_frame_rate(16000)

audio.export("resampled_audio.wav", format="wav")
