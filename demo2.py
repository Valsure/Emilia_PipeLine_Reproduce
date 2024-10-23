import pandas as pd
import torch
import silero_vad
from scipy.io import wavfile
import numpy as np
from pyannote.audio import Pipeline

# 加载 Silero VAD 模型
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
get_speech_ts = utils[0]  # 获取 VAD 中的 get_speech_timestamps 函数

# 加载 Pyannote 的 Speaker Diarization 模型
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
)

# 加载音频文件
sample_rate, wav_data = wavfile.read("audio_samples/audio1.wav")

# 如果音频是立体声（2通道），将其转换为单声道
if len(wav_data.shape) > 1:
    wav_data = np.mean(wav_data, axis=1).astype(np.int16)  # 平均两个声道，转换为单声道

# 先进行 Speaker Diarization（说话人分离）
diarization = pipeline("audio1.wav")

# 初始化用于保存分离结果和 VAD 结果的数据
data = []

# 遍历 diarization 的输出，逐个处理每个说话人的段落
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start, end = int(turn.start * sample_rate), int(turn.end * sample_rate)
    segment_wav = wav_data[start:end]  # 获取说话人段落的音频数据

    # 对这个说话人段落使用 Silero VAD 进行语音活动检测
    speech_timestamps = get_speech_ts(segment_wav, vad_model, sampling_rate=sample_rate)

    # 如果这个段落中有语音活动，保存结果
    if speech_timestamps:
        for ts in speech_timestamps:
            segment = f"{turn.start + ts['start'] / sample_rate:.1f}s - {turn.start + ts['end'] / sample_rate:.1f}s"
            data.append([segment, "speech", speaker])

# 将数据转换为 DataFrame
df = pd.DataFrame(data, columns=["segment", "label", "speaker"])

# 保存为 CSV 文件
# df.to_csv("diarization_vad_output.csv", index=False)
