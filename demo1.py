import pandas as pd
import torch
import silero_vad
from scipy.io import wavfile
import numpy as np
from pyannote.audio import Pipeline

# 加载 Silero VAD 模型
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)

# utils 是一个元组，按索引获取需要的函数
get_speech_ts = utils[0]  # get_speech_timestamps 是元组的第一个元素

# 加载音频文件
sample_rate, wav_data = wavfile.read("audio1.wav")

# 如果音频是立体声（2通道），将其转换为单声道
if len(wav_data.shape) > 1:
    wav_data = np.mean(wav_data, axis=1).astype(np.int16)  # 平均两个声道，转换为单声道

# 使用 Silero VAD 进行语音活动检测
speech_timestamps = get_speech_ts(wav_data, vad_model, sampling_rate=sample_rate)

# 打印语音活动段落
for ts in speech_timestamps:
    print(f"Speech segment: {ts['start']} - {ts['end']}")

# 加载 Pyannote 的说话人分离模型
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
)

data = []

# 针对每个 VAD 结果（语音活动片段）运行说话人分离
for ts in speech_timestamps:
    start_time = ts['start'] / sample_rate  # 转换为秒
    end_time = ts['end'] / sample_rate      # 转换为秒
    segment_wav = wav_data[ts['start']:ts['end']]  # 提取音频片段

    # 转换为 torch Tensor，并调整形状为 (1, time)
    segment_wav_tensor = torch.from_numpy(segment_wav).float().unsqueeze(0)  # 单声道，形状为 (1, time)

    # 对该片段运行说话人分离模型
    diarization = pipeline({"waveform": segment_wav_tensor, "sample_rate": sample_rate})

    # 遍历 diarization 的输出并存储到列表中
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = f"{start_time + turn.start:.1f}s - {start_time + turn.end:.1f}s"
        data.append([segment, "speech", speaker])

# 将数据转换为 DataFrame
df = pd.DataFrame(data, columns=["segment", "label", "speaker"])

# 保存为 CSV 文件
df.to_csv("diarization_output.csv", index=False)
