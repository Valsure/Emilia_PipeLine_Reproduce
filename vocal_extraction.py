from spleeter.separator import Separator
import subprocess
import ffmpeg
import numpy as np
import librosa
import io
from scipy.io.wavfile import write
from demo import standardization

def extract_audio_from_mp4(mp4_file):
    # 从 mp4 文件中提取 wav 音频数据
    process = (
        ffmpeg.input(mp4_file)
        .output('pipe:', format='wav')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    audio_data, _ = process.communicate()

    # 将字节流转换为内存文件
    audio_stream = io.BytesIO(audio_data)

    # 使用 librosa 将音频数据转换为 NumPy 数组
    y, sr = librosa.load(audio_stream, sr=None)  # sr=None 保持原始采样率
    return y, sr


def to_stereo_if_needed(waveform):
    # 如果是单声道，转换为立体声
    if len(waveform.shape) == 1:
        waveform = np.stack([waveform, waveform], axis=1)
    return waveform


def save_vocals_to_wav(vocals, sample_rate, output_path):
    # 使用 scipy.io.wavfile 将 numpy array 保存为 wav 文件
    write(output_path, sample_rate, vocals)


def main(mp4_file, output_wav):
    # 提取音频数据并转换为 NumPy 数组
    waveform, sample_rate = extract_audio_from_mp4(mp4_file)

    # 将音频转换为立体声格式
    waveform = to_stereo_if_needed(waveform)

    # 初始化 Spleeter Separator
    separator = Separator('spleeter:2stems')

    # 将音频波形传递给 separate 函数
    result = separator.separate(waveform)

    # 提取 vocals 并保存为 WAV 文件
    vocals = result['vocals']
    save_vocals_to_wav(vocals, sample_rate, output_wav)

    print("Vocal extraction and save complete.")
    return result


# 输入视频文件路径和输出 WAV 文件路径
mp4_file = 'audio1.wav'
output_wav = 'vocals_output1.wav'
vocals = main(mp4_file, output_wav)
