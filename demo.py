import os
from pprint import pprint
from pydub import AudioSegment
import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from spleeter.separator import Separator
import ffmpeg
import numpy as np
import librosa
import io
import torch
import silero_vad
def detect_gpu():
    """Detect if GPU is available and print related information."""

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        print("ENV: CUDA_VISIBLE_DEVICES not set, use default setting")
    else:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"ENV: CUDA_VISIBLE_DEVICES = {gpu_id}")

    if not torch.cuda.is_available():
        print("Torch CUDA: No GPU detected. torch.cuda.is_available() = False.")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"Torch CUDA: Detected {num_gpus} GPUs.")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f" * GPU {i}: {gpu_name}")

    print("Torch: CUDNN version = " + str(torch.backends.cudnn.version()))
    if not torch.backends.cudnn.is_available():
        print("Torch: CUDNN is not available.")
        return False
    print("Torch: CUDNN is available.")

    return True


def standardization(audio):

    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    print("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    audio = audio.set_frame_rate(24000)
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)

    print("Audio file converted to WAV format")

    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    print(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize

    print(f"waveform shape: {waveform.shape}")
    print("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": 24000,
    }


def vocal_extraction(audio):
    """
    Spleeter接受的音频输入为双声道，而前面的standardization返回的是单声道，
    因此在此函数中先转成双声道，完成人声分离后再转回单声道
    """
    if isinstance(audio, dict) and "waveform" in audio:
        waveform = audio["waveform"]
        # 将单声道 waveform 转换为双声道
        stereo_waveform = np.stack([waveform, waveform], axis=1)

        separator = Separator('spleeter:2stems')
        try:
            result = separator.separate(stereo_waveform)
            vocals = result['vocals']
            print(f"人声分离成功：{audio['name']}")

            if vocals.shape[1] == 2:
                # 对两个声道求平均，得到单声道
                mono_vocals = np.mean(vocals, axis=1)
                return mono_vocals
            else:
                return vocals
        except Exception as e:
            print(e, f"人声分离失败：{audio['name']}")
            return None
    else:
        raise ValueError("Invalid audio format for vocal extraction")

def speaker_diarization(audio):
    if audio is None:
        print("音频为空，无法进行分段")
        return None

    waveform = torch.tensor(audio).unsqueeze(0).to(device)

    try:
        segments = diarization_pipeline(
            {
                "waveform": waveform,
                "sample_rate": 24000,
            }
        )

        # 将分段结果转换为 DataFrame
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

        print(f"说话人分段完成: {diarize_df}")
        return diarize_df
    except Exception as e:
        print(f"说话人分段失败: {e}")
        return None


def cut_by_speaker_label(vad_list):

    MERGE_GAP = 2  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            print("segment长度大于 30s，裁切为 30s 以内")
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker!= vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    print(f"合并： {len(vad_list) - len(updated_list)} segments")

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    print(f"移除： {len(updated_list) - len(filter_list)} segments by length")
    filter_list_df = pd.DataFrame(filter_list)
    return filter_list_df

def main(audio_file):
    audio = standardization(audio_file)
    vocal = vocal_extraction(audio)  # 提取人声部分
    if vocal is not None:
        speakerdia = speaker_diarization(vocal)  # 进行说话人分段
        if speakerdia is not None:
            vad_list = vad.vad(speakerdia, vocal)  # 音频活动（说话人）检测
            pprint( vad_list)
            segment_list = cut_by_speaker_label(vad_list)
            pprint(segment_list)
        else:
            print("说话人分段失败")
    else:
        print("人声分离失败")


if __name__ == "__main__":
    if detect_gpu():
        print("使用GPU")
        device_name = "cuda"
        device = torch.device(device_name)
    else:
        print("使用CPU")
        device_name = "cpu"
        device = torch.device(device_name)

    diarization_pipeline = Pipeline.from_pretrained(
      "pyannote/speaker-diarization-3.1",
      use_auth_token="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")

    diarization_pipeline.to(device)
    vad = silero_vad.SileroVAD(device=device)
    main("audio1.wav")