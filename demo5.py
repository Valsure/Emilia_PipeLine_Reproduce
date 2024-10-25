import json
import os
import time

from pydub import AudioSegment
import pandas as pd
from pyannote.audio import Pipeline
from spleeter.separator import Separator
import numpy as np
import librosa
import torch
from tqdm import tqdm
import silero_vad, whisper_asr, dnsmos
from tool import detect_gpu, calculate_audio_stats, export_to_mp3
from scipy.io.wavfile import write
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

    # #将音频数据保存为wav文件
    # output_file = f"outputs_in_progress/step1_standardized.wav"
    # write(output_file, 24000, waveform)
    # print("step1: 标准化的音频已保存为wav文件")

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
            vocal = {}
            vocal["waveform"] = vocals
            vocal["sample_rate"] = audio["sample_rate"]
            vocal["name"] = audio["name"]
            if vocals.shape[1] == 2:
                # 对两个声道求平均，得到单声道
                mono_vocals = np.mean(vocal["waveform"], axis=1)
                vocal["waveform"] = mono_vocals

            #将提取的人声保存为wav文件
            # output_file = f"outputs_in_progress/step2_vocal.wav"
            # write(output_file, 24000, vocal["waveform"])
            # print("step2: 人声已保存为wav文件")

            return vocal
        except Exception as e:
            print(e, f"人声分离失败：{audio['name']}")
            return None
    else:
        raise ValueError("Invalid audio format for vocal extraction")

def speaker_diarization(audio):
    if audio is None:
        print("音频为空，无法进行分段")
        return None
    waveform = audio["waveform"]
    waveform = torch.tensor(waveform).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    try:
        segments = diarization_pipeline(
            {
                "waveform": waveform,
                "sample_rate": audio["sample_rate"],
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

        # #将分段结果保存为excel文件
        # diarize_df.to_excel(f"outputs_in_progress/step3_diarization.xlsx", index=False)
        # print("step3: 分段结果保存为excel文件")
        return diarize_df
    except Exception as e:
        print(f"说话人分段失败: {e}")
        return None


def cut_by_speaker_label(vad_list):

    MERGE_GAP = 2  # 如果两段segment的间隔低于这个值，则合并
    MIN_SEGMENT_LENGTH = 3  # 最短的segment时长
    MAX_SEGMENT_LENGTH = 30  # 最长的segment时长

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
            updated_list[-1]["end"] = vad["end"]

    print(f"合并： {len(vad_list) - len(updated_list)} segments")

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    print(f"移除： {len(updated_list) - len(filter_list)} segments by length")

    # # 将基于VAD的细粒度分割结果保存为excel文件
    # filter_list_df = pd.DataFrame(filter_list)
    # filter_list_df.to_excel(f"outputs_in_progress/step4_VAD.xlsx", index=False)
    # print("step4: 基于VAD的细粒度分割结果已保存为excel文件")
    return filter_list

def asr(vad_segments, audio):
    if len(vad_segments) == 0:
        return []
    # 提取音频片段并计算帧，通过采样率将时间转为帧索引
    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]  # 去掉首尾的空白音频

    # 将每一个segment的绝对时间戳变为相对于第一个VAD片段的相对时间戳，可以简化后续流程
    for idx, segment in enumerate(vad_segments):
        vad_segments[idx]["start"] -= start_time
        vad_segments[idx]["end"] -= start_time

    # 因为ASR普遍支持16k的采样率，因此这里从24k转到16k
    temp_audio = librosa.resample(
        temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
    )

    if multilingual_flag:# 包含多种语言
        print("Multilingual flag is on")
        valid_vad_segments, valid_vad_segments_language = [], []
        # 对合法的segments进行进一步处理
        for idx, segment in enumerate(vad_segments):
            start_frame = int(segment["start"] * 16000)
            end_frame = int(segment["end"] * 16000)
            segment_audio = temp_audio[start_frame:end_frame]
            language, prob = asr_model.detect_language(segment_audio)
            # 检测到的语言在支持的列表中且置信度大于0.8
            if language in supported_languages and prob > 0.8:
                valid_vad_segments.append(vad_segments[idx])
                valid_vad_segments_language.append(language)

        if len(valid_vad_segments) == 0:
            return []
        all_transcribe_result = []
        print(f"valid_vad_segments_language: {valid_vad_segments_language}")
        unique_languages = list(set(valid_vad_segments_language))
        print(f"unique_languages: {unique_languages}")
        for language_token in unique_languages:
            language = language_token
            # 过滤出包含多种语言的segments
            vad_segments = [
                valid_vad_segments[i]
                for i, x in enumerate(valid_vad_segments_language)
                if x == language
            ]
            transcribe_result_temp = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=16,
                language=language,
                print_progress=True,
            )
            result = transcribe_result_temp["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
            all_transcribe_result.extend(result)
        # sort by start time
        all_transcribe_result = sorted(all_transcribe_result, key=lambda x: x["start"])
        return all_transcribe_result
    else:
        # 单语言
        print("Multilingual flag is off")
        language, prob = asr_model.detect_language(temp_audio)
        if language in supported_languages and prob > 0.8:
            transcribe_result = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=16,
                language=language,
                print_progress=True,
            )
            result = transcribe_result["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result["language"]
            return result
        else:
            return []

def mos_prediction(audio, vad_list):
    """
    预测给定音频和语音活动检测（VAD）片段的平均意见得分（MOS）。
    args：
    audio（字典）：一个包含音频波形和采样率的字典。
    vad_list（列表）：包含 VAD 片段起始和结束时间的列表。
    return：
    元组：一个包含平均 MOS 和带有 MOS 得分更新后的 VAD 片段的元组。

    """
    audio = audio["waveform"]
    sample_rate = 16000

    audio = librosa.resample(
        audio, orig_sr=24000, target_sr=sample_rate
    )

    for index, vad in enumerate(tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio[start:end]

        dnsmos = dnsmos_compute_score(segment, sample_rate, False)["OVRL"]

        vad_list[index]["dnsmos"] = dnsmos

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])

    # #保存经过MOS计算后的结果为excel文件
    # vad_list_df = pd.DataFrame(vad_list)
    # vad_list_df.to_excel(f"outputs_in_progress/step6_MOS.xlsx", index=False)
    # print("step6：MOS计算后的结果已保存为excel文件")

    return predict_dnsmos, vad_list

def filter(mos_list, dnsmos):
    """
    过滤掉具有 MOS 得分、wrong char duration和total duration的片段。
    args：
    mos_list（列表）：带有 MOS 得分的 VAD 片段列表。
    return：
    列表：一个包含 MOS 得分高于平均 MOS 的 VAD 片段的列表。
    """
    filtered_audio_stats, all_audio_stats = calculate_audio_stats(mos_list, dnsmos)
    filtered_segment = len(filtered_audio_stats)
    all_segment = len(all_audio_stats)
    print(
        f"> {all_segment - filtered_segment}/{all_segment} {(all_segment - filtered_segment) / all_segment:.2%} 的segments被过滤"
    )
    filtered_list = [mos_list[idx] for idx, _ in filtered_audio_stats]

    # #保存过滤后的结果为excel文件
    # filtered_list_df = pd.DataFrame(filtered_list)
    # filtered_list_df.to_excel(f"outputs_in_progress/step7_filtered.xlsx", index=False)
    # print("step7：过滤后的结果已保存为excel文件")

    return filtered_list


def main(audio_file, save_path, audio_name):
    audio = standardization(audio_file)
    vocal = vocal_extraction(audio)  # 提取人声部分
    if vocal is None:
        print("人声分离失败")
        return 0
    speakerdia = speaker_diarization(vocal)  # 进行说话人分段
    if speakerdia is None:
        print("说话人分段失败")
        return 0
    vad_list = vad.vad(speakerdia, vocal)  # 音频活动（说话人）检测
    # pprint( vad_list)
    segment_list = cut_by_speaker_label(vad_list)
    print('segment_list: ', segment_list)
    asr_result = asr(segment_list, vocal)

    # asr_result_df = pd.DataFrame(asr_result)
    # asr_result_df.to_excel(f"outputs_in_progress/step5_ASR.xlsx", index=False)
    # print("step5: ASR结果已保存为excel文件")

    print('asr_result: ', asr_result)
    avg_mos, mos_list = mos_prediction(vocal, asr_result)
    print("平均MOS预测值为：", avg_mos)
    filtered_list = filter(mos_list, avg_mos)
    print("已按各种条件过滤掉不合格的片段")
    print("过滤后的片段：", filtered_list)
    total_time_after_filter = 0
    for item in filtered_list:
        total_time_after_filter += item["end"] - item["start"]
    print("过滤后的总时长：", total_time_after_filter)

    print("Step 8: 将结果写入MP3和JSON")
    export_to_mp3(audio, filtered_list, save_path, audio_name)
    final_path = os.path.join(save_path, audio_name + ".json")
    with open(final_path, "w") as f:
        json.dump(filtered_list, f, ensure_ascii=False)

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
    print("加载VAD模型")
    diarization_pipeline.to(device)
    vad = silero_vad.SileroVAD(device=device)

    multilingual_flag = 'true'
    supported_languages = [
            "zh",
            "en",
            "fr",
            "ja",
            "ko",
            "de"
        ]
    print("加载ASR模型")
    asr_model = whisper_asr.load_asr_model(
        'medium',
        device_name,
        compute_type='float32',
        threads=4,
        asr_options={
            "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."
        },
    )

    print("加载MOS模型")
    mos_model_path = 'sig_bak_ovr.onnx'
    dnsmos_compute_score = dnsmos.ComputeScore(mos_model_path, device_name)

    main("audio_samples/audio1.wav", "outputs_in_progress/final_results", "audio1")