import os
import itertools
import random
import librosa
import soundfile as sf


def generate_mixed_audio(data_dir, output_dir, samples_per_combination=2000):
    """生成混合音频及分离源，每个乐器组合生成指定数量的样本"""
    classes = sorted(os.listdir(data_dir))  # 获取排序后的乐器类别
    combinations = list(itertools.combinations(classes, 2))  # 生成所有两两组合
    os.makedirs(output_dir, exist_ok=True)

    # 清空或创建标签文件
    with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
        pass

    sample_index = 0  # 全局样本计数器

    for inst1, inst2 in combinations:
        for _ in range(samples_per_combination):
            # 随机选择两个乐器的音频文件
            inst1_path = os.path.join(data_dir, inst1, random.choice(os.listdir(os.path.join(data_dir, inst1))))
            inst2_path = os.path.join(data_dir, inst2, random.choice(os.listdir(os.path.join(data_dir, inst2))))

            # 加载2秒音频
            audio1, sr = librosa.load(inst1_path, sr=16000, duration=2)
            audio2, _ = librosa.load(inst2_path, sr=16000, duration=2)

            # 确保长度一致
            min_len = min(len(audio1), len(audio2))
            mixed = audio1[:min_len] + audio2[:min_len]

            # 创建样本目录
            sample_dir = os.path.join(output_dir, f'sample_{sample_index}')
            os.makedirs(sample_dir, exist_ok=True)

            # 保存音频文件
            sf.write(os.path.join(sample_dir, 'mixed.wav'), mixed, sr)
            sf.write(os.path.join(sample_dir, 'source1.wav'), audio1[:min_len], sr)
            sf.write(os.path.join(sample_dir, 'source2.wav'), audio2[:min_len], sr)

            # 写入标签信息
            with open(os.path.join(output_dir, 'labels.txt'), 'a') as f:
                f.write(f'{sample_dir}/mixed.wav {sample_dir}/source1.wav {sample_dir}/source2.wav {inst1} {inst2}\n')

            sample_index += 1


if __name__ == '__main__':
    generate_mixed_audio('./raw_data/', 'train_data/')