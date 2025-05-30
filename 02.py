import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import librosa
from skimage.transform import resize
import torchvision
from tqdm import tqdm
import os

os.makedirs('./logs', exist_ok=True)


# 修改点：移除数据增强后的数据集类
class MixedAudioDataset(Dataset):
    def __init__(self, label_file):
        self.samples = []
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"数据格式错误：第{line_num}行应有5列，实际得到{len(parts)}列。"
                        f"问题行内容：'{line}'"
                    )
                mixed_path, source1_path, source2_path, label1, label2 = parts
                self.samples.append((mixed_path, source1_path, source2_path, label1, label2))
        self.classes = sorted(
            list(set([label for _, _, _, label1, label2 in self.samples for label in (label1, label2)])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mixed_path, source1_path, source2_path, label1, label2 = self.samples[idx]

        def compute_mel(path):
            wav, sr = librosa.load(path, sr=16000)
            wav = librosa.util.fix_length(wav, size=16000 * 2)  # 仅保留基础预处理

            # 移除所有数据增强操作（时移/噪声等）
            mel = librosa.feature.melspectrogram(
                y=wav, sr=sr,
                n_mels=128,
                hop_length=512,
                n_fft=2048
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            mel_min = np.min(mel)
            mel_max = np.max(mel)
            if mel_max != mel_min:
                mel = (mel - mel_min) / (mel_max - mel_min)  # 归一化到[0, 1]
            else:
                mel = np.zeros_like(mel)  # 处理全零情况
            if mel.shape != (128, 128):
                mel = resize(mel, (128, 128))
            return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        def process_source(path):
            wav, sr = librosa.load(path, sr=16000)
            wav = librosa.util.fix_length(wav, size=16000 * 2)
            return wav

        # 加载两路源信号
        source1_wav = process_source(source1_path)
        source2_wav = process_source(source2_path)

        # 计算能量并调整增益
        rms1 = np.sqrt(np.mean(source1_wav ** 2)) + 1e-8  # 均方根振幅
        rms2 = np.sqrt(np.mean(source2_wav ** 2)) + 1e-8
        gain1 = rms2 / rms1
        gain2 = rms1 / rms2
        balanced_source1 = source1_wav * gain1
        balanced_source2 = source2_wav * gain2

        # 混合信号
        mixed_wav = balanced_source1 + balanced_source2

        mixed_mel = compute_mel(mixed_wav)
        source1_mel = compute_mel(balanced_source1)
        source2_mel = compute_mel(balanced_source2)

        label1_idx = self.classes.index(label1)
        label2_idx = self.classes.index(label2)
        return mixed_mel, source1_mel, source2_mel, label1_idx, label2_idx


# 保留所有正则化改进的U-Net
class EnhancedUNet(nn.Module):
    def __init__(self, num_sources=2):
        super().__init__()
        # 编码器结构不变
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 解码器保留Dropout层
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, num_sources, 2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(num_sources, num_sources, 3, padding=1)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))
        return dec1


class JointModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.unet = EnhancedUNet(num_sources=2)
        self.classifier = torchvision.models.resnet50(pretrained=False)
        self.classifier.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        separated_specs = self.unet(x)
        outputs = []
        for i in range(separated_specs.shape[1]):
            source = separated_specs[:, i, :, :].unsqueeze(1)
            outputs.append(self.classifier(source))
        return outputs, separated_specs


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MixedAudioDataset('train_data/labels.txt')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # 保留正则化和学习率策略
    model = JointModel(num_classes=len(dataset.classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)  # L2正则化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    separation_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    no_improve_epochs = 0
    patience = 4

    for epoch in range(30):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1:02d}',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for batch_idx, (mixed, source1, source2, label1, label2) in enumerate(progress_bar):
            mixed = mixed.to(device, non_blocking=True)
            source1 = source1.to(device, non_blocking=True)
            source2 = source2.to(device, non_blocking=True)
            label1 = label1.to(device, non_blocking=True)
            label2 = label2.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs, separated_specs = model(mixed)

            loss_sep = separation_criterion(separated_specs[:, 0], source1) + \
                       separation_criterion(separated_specs[:, 1], source2)
            loss_cls = classification_criterion(outputs[0], label1) + \
                       classification_criterion(outputs[1], label2)
            current_loss = 0.7 * loss_sep + 0.3 * loss_cls

            current_loss.backward()
            optimizer.step()

            running_loss += current_loss.item()
            _, pred1 = torch.max(outputs[0], 1)
            _, pred2 = torch.max(outputs[1], 1)
            train_correct += (pred1 == label1).sum().item() + (pred2 == label2).sum().item()
            train_total += label1.size(0) * 2

            avg_loss = running_loss / (batch_idx + 1)
            current_acc = train_correct / train_total if train_total > 0 else 0.0
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{current_acc:.2%}',
                'GPU Mem': f'{torch.cuda.memory_allocated() / 1e9:.1f}G'
            })

        scheduler.step(current_loss)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for mixed, source1, source2, label1, label2 in test_loader:
                mixed = mixed.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)

                outputs, _ = model(mixed)
                _, pred1 = torch.max(outputs[0], 1)
                _, pred2 = torch.max(outputs[1], 1)
                test_correct += (pred1 == label1).sum().item() + (pred2 == label2).sum().item()
                test_total += label1.size(0) * 2

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        print(
            f'\nEpoch {epoch + 1:02d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')


    torch.save(model.state_dict(), 'logs/m2.pth')