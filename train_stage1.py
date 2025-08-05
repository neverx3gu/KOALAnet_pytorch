import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from models import DownsamplingNetwork, local_conv_ds, kernel_normalize
from dataset import SISRDataset

# --- 1. 커스텀 Sampler 정의 ---
class EpochBasedSampler(Sampler):
    def __init__(self, data_source, num_samples_per_epoch):
        self.data_source = data_source
        self.num_samples_per_epoch = num_samples_per_epoch

    def __iter__(self):
        indices = torch.randint(high=len(self.data_source), size=(self.num_samples_per_epoch,)).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_epoch


# --- 2. 하이퍼파라미터 및 설정 ---
HR_DATA_DIR = './data/DIV2K_train_HR'
MODEL_SAVE_PATH = './koalanet_stage1.pth'

SCALE_FACTOR = 4
PATCH_SIZE_LR = 64
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

TOTAL_ITERATIONS = 20000
UPDATES_PER_EPOCH = 100
EPOCHS = TOTAL_ITERATIONS // UPDATES_PER_EPOCH  # 200
DOWNSAMPLING_KERNEL_SIZE = 20

LR_DECAY_MILESTONES = [int(0.8 * EPOCHS), int(0.9 * EPOCHS)]
LR_DECAY_FACTOR = 0.1


def main():
    # --- 3. 훈련 준비 ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    dataset = SISRDataset(hr_dir=HR_DATA_DIR, scale_factor=SCALE_FACTOR, patch_size_lr=PATCH_SIZE_LR)

    num_samples_per_epoch = UPDATES_PER_EPOCH * BATCH_SIZE
    epoch_sampler = EpochBasedSampler(dataset, num_samples_per_epoch)

    pin_memory_flag = True if device.type == 'cuda' else False
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=epoch_sampler,
        shuffle=False, num_workers=12, pin_memory=pin_memory_flag, persistent_workers=True
    )

    model = DownsamplingNetwork(kernel_size=DOWNSAMPLING_KERNEL_SIZE).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY_MILESTONES, gamma=LR_DECAY_FACTOR)

    # --- 4. 훈련 루프 ---
    print("Downsampling Network 훈련을 시작합니다...")
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"epoch [{epoch + 1}/{EPOCHS}]")

        for lr_imgs, hr_imgs, gt_kernels in progress_bar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            gt_kernels = gt_kernels.to(device)

            optimizer.zero_grad()

            # 순전파: LR 이미지로부터 커널 맵 예측
            pred_kernel_map = model(lr_imgs)

            # --- 손실 계산 (논문 로직) ---
            # 1. 재구성 손실 (정규화 X)
            reconstructed_lr = local_conv_ds(hr_imgs, pred_kernel_map, SCALE_FACTOR, DOWNSAMPLING_KERNEL_SIZE)
            loss_recon = criterion(reconstructed_lr, lr_imgs)

            # 2. 커널 손실 (정규화 O)
            B, _, H_lr, W_lr = pred_kernel_map.shape
            pred_kernel_map_norm = pred_kernel_map.view(B, DOWNSAMPLING_KERNEL_SIZE ** 2, H_lr, W_lr)
            pred_kernel_map_norm = kernel_normalize(pred_kernel_map_norm, DOWNSAMPLING_KERNEL_SIZE)
            pred_kernel_mean = torch.mean(pred_kernel_map_norm, dim=[2, 3], keepdim=True)
            gt_kernel_flat = gt_kernels.view(B, DOWNSAMPLING_KERNEL_SIZE ** 2, 1, 1)
            loss_kernel = criterion(pred_kernel_mean, gt_kernel_flat)

            # 3. 최종 손실
            total_loss = loss_recon + loss_kernel

            total_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=total_loss.item(), recon=loss_recon.item(), kernel=loss_kernel.item())

        scheduler.step()

    print("훈련 완료!")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"모델이 '{MODEL_SAVE_PATH}' 파일로 저장되었습니다.")


if __name__ == '__main__':
    main()