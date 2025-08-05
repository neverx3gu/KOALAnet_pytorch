import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from models import DownsamplingNetwork, local_conv_ds, kernel_normalize
from dataset import SISRDataset

# --- 1. 하이퍼파라미터 및 설정 ---
HR_DATA_DIR = './data/DIV2K_train_HR'
MODEL_SAVE_PATH = './koalanet_stage1.pth'

SCALE_FACTOR = 4
PATCH_SIZE_LR = 64
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-4
LR_DECAY_MILESTONES = [int(0.8 * EPOCHS), int(0.9 * EPOCHS)]
LR_DECAY_FACTOR = 0.1
DOWNSAMPLING_KERNEL_SIZE = 20


def main():
    # --- 2. 훈련 준비 ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    dataset = SISRDataset(hr_dir=HR_DATA_DIR, scale_factor=SCALE_FACTOR, patch_size_lr=PATCH_SIZE_LR)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False,
                             persistent_workers=True)

    # DownsamplingNetwork 모델을 불러옵니다.
    model = DownsamplingNetwork(kernel_size=DOWNSAMPLING_KERNEL_SIZE).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY_MILESTONES, gamma=LR_DECAY_FACTOR)

    # --- 3. 훈련 루프 ---
    print("Downsampling Network 훈련을 시작합니다...")
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"에포크 [{epoch + 1}/{EPOCHS}]")

        for lr_imgs, hr_imgs, gt_kernels in progress_bar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            gt_kernels = gt_kernels.to(device)

            optimizer.zero_grad()

            # 순전파: LR 이미지로부터 커널 맵 예측
            pred_kernel_map = model(lr_imgs)

            # --- 손실 계산 (논문 로직) ---
            # 1. 재구성 손실 (정규화 X)
            # 예측된 커널(정규화 안된)로 HR 이미지를 다운샘플링하여 LR 이미지를 재구성
            reconstructed_lr = local_conv_ds_pytorch(hr_imgs, pred_kernel_map, SCALE_FACTOR, DOWNSAMPLING_KERNEL_SIZE)
            loss_recon = criterion(reconstructed_lr, lr_imgs)

            # 2. 커널 손실 (정규화 O)
            # 예측된 커널을 정규화
            B, _, H, W = pred_kernel_map.shape
            pred_kernel_map_norm = pred_kernel_map.view(B, DOWNSAMPLING_KERNEL_SIZE ** 2, H, W)
            pred_kernel_map_norm = kernel_normalize(pred_kernel_map_norm, DOWNSAMPLING_KERNEL_SIZE)
            # 공간적 평균을 내고, 정답 커널과 비교
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