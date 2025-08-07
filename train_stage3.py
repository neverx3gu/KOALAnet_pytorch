import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

# models.py에서 필요한 모든 모델과 함수를 불러옵니다.
from models import DownsamplingNetwork, UpsamplingNetwork, local_conv_ds, kernel_normalize
from dataset import SISRDataset


# --- 1. 커스텀 Sampler 정의 (이전과 동일) ---
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
# Stage 1, 2에서 저장한 가중치 파일 경로
STAGE1_WEIGHTS_PATH = 'pth/koalanet_stage1.pth'
STAGE2_WEIGHTS_PATH = 'pth/koalanet_stage2.pth'
MODEL_SAVE_PATH = 'pth/koalanet_stage3_final.pth'

SCALE_FACTOR = 4
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

    # Stage 3 훈련을 위해 커널을 반환하는 dataset.py가 필요합니다.
    dataset = SISRDataset(hr_dir=HR_DATA_DIR, scale_factor=SCALE_FACTOR, patch_size_lr=64)
    num_samples_per_epoch = UPDATES_PER_EPOCH * BATCH_SIZE
    epoch_sampler = EpochBasedSampler(dataset, num_samples_per_epoch)

    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=epoch_sampler,
        shuffle=False, num_workers=12, pin_memory=False, persistent_workers=True
    )

    # 두 개의 네트워크를 모두 생성합니다.
    downsampling_net = DownsamplingNetwork(kernel_size=DOWNSAMPLING_KERNEL_SIZE).to(device)
    upsampling_net = UpsamplingNetwork(scale_factor=SCALE_FACTOR).to(device)

    # --- 4. 미리 훈련된 가중치 불러오기 ---
    print("미리 훈련된 Stage 1, 2 가중치를 불러옵니다...")
    downsampling_net.load_state_dict(torch.load(STAGE1_WEIGHTS_PATH, map_location=device))
    # Stage 2 (Baseline)의 가중치를 Stage 3 모델에 불러옵니다.
    # KOALA 모듈 관련 가중치는 없으므로, strict=False로 설정하여 에러를 방지합니다.
    upsampling_net.load_state_dict(torch.load(STAGE2_WEIGHTS_PATH, map_location=device), strict=False)
    print("가중치 로딩 완료!")

    # 옵티마이저는 이제 두 네트워크의 모든 파라미터를 함께 학습합니다.
    all_params = list(downsampling_net.parameters()) + list(upsampling_net.parameters())
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE)

    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY_MILESTONES, gamma=LR_DECAY_FACTOR)

    # --- 5. 훈련 루프 ---
    print("Stage 3: KOALAnet 공동 훈련을 시작합니다...")
    for epoch in range(EPOCHS):
        downsampling_net.train()
        upsampling_net.train()

        progress_bar = tqdm(data_loader, desc=f"epoch [{epoch + 1}/{EPOCHS}]")

        for lr_imgs, hr_imgs, gt_kernels in progress_bar:
            lr_imgs, hr_imgs, gt_kernels = lr_imgs.to(device), hr_imgs.to(device), gt_kernels.to(device)
            optimizer.zero_grad()

            # --- 순전파 ---
            # 1. Downsampling Network가 LR 이미지로부터 커널 맵(F_d) 예측
            pred_kernel_map = downsampling_net(lr_imgs)

            # 2. Upsampling Network가 LR 이미지와 예측된 커널 맵을 모두 입력받아 SR 이미지(Y_hat) 생성
            sr_imgs = upsampling_net(lr_imgs, pred_kernel_map)

            # --- 손실 계산 (논문 Stage 3 로직) ---
            # 1. SR 재구성 손실: l1(Y_hat, Y)
            loss_sr = criterion(sr_imgs, hr_imgs)

            # 2. LR 재구성 손실: l1(X_hat, X)
            reconstructed_lr = local_conv_ds(hr_imgs, pred_kernel_map, SCALE_FACTOR, DOWNSAMPLING_KERNEL_SIZE)
            loss_lr = criterion(reconstructed_lr, lr_imgs)

            # 3. 최종 손실
            total_loss = loss_sr + loss_lr

            total_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=total_loss.item(), sr_loss=loss_sr.item(), lr_loss=loss_lr.item())

        scheduler.step()

    print("훈련 완료!")
    torch.save({
        'downsampling_net': downsampling_net.state_dict(),
        'upsampling_net': upsampling_net.state_dict(),
    }, MODEL_SAVE_PATH)
    print(f"최종 모델이 '{MODEL_SAVE_PATH}' 파일로 저장되었습니다.")


if __name__ == '__main__':
    main()