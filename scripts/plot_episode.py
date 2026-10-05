# 파일/폴더 경로를 다루는 파이썬 도구 (javascript의 path와 비슷)
from pathlib import Path

# 그래프를 그리는 파이썬 라이브러리 (javascript의 chart.js와 비슷)
import matplotlib.pyplot as plt

# 숫자 묶음인 텐서를 다루는 파이썬 라이브러리
import torch

# 데이터셋을 여는 파이썬 라이브러리
from lerobot.datasets import LeRobotDataset

# 0번째 시연 데이터셋
ds = LeRobotDataset("lerobot/svla_so101_pickplace", episodes=[0])

# 데이터셋에서 "숫자 표"를 읽어옴
# ds[i]로 사진을 꺼내면 오래걸리기 때문에
# 그래프를 그리는데에 사진은 필요가 없어서 표만 쓴다.
table = ds.hf_dataset

state = torch.stack(list(table["observation.state"]))

action = torch.stack(list(table["action"]))

time = torch.stack(list(table["timestamp"]))

names = ds.features["observation.state"]["names"]

print(state.shape, action.shape, time.shape)

print(names)

fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(time, state[:, i], label="state")
    ax.plot(time, action[:, i], label="action", linestyle="--")
    ax.set_ylabel(names[i])

axes[0].legend()

axes[-1].set_xlabel("time (s)")

plt.tight_layout()

out_dir = Path("outputs")

out_dir.mkdir(exist_ok=True)

plt.savefig(out_dir / "episode0_joints.png")

print("저장: ", out_dir / "episode0_joints.png")
