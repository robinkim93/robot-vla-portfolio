from lerobot.datasets import LeRobotDataset
import torch

ds = LeRobotDataset("lerobot/svla_so101_pickplace", episodes=[0])

stats = ds.meta.stats["observation.state"]

names = ds.features["observation.state"]["names"]

print(f"{'joint':20s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>8s}")

for i, name in enumerate(names):
    print(f"{name:20s} " f"{stats['min'][i]:>8.2f} " f"{stats['max'][i]:>8.2f} " f"{stats['mean'][i]:>8.2f} " f"{stats['std'][i]:>8.2f}")

    print()
    print("count:", stats['count'])

    
print()
state0 = ds[0]["observation.state"]

mean = torch.tensor(stats['mean'])
std = torch.tensor(stats['std'])

normalized = (state0 - mean) / std

for i, name in enumerate(names):
    print(f"{name:20s} {state0[i]:>8.2f} -> {normalized[i]:>6.2f}")