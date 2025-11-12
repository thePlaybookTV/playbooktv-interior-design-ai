#!/usr/bin/env python3
import torch

checkpoint = torch.load("models_best_interior_model.pth", map_location='cpu')
state_dict = checkpoint['model_state_dict']

print("All layer names:")
print("=" * 80)
for i, key in enumerate(state_dict.keys(), 1):
    print(f"{i:3d}. {key:50s} {str(state_dict[key].shape):20s}")
