#!/usr/bin/env python3
"""
Inspect Phase 1 Model Structure
Quick script to understand models_best_interior_model.pth
"""

import torch

print("=" * 80)
print("PHASE 1 MODEL INSPECTOR")
print("=" * 80)

# Load the checkpoint
model_path = "models_best_interior_model.pth"
print(f"\n📦 Loading: {model_path}")

checkpoint = torch.load(model_path, map_location='cpu')

print("\n" + "=" * 80)
print("CHECKPOINT KEYS")
print("=" * 80)
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, (int, float)):
        print(f"  {key}: {value}")
    elif isinstance(value, list):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: <{type(value).__name__}>")

print("\n" + "=" * 80)
print("TRAINING METADATA")
print("=" * 80)

if 'epoch' in checkpoint:
    print(f"Epoch: {checkpoint['epoch']}")

if 'val_loss' in checkpoint:
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")

if 'val_room_acc' in checkpoint:
    print(f"Room Accuracy: {checkpoint['val_room_acc']:.2%}")

if 'val_style_acc' in checkpoint:
    print(f"Style Accuracy: {checkpoint['val_style_acc']:.2%}")

print("\n" + "=" * 80)
print("ROOM TYPES")
print("=" * 80)
if 'room_types' in checkpoint:
    room_types = checkpoint['room_types']
    print(f"Total: {len(room_types)}")
    for i, room in enumerate(room_types, 1):
        print(f"  {i}. {room}")

print("\n" + "=" * 80)
print("STYLES")
print("=" * 80)
if 'styles' in checkpoint:
    styles = checkpoint['styles']
    print(f"Total: {len(styles)}")
    for i, style in enumerate(styles, 1):
        print(f"  {i}. {style}")

print("\n" + "=" * 80)
print("MODEL ARCHITECTURE")
print("=" * 80)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    
    print(f"\nTotal parameters: {len(state_dict)} layers")
    
    # Group by module
    backbone_params = [k for k in state_dict.keys() if 'backbone' in k]
    room_params = [k for k in state_dict.keys() if 'room_head' in k]
    style_params = [k for k in state_dict.keys() if 'style_head' in k]
    
    print(f"\nBackbone layers: {len(backbone_params)}")
    print(f"Room head layers: {len(room_params)}")
    print(f"Style head layers: {len(style_params)}")
    
    # Show first few layers of each section
    print("\nBackbone (first 5 layers):")
    for key in list(backbone_params)[:5]:
        print(f"  {key}: {state_dict[key].shape}")
    if len(backbone_params) > 5:
        print(f"  ... and {len(backbone_params) - 5} more layers")
    
    print("\nRoom head layers:")
    for key in room_params:
        print(f"  {key}: {state_dict[key].shape}")
    
    print("\nStyle head layers:")
    for key in style_params:
        print(f"  {key}: {state_dict[key].shape}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✅ Model Type: Multi-task (Room + Style)")
print(f"✅ Architecture: ResNet50 backbone with dual heads")
print(f"✅ Room Types: {len(checkpoint.get('room_types', []))}")
print(f"✅ Styles: {len(checkpoint.get('styles', []))}")
print(f"✅ Room Accuracy: {checkpoint.get('val_room_acc', 0):.2%}")
print(f"✅ Style Accuracy: {checkpoint.get('val_style_acc', 0):.2%}")
print("=" * 80)
