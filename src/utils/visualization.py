"""Visualization utilities for interior design analysis"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import duckdb
from PIL import Image
import numpy as np
from typing import Optional


def visualize_pristine(
    db_path: str, 
    num_samples: int = 6,
    output_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Visualize images with SAM2 masks + bboxes from DuckDB database.
    
    Args:
        db_path: Path to DuckDB database file
        num_samples: Number of sample images to visualize
        output_path: Path to save visualization (default: 'pristine_visualization.png')
        show_plot: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    conn = duckdb.connect(db_path)
    
    # Get random samples
    samples = conn.execute("""
        SELECT 
            i.image_id,
            i.original_path,
            i.room_type,
            i.style,
            i.furniture_count
        FROM images i
        WHERE i.furniture_count > 0
        ORDER BY RANDOM()
        LIMIT ?
    """, (num_samples,)).df()
    
    if len(samples) == 0:
        print("⚠️ No images with furniture detections found in database")
        conn.close()
        return None
    
    # Calculate grid size
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.ravel()
    
    for idx, row in samples.iterrows():
        if idx >= len(axes):
            break
        
        # Load image
        try:
            img = Image.open(row['original_path'])
            axes[idx].imshow(img)
        except Exception as e:
            print(f"⚠️ Could not load image {row['original_path']}: {e}")
            axes[idx].text(0.5, 0.5, 'Image not found', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
            continue
        
        # Get detections
        detections = conn.execute("""
            SELECT * FROM furniture_detections
            WHERE image_id = ?
        """, (row['image_id'],)).df()
        
        # Draw bboxes
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for det_idx, det in detections.iterrows():
            color = colors[det_idx % len(colors)]
            
            # Draw bbox
            try:
                rect = Rectangle(
                    (det['bbox_x1'], det['bbox_y1']),
                    det['bbox_x2'] - det['bbox_x1'],
                    det['bbox_y2'] - det['bbox_y1'],
                    linewidth=3,
                    edgecolor=color,
                    facecolor='none'
                )
                axes[idx].add_patch(rect)
                
                # Label
                label = f"{det['item_type']}"
                if det.get('has_mask', False):
                    label += " 🎭"  # Mask indicator
                
                axes[idx].text(
                    det['bbox_x1'], det['bbox_y1'] - 5,
                    label,
                    color=color,
                    fontsize=9,
                    weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
            except Exception as e:
                print(f"⚠️ Error drawing bbox: {e}")
                continue
        
        axes[idx].axis('off')
        axes[idx].set_title(
            f"{row['room_type']} | {row['style']}\n{row['furniture_count']} items (SAM2 enabled)",
            fontsize=10,
            weight='bold'
        )
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Pristine MVP: YOLO Bboxes + SAM2 Segmentation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if output path provided
    if output_path is None:
        output_path = 'pristine_visualization.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    conn.close()
    
    print(f"✅ Saved: {output_path}")
    return fig