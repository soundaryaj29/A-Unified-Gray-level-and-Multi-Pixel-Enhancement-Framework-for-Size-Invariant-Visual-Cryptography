import cv2
import numpy as np
from typing import List, Dict
import pickle
import os
import traceback

# ============================================================================
# MULTIPIXEL BLOCK - COMPLETE IMPLEMENTATION
# ============================================================================
def multipixel_block(enhanced_image: np.ndarray, 
                    contrast_map: np.ndarray) -> Dict:
    """
    COMPLETE MULTIPIXEL BLOCK
    Processes 256×256 image into 2×2 blocks with statistics
    
    Input: Enhanced grayscale image + Contrast map (both 256×256)
    Output: Block statistics for Gray-Level Analysis
    """
    
    print("\n" + "="*70)
    print("MULTIPIXEL BLOCK - Processing 256×256 Image")
    print("="*70)
    
    # =====================================================
    # STEP 1: DIVIDE IMAGE INTO 2×2 NON-OVERLAPPING BLOCKS
    # =====================================================
    height, width = 256, 256
    
    # Ensure correct size
    if enhanced_image.shape != (height, width):
        print(f"⚠️  Resizing enhanced image to 256×256")
        enhanced_image = cv2.resize(enhanced_image, (width, height))
    
    if contrast_map.shape != (height, width):
        print(f"⚠️  Resizing contrast map to 256×256")
        contrast_map = cv2.resize(contrast_map, (width, height))
    
    block_size = 2
    blocks = []  # Store all blocks
    positions = []  # Store block positions
    
    print(f"\n[STEP 1] Dividing {height}×{width} image into {block_size}×{block_size} blocks...")
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Extract 2×2 block from enhanced image
            pixel_block = enhanced_image[y:y+block_size, x:x+block_size]
            
            # Store block and its position
            blocks.append(pixel_block)
            positions.append((y, x))
    
    total_blocks = len(blocks)
    print(f"✓ Created {total_blocks} blocks")
    print(f"✓ Each block: {block_size}×{block_size} = {block_size*block_size} pixels")
    
    # =====================================================
    # STEP 2: EXTRACT PIXEL VALUES FOR EACH BLOCK
    # =====================================================
    print(f"\n[STEP 2] Extracting pixel values...")
    pixel_values = []
    
    for block in blocks:
        # Convert numpy array to Python list
        pixel_values.append(block.tolist())
    
    print(f"✓ Extracted pixel values from {len(pixel_values)} blocks")
    
    # =====================================================
    # STEP 3: EXTRACT CONTRAST MAP VALUES
    # =====================================================
    print(f"\n[STEP 3] Extracting contrast map values...")
    contrast_values = []
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Extract 2×2 contrast values
            contrast_block = contrast_map[y:y+block_size, x:x+block_size]
            contrast_values.append(contrast_block.tolist())
    
    print(f"✓ Extracted contrast values from {len(contrast_values)} blocks")
    
    # =====================================================
    # STEP 4: CALCULATE STATISTICS FOR EACH BLOCK
    # =====================================================
    print(f"\n[STEP 4] Calculating block statistics...")
    
    # Initialize lists for statistics
    avg_gray_list = []
    contrast_weight_list = []
    local_range_list = []
    variance_list = []
    
    for i, block in enumerate(blocks):
        # 4A: AVERAGE GRAY VALUE = mean of 4 pixels
        avg_gray = float(np.mean(block))
        avg_gray_list.append(avg_gray)
        
        # 4C: LOCAL RANGE = max pixel - min pixel
        local_range = float(np.max(block) - np.min(block))
        local_range_list.append(local_range)
        
        # 4D: VARIANCE = spread of pixel values
        variance = float(np.var(block))
        variance_list.append(variance)
    
    for i, contrast_block in enumerate(contrast_values):
        # 4B: CONTRAST WEIGHT = mean of contrast map over block
        contrast_array = np.array(contrast_block)
        contrast_weight = float(np.mean(contrast_array))
        contrast_weight_list.append(contrast_weight)
    
    print(f"✓ Calculated all 4 statistics for {total_blocks} blocks")
    print(f"  • Average gray: {len(avg_gray_list)} values")
    print(f"  • Contrast weight: {len(contrast_weight_list)} values")
    print(f"  • Local range: {len(local_range_list)} values")
    print(f"  • Variance: {len(variance_list)} values")
    
    # =====================================================
    # COMPILE RESULTS
    # =====================================================
    print(f"\n[COMPILING RESULTS] Creating output structure...")
    
    # Create detailed block data
    blocks_data = []
    
    for i in range(total_blocks):
        block_info = {
            'block_id': i,
            'position': positions[i],  # (y, x) coordinates
            'block_coords': (positions[i][0] // 2, positions[i][1] // 2),
            'pixel_values': pixel_values[i],
            'contrast_values': contrast_values[i],
            'avg_gray': avg_gray_list[i],
            'contrast_weight': contrast_weight_list[i],
            'local_range': local_range_list[i],
            'variance': variance_list[i],
            'std_dev': float(np.sqrt(variance_list[i]))  # Additional
        }
        blocks_data.append(block_info)
    
    # Calculate global statistics
    global_stats = {
        'total_blocks': total_blocks,
        'blocks_h': height // block_size,  # 128
        'blocks_w': width // block_size,   # 128
        'block_size': block_size,
        'compression_ratio': f"{block_size*block_size}:1",
        
        'avg_gray_stats': {
            'mean': float(np.mean(avg_gray_list)),
            'min': float(np.min(avg_gray_list)),
            'max': float(np.max(avg_gray_list)),
            'std': float(np.std(avg_gray_list))
        },
        
        'contrast_stats': {
            'mean': float(np.mean(contrast_weight_list)),
            'min': float(np.min(contrast_weight_list)),
            'max': float(np.max(contrast_weight_list)),
            'high_count': sum(1 for w in contrast_weight_list if w > 0.7),
            'medium_count': sum(1 for w in contrast_weight_list if 0.4 <= w <= 0.7),
            'low_count': sum(1 for w in contrast_weight_list if w < 0.4)
        }
    }
    
    # Final output structure
    output = {
        'blocks_data': blocks_data,      # List of all blocks with statistics
        'global_stats': global_stats,    # Overall image statistics
        'statistics_arrays': {           # For easy access in next module
            'avg_gray': avg_gray_list,
            'contrast_weight': contrast_weight_list,
            'local_range': local_range_list,
            'variance': variance_list
        },
        'metadata': {
            'module': 'Multipixel Block',
            'input_size': f"{height}×{width}",
            'output_size': f"{total_blocks} blocks",
            'next_module': 'Gray-Level Analysis'
        }
    }
    
    # =====================================================
    # DISPLAY SUMMARY
    # =====================================================
    print(f"\n" + "="*70)
    print("MULTIPIXEL BLOCK - COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print(f"\n📊 OUTPUT SUMMARY:")
    print(f"   • Total blocks created: {total_blocks:,}")
    print(f"   • Block grid: {global_stats['blocks_h']}×{global_stats['blocks_w']}")
    print(f"   • Compression: {global_stats['compression_ratio']} (pixels:block)")
    
    print(f"\n📈 IMAGE STATISTICS:")
    print(f"   • Average gray: {global_stats['avg_gray_stats']['mean']:.1f} "
          f"(range: {global_stats['avg_gray_stats']['min']:.1f}-{global_stats['avg_gray_stats']['max']:.1f})")
    print(f"   • Contrast weight: {global_stats['contrast_stats']['mean']:.3f}")
    print(f"   • High contrast blocks: {global_stats['contrast_stats']['high_count']:,} "
          f"({global_stats['contrast_stats']['high_count']/total_blocks*100:.1f}%)")
    print(f"   • Medium contrast blocks: {global_stats['contrast_stats']['medium_count']:,}")
    print(f"   • Low contrast blocks: {global_stats['contrast_stats']['low_count']:,}")
    
    print(f"\n🔍 SAMPLE BLOCK (First block):")
    sample = blocks_data[0]
    print(f"   • Position: {sample['position']}")
    print(f"   • Pixels: {sample['pixel_values']}")
    print(f"   • Contrast: {sample['contrast_values']}")
    print(f"   • Avg Gray: {sample['avg_gray']:.2f}")
    print(f"   • Contrast Weight: {sample['contrast_weight']:.3f}")
    print(f"   • Local Range: {sample['local_range']:.1f}")
    print(f"   • Variance: {sample['variance']:.2f}")
    
    print(f"\n✅ READY FOR: Gray-Level Analysis Block")
    
    return output


# ============================================================================
# SAVE OUTPUT FUNCTION
# ============================================================================
def save_multipixel_output(output: Dict, filename: str = "multipixel_output.pkl"):
    """
    Save multipixel block output to file
    """
    with open(filename, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\n💾 SAVED OUTPUT:")
    print(f"   • File: {filename}")
    print(f"   • Size: {len(pickle.dumps(output)) / 1024:.1f} KB")
    print(f"   • Contains: {len(output['blocks_data'])} blocks")
    print(f"   • Ready for Gray-Level Analysis")
    
    return filename


# ============================================================================
# LOAD OUTPUT FUNCTION (For Gray-Level Analysis)
# ============================================================================
def load_multipixel_output(filename: str = "multipixel_output.pkl") -> Dict:
    """
    Load multipixel block output (for next module)
    """
    with open(filename, 'rb') as f:
        output = pickle.load(f)
    
    print(f"📂 Loaded multipixel output from {filename}")
    print(f"   • Blocks: {output['global_stats']['total_blocks']:,}")
    print(f"   • Grid: {output['global_stats']['blocks_h']}×{output['global_stats']['blocks_w']}")
    
    return output


# ============================================================================
# MAIN EXECUTION BLOCK - USING YOUR FILES
# ============================================================================
if __name__ == "__main__":
    """
    MAIN EXECUTION - USES YOUR ACTUAL PREPROCESSING OUTPUT FILES
    """
    
    print("="*70)
    print("MULTIPIXEL BLOCK - EXECUTION")
    print("="*70)
    
    print("📂 Loading your preprocessing output files...")
    
    try:
        # ==============================================
        # LOAD YOUR PREPROCESSING OUTPUT FILES
        # ==============================================
        # Load enhanced grayscale image
        enhanced_img = cv2.imread("enhanced_image.png", cv2.IMREAD_GRAYSCALE)
        if enhanced_img is None:
            raise FileNotFoundError("❌ enhanced_image.png not found in current directory")
        
        # Load contrast map (convert to float 0-1 range)
        contrast_map_img = cv2.imread("contrast_map.png", cv2.IMREAD_GRAYSCALE)
        if contrast_map_img is None:
            raise FileNotFoundError("❌ contrast_map.png not found in current directory")
        
        print(f"✓ Loaded enhanced_image.png: {enhanced_img.shape}, dtype: {enhanced_img.dtype}")
        print(f"✓ Loaded contrast_map.png: {contrast_map_img.shape}, dtype: {contrast_map_img.dtype}")
        
        # Convert contrast map to float32 in range 0.0-1.0
        if contrast_map_img.dtype != np.float32:
            contrast_map = contrast_map_img.astype(np.float32) / 255.0
            print(f"✓ Converted contrast map to float32 range [0.0, 1.0]")
        else:
            contrast_map = contrast_map_img
        
        # ==============================================
        # RUN MULTIPIXEL BLOCK PROCESSING
        # ==============================================
        print("\n" + "="*70)
        print("PROCESSING IMAGE...")
        print("="*70)
        
        multipixel_output = multipixel_block(enhanced_img, contrast_map)
        
        # ==============================================
        # SAVE OUTPUT FOR GRAY-LEVEL ANALYSIS
        # ==============================================
        print("\n" + "="*70)
        print("SAVING OUTPUT...")
        print("="*70)
        
        output_file = save_multipixel_output(multipixel_output)
        
        # ==============================================
        # DISPLAY FINAL SUMMARY
        # ==============================================
        print("\n" + "="*70)
        print("🎯 EXECUTION COMPLETE - READY FOR GRAY-LEVEL ANALYSIS")
        print("="*70)
        
        stats = multipixel_output['global_stats']
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   • Total blocks: {stats['total_blocks']:,}")
        print(f"   • Block grid: {stats['blocks_h']}×{stats['blocks_w']}")
        print(f"   • Average gray: {stats['avg_gray_stats']['mean']:.1f}")
        print(f"   • Average contrast: {stats['contrast_stats']['mean']:.3f}")
        print(f"   • High contrast blocks: {stats['contrast_stats']['high_count']:,} "
              f"({stats['contrast_stats']['high_count']/stats['total_blocks']*100:.1f}%)")
        print(f"   • Medium contrast blocks: {stats['contrast_stats']['medium_count']:,}")
        print(f"   • Low contrast blocks: {stats['contrast_stats']['low_count']:,}")
        
        print(f"\n💾 OUTPUT FILE: {output_file}")
        print(f"   • Contains: {stats['total_blocks']:,} blocks")
        print(f"   • Each block has: avg_gray, contrast_weight, local_range, variance")
        
        print(f"\n➡️  NEXT STEP: Gray-Level Analysis will:")
        print(f"   1. Load '{output_file}'")
        print(f"   2. Quantize gray levels (0-3)")
        print(f"   3. Classify edge vs. smooth blocks")
        print(f"   4. Prepare for GM-SIVCS encoding")
        
        # Show sample data
        print("\n" + "="*70)
        print("🔍 SAMPLE BLOCK DATA (First 2 blocks):")
        print("="*70)
        
        for i in range(min(2, len(multipixel_output['blocks_data']))):
            block = multipixel_output['blocks_data'][i]
            print(f"\nBlock {i} at position {block['position']}:")
            print(f"  Pixels: {block['pixel_values']}")
            print(f"  Contrast: {block['contrast_values']}")
            print(f"  Avg Gray: {block['avg_gray']:.2f}")
            print(f"  Contrast Weight: {block['contrast_weight']:.3f}")
            print(f"  Local Range: {block['local_range']:.1f}")
            print(f"  Variance: {block['variance']:.2f}")
        
        print("\n✅ SUCCESS: Multipixel Block processing completed!")
        
    except FileNotFoundError as e:
        print(f"\n❌ FILE ERROR: {e}")
        print("\n⚠️  Please ensure these files exist in the current directory:")
        print("   1. enhanced_image.png (256×256 grayscale image)")
        print("   2. contrast_map.png (256×256 grayscale image)")
        print("\n📁 Current directory files:")
        for file in os.listdir('.'):
            if file.endswith('.png'):
                print(f"   - {file}")
                
    except Exception as e:
        print(f"\n❌ PROCESSING ERROR: {e}")
        traceback.print_exc()