import numpy as np
import pickle
from typing import Dict

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bin_range(gray_level: int) -> tuple:
    """Get original range for each quantization level"""
    ranges = {
        0: (0, 63),     # DARKEST
        1: (64, 127),   # DARK
        2: (128, 191),  # LIGHT
        3: (192, 255)   # BRIGHTEST
    }
    return ranges.get(gray_level, (0, 255))


def create_category_mapping() -> Dict:
    """Create mapping of categories to VC encoding actions"""
    return {
        "DARKEST_SMOOTH_HIGH": "ENHANCED_DARK",
        "DARKEST_SMOOTH_MEDIUM": "STANDARD_DARK",
        "DARKEST_SMOOTH_LOW": "BASIC_DARK",

        "DARK_SMOOTH_HIGH": "ENHANCED_DARK",
        "DARK_SMOOTH_MEDIUM": "STANDARD_DARK",
        "DARK_SMOOTH_LOW": "BASIC_DARK",

        "LIGHT_SMOOTH_HIGH": "ENHANCED_LIGHT",
        "LIGHT_SMOOTH_MEDIUM": "STANDARD_LIGHT",
        "LIGHT_SMOOTH_LOW": "BASIC_LIGHT",

        "BRIGHTEST_SMOOTH_HIGH": "ENHANCED_BRIGHT",
        "BRIGHTEST_SMOOTH_MEDIUM": "STANDARD_BRIGHT",
        "BRIGHTEST_SMOOTH_LOW": "BASIC_BRIGHT",
    }


# ============================================================================
# GRAY-LEVEL ANALYSIS BLOCK (EDGE REMOVED)
# ============================================================================

def gray_level_analysis(multipixel_output: Dict) -> Dict:
    print("\n" + "=" * 70)
    print("GRAY-LEVEL ANALYSIS BLOCK (EDGE REMOVED)")
    print("=" * 70)

    blocks_data = multipixel_output['blocks_data']
    total_blocks = len(blocks_data)

    analyzed_blocks = []
    level_counts = [0, 0, 0, 0]
    priority_counts = [0, 0, 0]

    for block in blocks_data:
        avg_gray = block['avg_gray']
        variance = block['variance']
        contrast_weight = block['contrast_weight']

        # -----------------------------
        # GRAY-LEVEL QUANTIZATION
        # -----------------------------
        if avg_gray < 64:
            gray_level, level_name, bin_center = 0, "DARKEST", 32
        elif avg_gray < 128:
            gray_level, level_name, bin_center = 1, "DARK", 96
        elif avg_gray < 192:
            gray_level, level_name, bin_center = 2, "LIGHT", 160
        else:
            gray_level, level_name, bin_center = 3, "BRIGHTEST", 224

        level_counts[gray_level] += 1

        # -----------------------------
        # TEXTURE ANALYSIS
        # -----------------------------
        if variance < 50:
            texture_type = "SMOOTH"
        elif variance < 200:
            texture_type = "TEXTURED"
        else:
            texture_type = "COMPLEX"

        # -----------------------------
        # PRIORITY ASSIGNMENT
        # -----------------------------
        if contrast_weight > 0.7:
            priority, priority_name = 3, "HIGH"
        elif contrast_weight > 0.4:
            priority, priority_name = 2, "MEDIUM"
        else:
            priority, priority_name = 1, "LOW"

        priority_counts[priority - 1] += 1

        # -----------------------------
        # CATEGORY FORMATION
        # -----------------------------
        category = f"{level_name}_SMOOTH_{priority_name}"

        analyzed_blocks.append({
            'position': block['position'],
            'original_avg_gray': avg_gray,
            'gray_level': gray_level,
            'level_name': level_name,
            'texture_type': texture_type,
            'priority': priority,
            'priority_name': priority_name,
            'category': category,
            'quantization_error': avg_gray - bin_center,
            'bin_center': bin_center,
            'bin_range': get_bin_range(gray_level)
        })

    output = {
        'analyzed_blocks': analyzed_blocks,
        'category_mapping': create_category_mapping(),
        'global_stats': {
            'total_blocks': total_blocks,
            'level_distribution': level_counts,
            'priority_distribution': priority_counts
        }
    }

    return output


# ============================================================================
# SAVE / LOAD
# ============================================================================

def save_gray_level_output(output: Dict, filename="gray_level_output.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(output, f)
    print(f"💾 Saved: {filename}")


def load_multipixel_output(filename="multipixel_output.pkl") -> Dict:
    with open(filename, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    multipixel_output = load_multipixel_output()
    gray_level_output = gray_level_analysis(multipixel_output)
    save_gray_level_output(gray_level_output)
