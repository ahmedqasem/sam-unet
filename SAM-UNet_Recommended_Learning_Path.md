# Recommended Learning Path

## Phase 1: Foundation — understand the building blocks

1. `sam_unet/models/resnet.py`
   - Simple, self-contained
   - Standard ResNet usage
   - Shows how multi-scale features are extracted
   - ~45 lines

2. `sam_unet/models/segment_anything/modeling/common.py`
   - Utility classes (`LayerNorm2d`, `MLPBlock`, `Reshaper`)
   - `Reshaper` is used to adapt feature dimensions
   - ~112 lines

3. `sam_unet/config.py`
   - Configuration and constants
   - Shows what the model expects

## Phase 2: Data flow — how the model is used

4. `sam_unet/dataset.py`
   - How data is loaded and preprocessed
   - Input format for the model
   - Shows what `list_input` contains

5. `sam_unet/train_on_single_gpu.py` (or test scripts)
   - How the model is instantiated and called
   - Training loop structure
   - Focus on model usage, not the full training logic

## Phase 3: SAM components — understand the base model

6. `sam_unet/models/segment_anything/build_sam.py`
   - How SAM is constructed
   - Different SAM variants
   - ~149 lines

7. `sam_unet/models/segment_anything/modeling/sam.py` (read the `__init__` and basic structure)
   - SAM’s three main components: image encoder, prompt encoder, mask decoder
   - High-level structure

## Phase 4: Wrappers — how SAM is adapted

8. `sam_unet/models/wrapped.py`
   - How SAM components are wrapped
   - Why step-by-step forward methods are needed
   - Medical-specific modifications
   - ~149 lines

## Phase 5: Main architecture — putting it all together

9. `sam_unet/models/sam_unet_model.py`
   - Full architecture
   - How ResNet and SAM are integrated
   - U-Net-style connections
   - Training vs inference paths
   - ~302 lines

10. `sam_unet/models/build_sam_unet.py`
    - Model factory and checkpoint loading
    - Final piece for understanding model creation

---

## Detailed Starting Point: `resnet.py`

**Why start here**
- Self-contained
- Familiar architecture
- Clear input/output
- Shows the feature extraction pattern used throughout

**What to focus on**
- How it extracts 4 feature maps (`f1, f2, f3, f4`)
- The channel dimensions for each stage
- How it’s used as a feature extractor (not a classifier)

---

## After `resnet.py`, read `sam_unet_model.py` with this focus

Trace this flow:

1. Lines 37–38: ResNet extracts features  
2. Lines 42–45: `adapters_in` reshape ResNet features to match SAM dimensions  
3. Lines 101–114: How ResNet features are injected into SAM encoder  
4. Lines 47–53: `adapters_bridge` creates U-Net skip connections  
5. Lines 117–120: Multi-scale feature fusion  
6. Lines 128–139: How prompts and features flow to the decoder  

---

## Pro Tips for Learning

1. Use print statements or a debugger to inspect tensor shapes at each step
2. Draw a diagram of the data flow as you read
3. Read the docstrings; they explain tensor shapes and purposes
4. Start with inference (`infer` method) before training (`train_forward`)
5. Focus on one path at a time (e.g., encoder path, then decoder path)

---

## Quick Reference: File Complexity

- **Easy:** `resnet.py`, `config.py`, `common.py`
- **Medium:** `wrapped.py`, `build_sam.py`, `dataset.py`
- **Complex:** `sam_unet_model.py` (main architecture)

Start with `resnet.py`, then move to `sam_unet_model.py` and trace how ResNet features flow through the architecture. This bottom-up approach builds understanding step by step.
