import torch
import torch.nn as nn
import pandas as pd
from model.brainmodule import BrainModule

# ==========================================
# Mock Batch Helper
# ==========================================
class MockBatchWithPositions:
    def __init__(self, batch_size, n_subjects):
        self.subject_index = torch.zeros(batch_size, dtype=torch.long)
        self.meg_positions = torch.randn(batch_size, 272, 2) 

def count_module_params(module):
    if module is None: return 0
    return sum(p.numel() for p in module.parameters())

def main():
    print("🚀 모델 파라미터 정밀 분석 중...")
    
    # 1. 모델 설정 (사용자분의 최신 코드 반영)
    in_channels = {'meg': 272}
    hidden_channels = {'meg': 320}
    out_dim = 768
    time_len = 181
    n_subjects = 4

    model = BrainModule(
        in_channels=in_channels,
        out_dim=out_dim,
        time_len=time_len,
        hidden=hidden_channels,
        n_subjects=n_subjects,
        depth=2,
        
        # [논문 스펙]
        merger=True,
        merger_channels=270,
        merger_pos_dim=512,
        rewrite=True,
        glu=1,
        glu_context=1,
        
        subject_layers=True 
    )

    # Merger Patch
    if model.merger:
        model.merger.position_getter.get_positions = lambda batch: torch.randn(1, 272, 2).to(next(model.parameters()).device)
        model.merger.position_getter.is_invalid = lambda pos: torch.zeros(1, 272, dtype=torch.bool).to(pos.device)

    # ==========================================
    # Hook 등록 및 데이터 수집
    # ==========================================
    layer_stats = []

    # 일반 Hook 함수
    def get_info_hook(name, extra_module=None):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0: in_shape = list(input[0].shape)
            else: in_shape = "Unknown"
            if isinstance(output, torch.Tensor): out_shape = list(output.shape)
            else: out_shape = "Unknown"
            
            # 파라미터 계산 시 extra_module(GLU 등)이 있으면 합산
            current_params = count_module_params(module)
            extra_params = count_module_params(extra_module)
            total_params = current_params + extra_params

            layer_stats.append({
                "Layer Name": name, 
                "Input Shape": str(in_shape), 
                "Output Shape": str(out_shape), 
                "Params": f"{total_params:,}"
            })
        return hook

    hooks = []

    # 1~3. 앞부분 레이어들
    if model.merger: hooks.append(model.merger.register_forward_hook(get_info_hook("1. Spatial Attention")))
    if hasattr(model, 'post_merger_linear'): hooks.append(model.post_merger_linear.register_forward_hook(get_info_hook("2. Linear Proj (Early)")))
    if model.subject_layers: hooks.append(model.subject_layers.register_forward_hook(get_info_hook("3. Subject Layer")))

    # 4~5. [핵심 수정] Res Blocks (Sequence + GLU 합산)
    if 'meg' in model.encoders:
        encoder = model.encoders['meg']
        # Block 1
        if len(encoder.sequence) > 0:
            # GLU가 있으면 찾아서 같이 계산하도록 넘김
            glu_module = encoder.glus[0] if len(encoder.glus) > 0 else None
            hooks.append(encoder.sequence[0].register_forward_hook(get_info_hook("4. Res Block 1", glu_module)))
        
        # Block 2
        if len(encoder.sequence) > 1:
            glu_module = encoder.glus[1] if len(encoder.glus) > 1 else None
            hooks.append(encoder.sequence[1].register_forward_hook(get_info_hook("5. Res Block 2", glu_module)))

    # 6~9. 뒷부분 레이어들
    if hasattr(model, 'linear_projection'): hooks.append(model.linear_projection.register_forward_hook(get_info_hook("6. Linear Proj (Deep)")))
    hooks.append(model.temporal_aggregator.register_forward_hook(get_info_hook("7. Temporal Aggregation")))
    hooks.append(model.head.register_forward_hook(get_info_hook("8. MLP Head")))

    # --- 실행 ---
    dummy_input = {'meg': torch.randn(1, 272, time_len)}
    dummy_batch = MockBatchWithPositions(1, n_subjects)

    try:
        _ = model(dummy_input, dummy_batch)
        
        total = sum(p.numel() for p in model.parameters())
        print("=" * 100)
        print(f"📊 Total Parameters: {total:,}")
        print("=" * 100)
        
        df = pd.DataFrame(layer_stats)
        print(f"{'Layer Name':<25} | {'Input Shape':<20} | {'Output Shape':<20} | {'Params':>15}")
        print("-" * 100)
        for _, row in df.iterrows():
            print(f"{row['Layer Name']:<25} | {row['Input Shape']:<20} | {row['Output Shape']:<20} | {row['Params']:>15}")
        print("-" * 100)

    except Exception as e:
        print(f"\n[Error] {e}")
    finally:
        for h in hooks: h.remove()

if __name__ == "__main__":
    main()