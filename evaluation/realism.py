from __future__ import annotations


def build_protocol_checklist(
    *,
    target_checkpoint: str,
    forget_perc: float,
    shadow_models: int,
    shared_forget_index: bool,
    same_target_checkpoint_for_all_attacks: bool = True,
    same_shadow_setup_for_all_attacks: bool = True,
    dataset_split_name: str | None = None,
    same_dataset_split_for_all_attacks: bool = True,
    uses_original_unlearning_pipeline: bool = True,
) -> dict[str, object]:
    return {
        "target_checkpoint": target_checkpoint,
        "forget_ratio": forget_perc,
        "shadow_model_count": shadow_models,
        "shared_forget_index": shared_forget_index,
        "same_target_checkpoint_for_all_attacks": same_target_checkpoint_for_all_attacks,
        "same_shadow_setup_for_all_attacks": same_shadow_setup_for_all_attacks,
        "dataset_split_name": dataset_split_name,
        "same_dataset_split_for_all_attacks": same_dataset_split_for_all_attacks,
        "uses_original_unlearning_pipeline": uses_original_unlearning_pipeline,
    }
