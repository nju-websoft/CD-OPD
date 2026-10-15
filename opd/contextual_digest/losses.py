import torch


def relay_token_losses(student_log_probs, old_log_probs, teacher_log_probs, clip_ratio=0.2):
    if student_log_probs.ndim != 1 or not (student_log_probs.shape == old_log_probs.shape == teacher_log_probs.shape):
        raise ValueError("Expected aligned actual-prefix token log probabilities")
    if not all(torch.isfinite(x).all() for x in (student_log_probs, old_log_probs, teacher_log_probs)):
        raise FloatingPointError("Nonfinite token log probabilities")
    advantages = (teacher_log_probs.detach() - old_log_probs.detach()).detach()
    ratio = torch.exp(torch.clamp(student_log_probs - old_log_probs.detach(), min=-20.0, max=20.0))
    first = -advantages * ratio
    second = -advantages * torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)
    clipped = torch.maximum(first, second)
    return torch.where(advantages < 0, torch.minimum(-advantages * 3.0, clipped), clipped)
