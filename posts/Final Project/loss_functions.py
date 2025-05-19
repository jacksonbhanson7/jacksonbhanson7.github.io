import torch
import torch.nn.functional as F

def custom_loss_with_diff(preds, targets, alpha=0.4, beta=0.2):
    home_pred, away_pred = preds[:, 0], preds[:, 1]
    home_true, away_true = targets[:, 0], targets[:, 1]

    mse_home = F.mse_loss(home_pred, home_true)
    mse_away = F.mse_loss(away_pred, away_true)

    pred_diff = home_pred - away_pred
    true_diff = home_true - away_true
    diff_loss = F.mse_loss(pred_diff, true_diff)

    return alpha * mse_home + (1 - alpha) * mse_away + beta * diff_loss


def custom_loss(preds, targets, alpha=0.5):
    home_pred, away_pred = preds[:, 0], preds[:, 1]
    home_true, away_true = targets[:, 0], targets[:, 1]

    mse_home = F.mse_loss(home_pred, home_true)
    mse_away = F.mse_loss(away_pred, away_true)

    return alpha * mse_home + (1 - alpha) * mse_away


def margin_weighted_loss(y_pred, y_true, alpha=0.7):
    mse_home = F.mse_loss(y_pred[:, 0], y_true[:, 0])
    mse_away = F.mse_loss(y_pred[:, 1], y_true[:, 1])
    margin_pred = y_pred[:, 0] - y_pred[:, 1]
    margin_true = y_true[:, 0] - y_true[:, 1]
    mse_margin = F.mse_loss(margin_pred, margin_true)
    return alpha * (mse_home + mse_away) + (1 - alpha) * mse_margin


def uncertainty_margin_loss(y_pred, y_true, margin_tolerance=7):
    # Base MSE
    mse = F.mse_loss(y_pred, y_true)

    # Penalize if margin prediction is wrong *and* off by more than the spread
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]
    margin_error = torch.abs(pred_margin - true_margin)

    # Add extra penalty if margin is outside a touchdown (7 points)
    big_miss = (margin_error > margin_tolerance).float()
    penalty = (big_miss * margin_error).mean()

    return mse + 0.3 * penalty  # adjust 0.3 weight as needed


def winner_sensitive_loss(y_pred, y_true, alpha=0.7):
    base_mse = F.mse_loss(y_pred, y_true)

    pred_diff = y_pred[:, 0] - y_pred[:, 1]
    true_diff = y_true[:, 0] - y_true[:, 1]
    wrong_direction = (torch.sign(pred_diff) != torch.sign(true_diff)).float()
    direction_penalty = wrong_direction.mean()

    return alpha * base_mse + (1 - alpha) * direction_penalty


def game_flow_loss(y_pred, y_true):
    base_mse = F.mse_loss(y_pred, y_true)
    pred_margin = torch.abs(y_pred[:, 0] - y_pred[:, 1])
    true_margin = torch.abs(y_true[:, 0] - y_true[:, 1])
    
    margin_bias = F.l1_loss(pred_margin, true_margin)
    return base_mse + 0.2 * margin_bias


def possession_margin_loss(y_pred, y_true):
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]
    margin_error = torch.abs(pred_margin - true_margin)

    # Penalize more if model crosses a possession boundary
    crossing_penalty = (margin_error > 7).float() * (margin_error - 7)
    return F.mse_loss(y_pred, y_true) + crossing_penalty.mean()


def margin_bucket_loss(y_pred, y_true):
    margin_pred = y_pred[:, 0] - y_pred[:, 1]
    margin_true = y_true[:, 0] - y_true[:, 1]

    # Bucket into 0: nail-biter (≤3 pts), 1: close (<14), 2: blowout (≥14)
    def bucket(margin):
        return (torch.abs(margin) > 14).long() * 2 + ((torch.abs(margin) > 3) & (torch.abs(margin) <= 14)).long()

    pred_bucket = bucket(margin_pred)
    true_bucket = bucket(margin_true)

    return F.mse_loss(y_pred, y_true) + F.cross_entropy(pred_bucket.unsqueeze(1).float(), true_bucket)


def direction_confidence_loss(y_pred, y_true):
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]

    margin_error = (pred_margin - true_margin).abs()
    direction_correct = (torch.sign(pred_margin) == torch.sign(true_margin)).float()
    
    # Penalize low confidence when direction is right, and big confidence when wrong
    return F.mse_loss(y_pred, y_true) + (1 - direction_correct) * torch.abs(pred_margin).mean()







def magnitude_normalized_loss(y_pred, y_true, eps=1e-3):
    scale = y_true.abs().mean(dim=1).clamp(min=eps)
    normalized_diff = ((y_pred - y_true) ** 2).mean(dim=1) / scale
    return normalized_diff.mean()


def clipped_margin_loss(y_pred, y_true):
    base_loss = F.mse_loss(y_pred, y_true)

    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]

    margin_error = (pred_margin - true_margin).abs()
    clipped_error = torch.where(margin_error > 21, margin_error * 0.3, margin_error)

    return base_loss + clipped_error.mean()


def dynamic_margin_weighted_loss(y_pred, y_true):
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]
    abs_true_margin = true_margin.abs()

    mse_loss = F.mse_loss(y_pred, y_true)
    direction_error = (torch.sign(pred_margin) != torch.sign(true_margin)).float()

    # Close games → focus on MSE, Blowouts → focus on direction
    weights = torch.sigmoid((abs_true_margin - 14) / 4)  # ~0.5 when margin ≈ 14
    dynamic_loss = (1 - weights) * mse_loss + weights * direction_error.mean()
    return dynamic_loss


def field_goal_scaled_loss(y_pred, y_true):
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]
    margin_error = (pred_margin - true_margin).abs()

    # Penalize by how many field goal "chunks" you're off by
    margin_chunks = (margin_error / 3).round()
    chunk_penalty = (margin_chunks ** 2).mean()

    base_score_loss = F.mse_loss(y_pred, y_true)
    return base_score_loss + 0.1 * chunk_penalty


def possession_error_loss(y_pred, y_true):
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]

    # How many possessions (rounded) the margin error is off by
    possession_error = ((pred_margin - true_margin).abs() / 7).round()
    possession_penalty = (possession_error ** 2).mean()

    score_loss = F.mse_loss(y_pred, y_true)
    return score_loss + 0.2 * possession_penalty


def win_confidence_loss(y_pred, y_true):
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]

    # Win confidence: sigmoid compresses margin into [0,1]
    pred_win_prob = torch.sigmoid(pred_margin / 10)  # scale margin
    true_win_prob = torch.sigmoid(true_margin / 10)

    return F.mse_loss(y_pred, y_true) + 0.3 * F.mse_loss(pred_win_prob, true_win_prob)



def margin_sensitive_loss(y_pred, y_true):
    # y_pred, y_true: shape [batch, 2] -> columns: [home_score, away_score]
    pred_margin = y_pred[:, 0] - y_pred[:, 1]
    true_margin = y_true[:, 0] - y_true[:, 1]

    # Margin error gets special attention
    margin_loss = torch.mean((pred_margin - true_margin) ** 2)

    # Traditional MSE on scores
    mse_home = torch.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)
    mse_away = torch.mean((y_pred[:, 1] - y_true[:, 1]) ** 2)

    # Combine: more weight on getting the *margin* right
    return 0.5 * margin_loss + 0.25 * mse_home + 0.25 * mse_away
