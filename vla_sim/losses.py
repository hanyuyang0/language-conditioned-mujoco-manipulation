import numpy as np
import torch
from torch.nn import functional as F


def gripper_positive_weight(dataset):
    """Balance open/closed labels using only the supplied training episodes."""
    closed = sum(int(np.count_nonzero(ep['actions'][:, 3] > 0)) for ep in dataset.episodes)
    total = sum(len(ep['actions']) for ep in dataset.episodes)
    if closed == 0 or closed == total:
        raise ValueError('Gripper classification requires both open and closed training actions')
    return (total - closed) / closed, {'closed': closed, 'open': total - closed}


def policy_loss(model, image, state, tokens, target, mask, positive_weight=None,
                gripper_loss_weight=1.0, return_metrics=False):
    valid = ~mask
    metrics = {'valid_steps': int(valid.sum())}
    if positive_weight is not None:
        xyz, logits = model.forward_logits(image, state, tokens)
        xyz_loss = F.smooth_l1_loss(xyz, target[..., :3], reduction='none').mean(-1)
        grip_loss = F.binary_cross_entropy_with_logits(
            logits, (target[..., 3] > 0).float(), pos_weight=positive_weight, reduction='none')
        loss = (xyz_loss + gripper_loss_weight * grip_loss)[valid].mean()
        truth = target[..., 3][valid] > 0
        predicted = logits[valid] > 0
        metrics.update(xyz_loss=float(xyz_loss[valid].mean().detach()),
                       gripper_loss=float(grip_loss[valid].mean().detach()),
                       tp=int((truth & predicted).sum()), fn=int((truth & ~predicted).sum()),
                       fp=int((~truth & predicted).sum()), tn=int((~truth & ~predicted).sum()))
    else:
        prediction = model(image, state, tokens)
        loss = F.smooth_l1_loss(prediction, target, reduction='none').mean(-1)[valid].mean()
    metrics['loss'] = float(loss.detach())
    return (loss, metrics) if return_metrics else loss


def aggregate_metrics(batches):
    count = sum(b['valid_steps'] for b in batches)
    result = {key: sum(b[key] * b['valid_steps'] for b in batches) / count
              for key in ('loss', 'xyz_loss', 'gripper_loss') if key in batches[0]}
    if 'tp' in batches[0]:
        tp, fn, fp, tn = [sum(b[key] for b in batches) for key in ('tp', 'fn', 'fp', 'tn')]
        result.update(closed_recall=tp/(tp+fn) if tp+fn else None,
                      open_false_close_rate=fp/(fp+tn) if fp+tn else None,
                      gripper_accuracy=(tp+tn)/(tp+fn+fp+tn),
                      true_closed_pred_closed=tp, true_closed_pred_open=fn,
                      true_open_pred_closed=fp, true_open_pred_open=tn)
    return result
