from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def _box_area_xyxy(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def compute_total_loss(loss_dict, weight_dict):
    loss = 0.0
    for k, v in loss_dict.items():
        if k in weight_dict:
            loss = loss + v * weight_dict[k]
    return loss


def generalized_box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = _box_area_xyxy(boxes1)
    area2 = _box_area_xyxy(boxes2)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)
    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]
    return iou - (area_c - union) / (area_c + 1e-6)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, _ = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]
        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]
            if tgt_bbox.numel() == 0:
                indices.append((torch.empty((0,), dtype=torch.int64), torch.empty((0,), dtype=torch.int64)))
                continue
            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()
            src_ind, tgt_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(src_ind, dtype=torch.int64), torch.as_tensor(tgt_ind, dtype=torch.int64)))
        return indices


class DETRSetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits"]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou_matrix = generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
        loss_giou = 1 - torch.diag(giou_matrix)
        loss_giou = loss_giou.sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_logits"].device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses
