import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou(boxes1_xyxy, boxes2_xyxy):
    num_boxes1 = boxes1_xyxy.size(0)
    num_boxes2 = boxes2_xyxy.size(0)

    top_left = torch.max(
        boxes1_xyxy[:, :2].unsqueeze(1).expand(num_boxes1, num_boxes2, 2),
        boxes2_xyxy[:, :2].unsqueeze(0).expand(num_boxes1, num_boxes2, 2),
    )
    bottom_right = torch.min(
        boxes1_xyxy[:, 2:].unsqueeze(1).expand(num_boxes1, num_boxes2, 2),
        boxes2_xyxy[:, 2:].unsqueeze(0).expand(num_boxes1, num_boxes2, 2),
    )

    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (
        boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]
    )
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (
        boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]
    )
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection_area

    return intersection_area / union_area.clamp(min=1e-6)


class YOLOLoss(nn.Module):
    def __init__(self, grid_size, boxes_per_cell, lambda_coord, lambda_noobj):
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def split_prediction_tensor(self, prediction_tensor):
        N, S, _, total_channels = prediction_tensor.shape
        B = self.boxes_per_cell
        box_channels = B * 5
        
        boxes_flat = prediction_tensor[..., :box_channels]
        class_scores = prediction_tensor[..., box_channels:]
        
        predicted_boxes = boxes_flat.view(N, S, S, B, 5)
        
        return predicted_boxes, class_scores

    def xywh_to_xyxy(self, boxes_xywh, col_row):
        S = self.grid_size
        
        x_offset = boxes_xywh[:, 0]
        y_offset = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        
        col = col_row[:, 0].float()
        row = col_row[:, 1].float()
        
        center_x = (col + x_offset) / S
        center_y = (row + y_offset) / S
        
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def choose_responsible_box(self, predicted_boxes, target_boxes, has_object_mask):
        N, S, _, B, _ = predicted_boxes.shape
        device = predicted_boxes.device
        
        if not has_object_mask.any():
            return (
                torch.zeros((0, 5), device=device),
                torch.zeros((0, 1), device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )
        
        obj_indices = has_object_mask.nonzero(as_tuple=False)
        num_obj_cells = obj_indices.shape[0]
        
        n_idx = obj_indices[:, 0]
        row_idx = obj_indices[:, 1]
        col_idx = obj_indices[:, 2]
        
        pred_boxes_obj = predicted_boxes[n_idx, row_idx, col_idx]
        target_boxes_obj = target_boxes[n_idx, row_idx, col_idx]
        
        col_row = torch.stack([col_idx, row_idx], dim=-1)
        target_xyxy = self.xywh_to_xyxy(target_boxes_obj, col_row)
        
        all_ious = []
        for b in range(B):
            pred_b = pred_boxes_obj[:, b, :4]
            pred_xyxy = self.xywh_to_xyxy(pred_b, col_row)
            iou_matrix = compute_iou(pred_xyxy, target_xyxy)
            iou_diag = torch.diag(iou_matrix)
            all_ious.append(iou_diag)
        
        all_ious = torch.stack(all_ious, dim=-1)
        best_ious, best_predictor_idx = all_ious.max(dim=-1)
        
        batch_indices = torch.arange(num_obj_cells, device=device)
        responsible_box_predictions = pred_boxes_obj[batch_indices, best_predictor_idx]
        
        return (
            responsible_box_predictions,
            best_ious.unsqueeze(-1),
            best_predictor_idx,
        )

    def build_responsible_mask(self, predicted_boxes, has_object_mask, responsible_predictor_index):
        N, S, _, B, _ = predicted_boxes.shape
        device = predicted_boxes.device
        
        responsible_mask = torch.zeros((N, S, S, B), dtype=torch.bool, device=device)
        
        if not has_object_mask.any():
            return responsible_mask
        
        obj_indices = has_object_mask.nonzero(as_tuple=False)
        n_idx = obj_indices[:, 0]
        row_idx = obj_indices[:, 1]
        col_idx = obj_indices[:, 2]
        
        responsible_mask[n_idx, row_idx, col_idx, responsible_predictor_index] = True
        
        return responsible_mask

    def regression_xy_loss(self, responsible_box_predictions, target_boxes_for_object_cells):
        if responsible_box_predictions.shape[0] == 0:
            return responsible_box_predictions.new_tensor(0.0)
        
        pred_xy = responsible_box_predictions[:, :2]
        target_xy = target_boxes_for_object_cells[:, :2]
        
        loss = ((pred_xy - target_xy) ** 2).sum()
        return loss

    def regression_wh_loss(self, responsible_box_predictions, target_boxes_for_object_cells):
        if responsible_box_predictions.shape[0] == 0:
            return responsible_box_predictions.new_tensor(0.0)
        
        pred_wh = responsible_box_predictions[:, 2:4]
        target_wh = target_boxes_for_object_cells[:, 2:4]
        
        pred_sqrt = torch.sqrt(pred_wh.clamp(min=1e-6))
        target_sqrt = torch.sqrt(target_wh.clamp(min=1e-6))
        
        loss = ((pred_sqrt - target_sqrt) ** 2).sum()
        return loss

    def object_confidence_loss(self, responsible_box_predictions, responsible_box_ious):
        if responsible_box_predictions.shape[0] == 0:
            return responsible_box_predictions.new_tensor(0.0)
        
        pred_conf = responsible_box_predictions[:, 4]
        target_conf = responsible_box_ious.squeeze(-1).detach()
        
        loss = ((pred_conf - target_conf) ** 2).sum()
        return loss

    def no_object_confidence_loss(self, predicted_boxes, responsible_mask):
        no_obj_mask = ~responsible_mask
        pred_conf = predicted_boxes[..., 4]
        loss = (pred_conf[no_obj_mask] ** 2).sum()
        return loss

    def class_probability_loss(self, predicted_class_scores, target_class_scores, has_object_mask):
        if not has_object_mask.any():
            return predicted_class_scores.new_tensor(0.0)
        
        pred_cls = predicted_class_scores[has_object_mask]
        target_cls = target_class_scores[has_object_mask]
        
        loss = ((pred_cls - target_cls) ** 2).sum()
        return loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        batch_size = pred_tensor.shape[0]
        
        predicted_boxes, predicted_class_scores = self.split_prediction_tensor(pred_tensor)
        
        responsible_box_predictions, responsible_box_ious, responsible_predictor_index = \
            self.choose_responsible_box(predicted_boxes, target_boxes, has_object_map)
        
        responsible_mask = self.build_responsible_mask(
            predicted_boxes, has_object_map, responsible_predictor_index
        )
        
        if has_object_map.any():
            obj_indices = has_object_map.nonzero(as_tuple=False)
            target_boxes_for_object_cells = target_boxes[
                obj_indices[:, 0], obj_indices[:, 1], obj_indices[:, 2]
            ]
        else:
            target_boxes_for_object_cells = target_boxes.new_zeros((0, 4))
        
        reg_xy_loss = self.regression_xy_loss(responsible_box_predictions, target_boxes_for_object_cells)
        reg_wh_loss = self.regression_wh_loss(responsible_box_predictions, target_boxes_for_object_cells)
        reg_loss = reg_xy_loss + reg_wh_loss
        
        obj_loss = self.object_confidence_loss(responsible_box_predictions, responsible_box_ious)
        no_obj_loss = self.no_object_confidence_loss(predicted_boxes, responsible_mask)
        
        cls_loss = self.class_probability_loss(predicted_class_scores, target_cls, has_object_map)
        
        total_loss = (
            self.lambda_coord * reg_loss +
            obj_loss +
            self.lambda_noobj * no_obj_loss +
            cls_loss
        ) / batch_size
        
        return {
            "total_loss": total_loss,
            "reg_loss": (self.lambda_coord * reg_loss) / batch_size,
            "reg_xy_loss": (self.lambda_coord * reg_xy_loss) / batch_size,
            "reg_wh_loss": (self.lambda_coord * reg_wh_loss) / batch_size,
            "obj_loss": obj_loss / batch_size,
            "no_obj_loss": (self.lambda_noobj * no_obj_loss) / batch_size,
            "cls_loss": cls_loss / batch_size,
        }