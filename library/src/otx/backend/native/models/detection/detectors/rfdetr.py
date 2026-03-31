# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR detector wrapper for OTX integration.

RF-DETR is a state-of-the-art real-time object detector from Roboflow based on
DINOv2 backbone with a lightweight DETR decoder.
Original implementation: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.util.misc import nested_tensor_from_tensor_list
from torch import Tensor, nn
from torchvision.tv_tensors import BoundingBoxes

from otx.backend.native.models.modules.base_module import BaseModule

if TYPE_CHECKING:
    from jsonargparse import Namespace


# COCO's original 91-category IDs that map to the 80 actual classes.
# Used to remap labels from the truncated 91-class head to contiguous 0-79 indices.
_COCO91_TO_CONTIGUOUS = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}


def _build_coco91_remap_tensor(num_classes: int) -> Tensor:
    """Build a lookup tensor that maps COCO-91 label indices to contiguous 0-indexed labels.

    When ``reinitialize_detection_head(80)`` truncates the 91-class head to 80 rows,
    the output label indices correspond to COCO's original category IDs (0-79 from the
    91-class space).  This tensor maps those back to contiguous 0-79 indices matching the
    standard 80-class COCO ordering (person=0, bicycle=1, ..., toothbrush=79).
    """
    remap = torch.full((num_classes,), -1, dtype=torch.long)
    for coco_id, contiguous_id in _COCO91_TO_CONTIGUOUS.items():
        if coco_id < num_classes:
            remap[coco_id] = contiguous_id
    return remap


class RFDETRDetector(BaseModule):
    """Wrapper around RF-DETR's LWDETR model for OTX integration.

    This wrapper handles the interface between OTX's training pipeline and
    the rfdetr package's LWDETR model and SetCriterion.

    Args:
        lwdetr_model: The LWDETR model instance from rfdetr package.
        criterion: The SetCriterion loss function from rfdetr package.
        postprocessor: The PostProcess module from rfdetr package.
        input_size: The input resolution of the model.
        multi_scale: Whether to enable multi-scale training.
    """

    def __init__(
        self,
        lwdetr_model: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        rfdetr_args: Namespace,
        input_size: int = 560,
        multi_scale: bool = False,
    ) -> None:
        super().__init__()
        self.lwdetr = lwdetr_model
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.input_size = input_size
        self.rng = np.random.default_rng(42)

        # Store scales for multi-scale training
        self.scales = (
            compute_multi_scale_scales(
                rfdetr_args.resolution, rfdetr_args.expanded_scales, rfdetr_args.patch_size, rfdetr_args.num_windows
            )
            if multi_scale
            else []
        )

        # Build label remap for COCO-91 → contiguous conversion.
        # Only applies when using pretrained COCO weights with truncated head.
        num_classes = rfdetr_args.num_classes
        remap = _build_coco91_remap_tensor(num_classes)
        self.register_buffer("_coco91_remap", remap)

    def forward(
        self,
        images: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Args:
            images: NestedTensor with images and masks from _customize_inputs.
            targets: List of target dictionaries with 'boxes' and 'labels'.

        Returns:
            During training: Loss dictionary.
            During inference: Predictions dictionary with 'pred_logits' and 'pred_boxes'.
        """
        # Multi-scale training - need to handle NestedTensor
        if self.training and self.scales:
            sz = int(self.rng.choice(self.scales))
            images = nn.functional.interpolate(images, size=[sz, sz], mode="bilinear", align_corners=False)

        # Convert to list of tensors if needed
        if isinstance(images, Tensor) and images.dim() == 4:
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = list(images)

        samples = nested_tensor_from_tensor_list(image_list)

        # Forward through model - images is already a NestedTensor
        outputs = self.lwdetr(samples)

        if self.training:
            self.criterion.train()
            if targets is None:
                msg = "targets should not be None"
                raise ValueError(msg)

            loss_dict = self.criterion(outputs, targets)
            weight_dict: dict[str, float] = self.criterion.weight_dict  # pyrefly: ignore[bad-assignment]
            return {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}

        return outputs

    def postprocess(
        self,
        outputs: dict[str, Tensor],
        original_sizes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[BoundingBoxes], list[Tensor], list[Tensor]]:
        """Post-process model outputs to get final predictions.

        Args:
            outputs: Model outputs with 'pred_logits' and 'pred_boxes'.
            original_sizes: List of original image sizes (H, W).

        Returns:
            Tuple of (scores_list, boxes_list, labels_list).
        """
        target_sizes = torch.tensor(original_sizes, device=outputs["pred_logits"].device)
        results = self.postprocessor(outputs, target_sizes)

        scores_list: list[Tensor] = []
        boxes_list: list[BoundingBoxes] = []
        labels_list: list[Tensor] = []
        masks_list: list[Tensor] = []

        for result, orig_size in zip(results, original_sizes):
            scores_list.append(result["scores"])
            boxes_list.append(
                BoundingBoxes(  # type: ignore[call-overload]
                    result["boxes"],
                    format="xyxy",
                    canvas_size=orig_size,
                ),
            )
            raw_labels = result["labels"].long()
            remap = self._coco91_remap.to(raw_labels.device)
            remapped = remap[raw_labels]
            # Keep only detections with valid COCO class mappings (drop background/gap labels)
            valid = remapped >= 0
            if not valid.all():
                result["scores"] = result["scores"][valid]
                result["boxes"] = result["boxes"][valid]
                raw_labels = raw_labels[valid]
                remapped = remapped[valid]
                # Update scores and boxes lists
                scores_list[-1] = result["scores"]
                boxes_list[-1] = BoundingBoxes(
                    result["boxes"],
                    format="xyxy",
                    canvas_size=orig_size,
                )
            labels_list.append(remapped)
            if "masks" in result:
                masks_list.append(torch.tensor(result["masks"].squeeze(1), dtype=torch.uint8))

        return scores_list, boxes_list, labels_list, masks_list

    def export(
        self,
        batch_inputs: Tensor,
        num_select: int = 300,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Export function for model tracing with mask support.

        Args:
            batch_inputs: Input images tensor.
            num_select: Number of top predictions to select.

        Returns:
            Tuple of (boxes, labels, scores, masks) tensors.
        """
        outputs = self.lwdetr(batch_inputs)
        # outputs may be dict or tuple in export mode
        if isinstance(outputs, dict):
            pred_boxes = outputs["pred_boxes"]
            pred_logits = outputs["pred_logits"]
            pred_masks = outputs.get("pred_masks")
        elif len(outputs) == 3:
            pred_boxes, pred_logits, pred_masks = outputs
        else:
            pred_boxes, pred_logits = outputs
            pred_masks = None
        # Process outputs similar to PostProcess
        scores = torch.sigmoid(pred_logits)
        scores, index = torch.topk(scores.flatten(1), num_select, dim=-1)

        num_classes = pred_logits.shape[-1]
        labels = index % num_classes
        box_index = index // num_classes
        boxes = pred_boxes.gather(
            dim=1,
            index=box_index.unsqueeze(-1).repeat(1, 1, pred_boxes.shape[-1]),
        )

        # Handle masks
        if pred_masks is not None:
            # pred_masks shape: [B, num_queries, H, W]
            # We need to gather masks for selected indices
            masks = pred_masks.gather(
                dim=1,
                index=box_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred_masks.shape[-2], pred_masks.shape[-1]),
            )
            # Apply sigmoid to get mask probabilities
            masks = torch.sigmoid(masks)
            return boxes, labels, scores, masks

        return boxes, labels, scores
