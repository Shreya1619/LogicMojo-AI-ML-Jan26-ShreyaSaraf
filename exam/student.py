from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch
from torch import nn


def build_customer_product_features(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()

    df["is_purchase"] = df["event_type"] == "purchase"
    df["is_view"] = df["event_type"] == "view"
    df["is_cart"] = df["event_type"] == "cart"
    df["is_non_returned_purchase"] = df["is_purchase"] & ~df["is_returned"]
    # Revenue only counts for non-returned purchases
    df["revenue"] = df["price"] * df["quantity"] * df["is_non_returned_purchase"]

    agg = df.groupby("customer_id").agg(
        view_count=("is_view", "sum"),
        cart_count=("is_cart", "sum"),
        purchase_count=("is_purchase", "sum"),
        non_returned_purchase_count=("is_non_returned_purchase", "sum"),
        gross_revenue=("revenue", "sum"),
        unique_products=("product_id", "nunique"),
    ).reset_index()

    agg["return_rate"] = (
        (agg["purchase_count"] - agg["non_returned_purchase_count"])
        / agg["purchase_count"]
    )

    agg["avg_order_value"] = (
        agg["gross_revenue"] / agg["non_returned_purchase_count"]
    )

    agg["cart_to_purchase_rate"] = (
        agg["purchase_count"] / agg["cart_count"]
    )

    agg["view_to_purchase_rate"] = (
        agg["purchase_count"] / agg["view_count"]
    )

    return agg


def top_products_by_revenue(events: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return top products by non-returned purchase revenue.

    Output columns:
    product_id, non_returned_units, gross_revenue, unique_buyers, return_rate
    """
    df = events.copy()

    df["is_purchase"] = df["event_type"] == "purchase"
    df["is_non_returned_purchase"] = df["is_purchase"] & ~df["is_returned"]
    df["revenue"] = df["price"] * df["quantity"] * df["is_non_returned_purchase"]

    purchases = df[df["is_purchase"]].copy()

    agg = purchases.groupby("product_id").agg(
        non_returned_units=("quantity", lambda x: x[purchases.loc[x.index, "is_non_returned_purchase"]].sum()),
        gross_revenue=("revenue", "sum"),
        unique_buyers=("customer_id", "nunique"),
        total_purchases=("is_purchase", "sum"),
        returned_purchases=("is_returned", "sum"),
    ).reset_index()

    agg["return_rate"] = (
        agg["returned_purchases"] / agg["total_purchases"]
    ).where(agg["total_purchases"] > 0, 0.0)

    result = (
        agg.sort_values("gross_revenue", ascending=False)
        .head(top_n)
        .drop(columns=["total_purchases", "returned_purchases"])
        .reset_index(drop=True)
    )

    return result


def make_balanced_sampler_weights(labels: torch.Tensor) -> torch.Tensor:
    """Return one sampling weight per label using inverse class frequency.

    Use this formula for class c:
        total_samples / (num_classes * count_of_class_c)
    """
    raise NotImplementedError


def conv2d_output_shape(
    input_hw: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """Return output height and width for a 2D convolution."""
    raise NotImplementedError


class TinyCnn(nn.Module):
    """Small CNN for 28x28 grayscale image classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch model."""
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    """Train for one epoch and return average loss and accuracy."""
    raise NotImplementedError


def confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return a num_classes x num_classes confusion matrix.

    Rows are true labels. Columns are predicted labels.
    """
    indices = y_true * num_classes + y_pred
    
    # Count occurrences and reshape back into the matrix
    return torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    """Compute macro F1 from a confusion matrix.

    Rows are true labels. Columns are predicted labels.
    """
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    # 1e-8 epsilon avoids division by zero without clumsy if/else blocks
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1.mean().item()

