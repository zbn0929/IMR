from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from src.config import Config

# ---------------------------------------------------------------------------
# Attention pooling.
# ---------------------------------------------------------------------------


class AttentionPool(nn.Module):
    """
    Attention pooling with learnable weights.

    This lets the model focus on the most relevant tokens in the sequence and
    gives cause-event representations stronger semantic localization than
    fixed mean pooling.

    scores = Linear(hidden) -> softmax with an optional padding mask
    output = Σ scores_t * hidden_t
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_weight = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden: torch.Tensor,  # (B, L, d_model)
        mask: Optional[torch.Tensor],  # (B, L) True=padding; may be None.
    ) -> torch.Tensor:  # (B, d_model)
        scores = self.attn_weight(hidden).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, L)
        # Guard against all-padding rows, which would otherwise produce NaNs.
        weights = torch.nan_to_num(weights, nan=0.0)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)


# ---------------------------------------------------------------------------
# Text encoder.
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """
    Text encoder for emotion features.

    Encodes token sequences with N Transformer layers to extract contextual
    features for implicit emotion recognition.

    Uses PyTorch's built-in TransformerEncoderLayer with norm_first=True
    (Pre-LN) for more stable training and better deep-gradient flow.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
    ) -> torch.Tensor:  # (B, L, d_model)
        return self.transformer(x, src_key_padding_mask=key_padding_mask)


# ---------------------------------------------------------------------------
# Loss.
# ---------------------------------------------------------------------------


class IMRLoss(nn.Module):
    """
    Multi-task training loss for the IECE demo.

    The main reported task is implicit emotion cause extraction, but the model
    still uses emotion recognition as an auxiliary training objective.
    """

    def __init__(
        self,
        label_smoothing: float,
        emotion_weight: float = Config.LOSS_WEIGHT_EMOTION,
        cause_weight: float = Config.LOSS_WEIGHT_CAUSE,
        center_weight: float = Config.LOSS_WEIGHT_CENTER,
        cause_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.emotion_ce = nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=label_smoothing,
        )
        self.cause_loss_type = Config.CAUSE_LOSS_TYPE.lower().strip()
        if self.cause_loss_type not in {"ce", "focal"}:
            raise ValueError(
                f"Invalid CAUSE_LOSS_TYPE='{Config.CAUSE_LOSS_TYPE}'. "
                "Expected 'ce' or 'focal'."
            )
        self.cause_ce = nn.CrossEntropyLoss(
            weight=cause_class_weights,
            label_smoothing=label_smoothing,
        )
        self.register_buffer(
            "cause_class_weights",
            cause_class_weights.detach().clone()
            if cause_class_weights is not None
            else None,
        )
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(Config.CAUSE_FOCAL_GAMMA)
        self.emotion_weight = float(emotion_weight)
        self.cause_weight = float(cause_weight)
        self.center_weight = float(center_weight)

    def forward(
        self,
        emotion_logits: torch.Tensor,
        cause_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        cause_labels: torch.Tensor,
        center_logits: Optional[torch.Tensor] = None,
        center_labels: Optional[torch.Tensor] = None,
    ):
        emotion_loss = self.emotion_ce(emotion_logits, emotion_labels)
        if self.cause_loss_type == "focal":
            cause_loss = self._focal_cause_loss(cause_logits, cause_labels)
        else:
            cause_loss = self.cause_ce(cause_logits, cause_labels)
        loss = self.emotion_weight * emotion_loss + self.cause_weight * cause_loss
        if (
            self.center_weight > 0
            and center_logits is not None
            and center_labels is not None
        ):
            center_loss = F.cross_entropy(
                center_logits,
                center_labels,
                weight=self.cause_class_weights,
                label_smoothing=self.label_smoothing,
            )
            loss = loss + self.center_weight * center_loss
        return loss

    def _focal_cause_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            labels,
            weight=self.cause_class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt).clamp_min(1e-6).pow(self.focal_gamma)
        return (focal * ce).mean()


# ===========================================================================
# IMR (Iterative Mutual Refinement) architecture v3.
# ===========================================================================
"""
IMR architecture for iterative emotion-cause joint reasoning.

Core ideas:
  1. **Task decoupling**: emotion and cause branches encode their own features.
  2. **Iterative interaction**: prediction runs T rounds of state refinement.
  3. **Event-specific embeddings**: the cause branch uses event_emb directly.

Inference flow:

  Inputs:
    text_emb:  (B, L_text, H)   # plain text for the text-state branch
    cause_input: (B, L_cause, H) # event_emb or concat_emb

  Stage 1 - independent feature encoding:
    emotion_feats = TextEncoder(text_emb)         (B, L_text, d)
    cause_feats   = TextEncoder(cause_input)        (B, L_cause, d)

  Stage 2 - initial state pooling:
    emo_state_0   = AttentionPool(emotion_feats)     (B, d)
    cause_state_0 = AttentionPool(cause_feats)       (B, d)

  Stage 3 - T rounds of iterative interaction:
    for t in range(T):
      # 1. Emotion update: gather clues from Cause
      emo_ctx = CrossAttentionInteraction(
                  query=emo_state_t, kv=cause_feats, kv_state=cause_state_t)
      emo_state_{t+1} = StateRefinement(emo_state_t, emo_ctx)

      # 2. Cause update: gather emotion priors from Emotion
      cause_ctx = CrossAttentionInteraction(
                    query=cause_state_t, kv=emotion_feats, kv_state=emo_state_{t+1})
      cause_state_{t+1} = StateRefinement(cause_state_t, cause_ctx)

  Stage 4 - final prediction:
    emotion_logits = EmotionHead(emo_state_T)        (B, 7)
    cause_logits   = CauseHead(cause_state_T)        (B, 2)

Benefits:
  - Two-branch design: text state and event-cause state reinforce each other.
  - Multi-round reasoning: repeatedly weighs emotion and cause relations.
  - Flexible control: interaction depth is controlled by iteration count T.
  - Task alignment: event_emb directly serves cause extraction.
"""


class CrossAttentionInteraction(nn.Module):
    """
    Cross-task interaction module.

    Extracts context related to the current task state from the other task's
    sequence features.

    query: (B, d) current task state vector
    key/value: (B, L, d) other task sequence features plus broadcast state
    output: (B, d) extracted interaction context

    Uses a small number of attention heads to reduce overfitting on small data.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,  # Use fewer heads to focus on global interaction.
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_state: torch.Tensor,  # (B, d) current task state
        kv_sequence: torch.Tensor,  # (B, L, d) other task sequence features
        kv_state: torch.Tensor,  # (B, d) other task state for key/value enhancement
        kv_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
    ) -> torch.Tensor:  # (B, d)
        # Broadcast the other task state into the sequence as a global signal.
        # kv_state: (B, d) -> (B, 1, d) -> broadcast to (B, L, d)
        kv_state_expanded = kv_state.unsqueeze(1).expand_as(kv_sequence)
        enhanced_kv = kv_sequence + kv_state_expanded  # (B, L, d) residual enhancement

        # query_state: (B, d) -> (B, 1, d) for attention
        q = query_state.unsqueeze(1)  # (B, 1, d)

        # The query extracts relevant information from the enhanced sequence.
        attn_out, _ = self.attn(
            query=q,
            key=enhanced_kv,
            value=enhanced_kv,
            key_padding_mask=kv_padding_mask,
        )  # (B, 1, d)
        return self.norm(attn_out.squeeze(1))  # (B, d)


class StateRefinementModule(nn.Module):
    """
    State update module.

    Fuses the previous state with the newly acquired cross-task context and
    returns the refined state.

    state_new = LayerNorm(state_old + FFN([state_old || cross_context]))

    Uses direct FFN fusion with a residual connection for stable gradients.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        state_old: torch.Tensor,  # (B, d)
        cross_context: torch.Tensor,  # (B, d) context extracted from the other task
    ) -> torch.Tensor:  # (B, d)
        # Concatenate the old state and cross-task context.
        fused_input = torch.cat([state_old, cross_context], dim=-1)  # (B, 2d)
        update = self.fusion(fused_input)  # (B, d)
        # Residual plus LayerNorm.
        return self.norm(state_old + update)


class StateFusionGate(nn.Module):
    """Fuse event-only and context-aware cause states with a learned gate."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, event_state: torch.Tensor, context_state: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                event_state,
                context_state,
                torch.abs(event_state - context_state),
                event_state * context_state,
            ],
            dim=-1,
        )
        gate = self.gate(features)
        fused = gate * event_state + (1.0 - gate) * context_state
        update = self.out(torch.cat([fused, features[:, : event_state.size(-1)]], dim=-1))
        return self.norm(fused + update)


class EmotionConditionedState(nn.Module):
    """Use the final emotion state to modulate cause prediction features."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cause_state: torch.Tensor, emotion_state: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(emotion_state).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        conditioned = cause_state * (1.0 + gamma) + beta
        return self.norm(cause_state + conditioned)


class CauseEventPrior(nn.Module):
    """Encode event position/count features as a weak center-event prior."""

    def __init__(self, d_model: int, dropout: float = 0.1, n_features: int = 4):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.center_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        cause_state: torch.Tensor,
        event_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior_state = self.feature_encoder(event_features)
        center_logits = self.center_head(torch.cat([cause_state, prior_state], dim=-1))
        return self.norm(cause_state + prior_state), center_logits


class RoleAwareEventEncoder(nn.Module):
    """Inject seven-tuple role structure into an event state."""

    def __init__(self, d_model: int, n_roles: int = Config.DL_ROLE_COUNT, dropout: float = 0.1):
        super().__init__()
        self.role_embedding = nn.Parameter(torch.empty(n_roles, d_model))
        nn.init.normal_(self.role_embedding, mean=0.0, std=0.02)
        self.role_value_proj = nn.Linear(1, d_model)
        self.role_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        event_state: torch.Tensor,
        role_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if role_features is None:
            return event_state

        role_features = role_features.to(dtype=event_state.dtype, device=event_state.device)
        role_states = (
            event_state.unsqueeze(1)
            + self.role_embedding.unsqueeze(0)
            + self.role_value_proj(role_features.unsqueeze(-1))
        )
        scores = self.role_score(role_states).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        role_context = (role_states * weights.unsqueeze(-1)).sum(dim=1)
        update = self.output(torch.cat([event_state, role_context], dim=-1))
        return self.norm(event_state + update)


class EventSetReasoner(nn.Module):
    """Contextualize candidate events within each document."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        n_layers: int = Config.DL_EVENT_SET_LAYERS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.position_proj = nn.Linear(4, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        event_states: torch.Tensor,
        event_features: Optional[torch.Tensor],
        doc_ids: Optional[List[str]],
        event_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if doc_ids is None or event_indices is None or event_states.size(0) <= 1:
            return event_states

        output = event_states.clone()
        device = event_states.device
        groups: Dict[str, List[int]] = {}
        for idx, doc_id in enumerate(doc_ids):
            groups.setdefault(str(doc_id), []).append(idx)

        if event_features is None:
            pos_state = torch.zeros_like(event_states)
        else:
            pos_state = self.position_proj(
                event_features.to(dtype=event_states.dtype, device=device)
            )

        for indices in groups.values():
            if len(indices) <= 1:
                continue
            sorted_indices = sorted(indices, key=lambda i: int(event_indices[i].item()))
            idx_tensor = torch.tensor(sorted_indices, device=device, dtype=torch.long)
            seq = event_states.index_select(0, idx_tensor) + pos_state.index_select(0, idx_tensor)
            contextual = self.transformer(seq.unsqueeze(0)).squeeze(0)
            output.index_copy_(0, idx_tensor, contextual)

        return self.norm(event_states + output)


class SelectiveGatedMutualRefinement(nn.Module):
    """RSGMR selective gated bidirectional refinement."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.event_relevance = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.event_to_emo_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.emo_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.emo_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.cause_from_text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.cause_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.cause_update = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.emo_norm = nn.LayerNorm(d_model)
        self.cause_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        emo_state: torch.Tensor,
        event_states: torch.Tensor,
        emotion_feats: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor],
        doc_ids: Optional[List[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if doc_ids is None:
            doc_ids = [str(i) for i in range(event_states.size(0))]

        groups: Dict[str, List[int]] = {}
        for idx, doc_id in enumerate(doc_ids):
            groups.setdefault(str(doc_id), []).append(idx)

        updated_emo = emo_state.clone()
        relevance_scale = float(Config.DL_SGMR_RELEVANCE_SCALE)
        for indices in groups.values():
            idx_tensor = torch.tensor(indices, device=event_states.device, dtype=torch.long)
            doc_emo = emo_state[idx_tensor[0]].unsqueeze(0)
            doc_events = event_states.index_select(0, idx_tensor)
            emo_expanded = doc_emo.expand_as(doc_events)
            relevance = self.event_relevance(torch.cat([doc_events, emo_expanded], dim=-1))
            selected_events = doc_events * (relevance * relevance_scale)
            context, _ = self.event_to_emo_attn(
                query=doc_emo.unsqueeze(1),
                key=selected_events.unsqueeze(0),
                value=selected_events.unsqueeze(0),
            )
            context = context.squeeze(1)
            gate_input = torch.cat([doc_emo, context, doc_emo * context], dim=-1)
            gate = self.emo_gate(gate_input)
            delta = self.emo_update(torch.cat([doc_emo, context], dim=-1))
            doc_updated = self.emo_norm(doc_emo + gate * delta)
            updated_emo.index_copy_(0, idx_tensor, doc_updated.expand(len(indices), -1))

        cause_context, _ = self.cause_from_text_attn(
            query=event_states.unsqueeze(1),
            key=emotion_feats,
            value=emotion_feats,
            key_padding_mask=text_padding_mask,
        )
        cause_context = cause_context.squeeze(1)
        cause_input = torch.cat([event_states, cause_context, updated_emo], dim=-1)
        cause_gate = self.cause_gate(cause_input)
        cause_delta = self.cause_update(cause_input)
        updated_events = self.cause_norm(event_states + cause_gate * cause_delta)
        return updated_emo, updated_events


class IMRModel(nn.Module):
    """
    IMR (Iterative Mutual Refinement) model v3.

    Iterative joint reasoning architecture for emotion and cause.

    Args:
        hidden_size         PLM embedding dimension, default 1024.
        d_model             Transformer feature dimension.
        nhead               Number of attention heads.
        dim_feedforward     FFN hidden dimension.
        n_emotion_layers    Number of emotion encoder layers.
        n_cause_layers      Number of cause encoder layers.
        num_emotions        Number of emotion classes.
        n_iterations        Number of iterative interaction rounds T, default 3.
        dropout             Dropout rate.
        use_event_emb       Backward-compatible input selector for non-dual modes.
        cause_input_mode    "event" | "concat" | "dual".
        ablation_mode       Ablation mode: "full" | "wo_imr" | "wo_backward".
    """

    def __init__(
        self,
        hidden_size: int = Config.DL_HIDDEN_SIZE,
        d_model: int = Config.DL_D_MODEL,
        nhead: int = Config.DL_NHEAD,
        dim_feedforward: int = Config.DL_DIM_FF,
        n_emotion_layers: int = Config.DL_N_EMOTION_LAYERS,
        n_cause_layers: int = Config.DL_N_CAUSE_LAYERS,
        num_emotions: int = Config.get_num_emotions(),
        n_iterations: int = Config.DL_IMR_ITERATIONS,
        dropout: float = Config.DL_DROPOUT,
        use_event_emb: bool = Config.DL_USE_EVENT_EMB,
        cause_input_mode: str = Config.DL_CAUSE_INPUT_MODE,
        ablation_mode: str = Config.DL_ABLATION_MODE,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_emotions = num_emotions
        self.n_iterations = n_iterations
        self.use_event_emb = use_event_emb
        valid_cause_modes = {"event", "concat", "dual"}
        cause_input_mode = cause_input_mode.lower().strip()
        if cause_input_mode not in valid_cause_modes:
            raise ValueError(
                f"Invalid cause_input_mode='{cause_input_mode}'. "
                f"Expected one of {sorted(valid_cause_modes)}."
            )
        self.cause_input_mode = cause_input_mode
        valid_modes = {"full", "wo_imr", "wo_backward"}
        if ablation_mode not in valid_modes:
            raise ValueError(
                f"Invalid ablation_mode='{ablation_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )
        self.ablation_mode = ablation_mode

        # Input projections.
        self.emotion_proj = nn.Linear(hidden_size, d_model)
        self.cause_proj = nn.Linear(hidden_size, d_model)
        self.cause_event_proj = nn.Linear(hidden_size, d_model)
        self.cause_context_proj = nn.Linear(hidden_size, d_model)

        # ===== Stage 1: independent feature encoding =====
        self.emotion_encoder = TextEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_emotion_layers,
            dropout=dropout,
        )
        self.cause_encoder = TextEncoder(  # Same structure, independent parameters.
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_cause_layers,
            dropout=dropout,
        )
        self.cause_event_encoder = TextEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_cause_layers,
            dropout=dropout,
        )
        self.cause_context_encoder = TextEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_cause_layers,
            dropout=dropout,
        )

        # ===== Stage 2: initial state pooling =====
        self.emotion_attn_pool = AttentionPool(d_model)
        self.cause_attn_pool = AttentionPool(d_model)
        self.cause_event_attn_pool = AttentionPool(d_model)
        self.cause_context_attn_pool = AttentionPool(d_model)
        self.cause_state_fusion = StateFusionGate(d_model, dropout)
        self.cause_event_prior = CauseEventPrior(d_model, dropout)
        self.role_event_encoder = RoleAwareEventEncoder(d_model, Config.DL_ROLE_COUNT, dropout)
        self.event_set_reasoner = EventSetReasoner(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=Config.DL_EVENT_SET_LAYERS,
            dropout=dropout,
        )

        # ===== Stage 3: iterative interaction modules =====
        # Emotion update: gather clues from Cause.
        self.emo_from_cause_attn = CrossAttentionInteraction(d_model, dropout)
        self.emo_refinement = StateRefinementModule(d_model, dropout)

        # Cause update: gather emotion priors from Emotion.
        self.cause_from_emo_attn = CrossAttentionInteraction(d_model, dropout)
        self.cause_refinement = StateRefinementModule(d_model, dropout)
        self.sgmr_refinement = SelectiveGatedMutualRefinement(d_model, dropout)
        self.cause_output_conditioner = EmotionConditionedState(d_model, dropout)

        # ===== Stage 4: final prediction heads =====
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_emotions),
        )

        self.cause_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(
        self,
        text_emb: torch.Tensor,  # (B, L, H) text embedding
        cause_input_emb: torch.Tensor,  # (B, L, H) event_emb or concat_emb
        text_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
        cause_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
        concat_input_emb: Optional[torch.Tensor] = None,  # (B, L, H) concat_emb
        concat_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
        event_features: Optional[torch.Tensor] = None,  # (B, 4) structural features
        role_features: Optional[torch.Tensor] = None,  # (B, 7) seven-tuple role features
        doc_ids: Optional[List[str]] = None,
        event_indices: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        return_auxiliary: bool = False,
    ):
        """
        Returns:
            emotion_logits: (B, num_emotions)  emotion class scores
            cause_logits:   (B, 2)             cause/non-cause scores
        """
        # ===== Stage 1: independent feature encoding =====
        emotion_seq = self.emotion_proj(text_emb)  # (B, L, d)
        emotion_feats = self.emotion_encoder(
            emotion_seq, text_padding_mask
        )  # (B, L, d)

        cause_feats, cause_padding_mask, cause_state = self._encode_cause_view(
            cause_input_emb=cause_input_emb,
            concat_input_emb=concat_input_emb,
            cause_padding_mask=cause_padding_mask,
            concat_padding_mask=concat_padding_mask,
        )

        # ===== Stage 2: initial states =====
        emo_state = self.emotion_attn_pool(emotion_feats, text_padding_mask)  # (B, d)
        center_logits = None
        if Config.DL_USE_RSGMR and Config.DL_USE_ROLE_AWARE_EVENT:
            cause_state = self.role_event_encoder(cause_state, role_features)

        if Config.DL_USE_RSGMR and Config.DL_USE_EVENT_SET_REASONING:
            cause_state = self.event_set_reasoner(
                event_states=cause_state,
                event_features=event_features,
                doc_ids=doc_ids,
                event_indices=event_indices,
            )

        if Config.DL_USE_CENTER_EVENT_PRIOR and event_features is not None:
            prior_state, center_logits = self.cause_event_prior(
                cause_state,
                event_features.to(dtype=cause_state.dtype),
            )
            scale = float(Config.DL_CENTER_PRIOR_STATE_SCALE)
            cause_state = torch.lerp(cause_state, prior_state, scale)

        intermediates: List[Dict[str, torch.Tensor]] = []

        def _record(iteration_idx: int, emo: torch.Tensor, cau: torch.Tensor) -> None:
            if not return_intermediate:
                return
            intermediates.append(
                {
                    "iteration": torch.tensor(iteration_idx),
                    "emotion_logits": self.emotion_head(emo).detach().cpu(),
                    "cause_logits": self.cause_head(cau).detach().cpu(),
                }
            )

        # Variant 1: w/o IMR - both tasks are independent, with no interaction.
        if self.ablation_mode == "wo_imr":
            emotion_logits = self.emotion_head(emo_state)  # (B, num_emotions)
            cause_logits = self._predict_cause_logits(cause_state, center_logits)
            _record(0, emo_state, cause_state)
            if return_intermediate:
                return emotion_logits, cause_logits, intermediates
            if return_auxiliary:
                return emotion_logits, cause_logits, center_logits
            return emotion_logits, cause_logits

        # Variant 2: w/o Backward - keep only one-way Emotion -> Cause interaction.
        if self.ablation_mode == "wo_backward":
            for i in range(self.n_iterations):
                cause_ctx = self.cause_from_emo_attn(
                    query_state=cause_state,
                    kv_sequence=emotion_feats,
                    kv_state=emo_state,  # Emotion state remains the initial pooled state.
                    kv_padding_mask=text_padding_mask,
                )  # (B, d)
                cause_state = self.cause_refinement(cause_state, cause_ctx)  # (B, d)
                _record(i + 1, emo_state, cause_state)

            emotion_logits = self.emotion_head(emo_state)  # (B, num_emotions)
            cause_pred_state = self.cause_output_conditioner(cause_state, emo_state)
            cause_logits = self._predict_cause_logits(cause_pred_state, center_logits)
            if return_intermediate:
                return emotion_logits, cause_logits, intermediates
            if return_auxiliary:
                return emotion_logits, cause_logits, center_logits
            return emotion_logits, cause_logits

        # ===== Stage 3: T rounds of iterative interaction =====
        for i in range(self.n_iterations):
            if Config.DL_USE_RSGMR and Config.DL_USE_SELECTIVE_GATED_REFINEMENT:
                emo_state, cause_state = self.sgmr_refinement(
                    emo_state=emo_state,
                    event_states=cause_state,
                    emotion_feats=emotion_feats,
                    text_padding_mask=text_padding_mask,
                    doc_ids=doc_ids,
                )
            else:
                # 1. Emotion update: gather clues from Cause.
                emo_ctx = self.emo_from_cause_attn(
                    query_state=emo_state,
                    kv_sequence=cause_feats,
                    kv_state=cause_state,
                    kv_padding_mask=cause_padding_mask,
                )  # (B, d)
                emo_state = self.emo_refinement(emo_state, emo_ctx)  # (B, d)

                # 2. Cause update: gather emotion priors from Emotion.
                cause_ctx = self.cause_from_emo_attn(
                    query_state=cause_state,
                    kv_sequence=emotion_feats,
                    kv_state=emo_state,  # Use the just-updated emotion state.
                    kv_padding_mask=text_padding_mask,
                )  # (B, d)
                cause_state = self.cause_refinement(cause_state, cause_ctx)  # (B, d)
            _record(i + 1, emo_state, cause_state)

        # ===== Stage 4: final prediction =====
        emotion_logits = self.emotion_head(emo_state)  # (B, 7)
        cause_pred_state = self.cause_output_conditioner(cause_state, emo_state)
        cause_logits = self._predict_cause_logits(cause_pred_state, center_logits)

        if return_intermediate:
            return emotion_logits, cause_logits, intermediates
        if return_auxiliary:
            return emotion_logits, cause_logits, center_logits
        return emotion_logits, cause_logits

    def _predict_cause_logits(
        self,
        cause_state: torch.Tensor,
        center_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cause_logits = self.cause_head(cause_state)
        if Config.DL_USE_CENTER_EVENT_PRIOR and center_logits is not None:
            cause_logits = cause_logits + float(Config.DL_CENTER_PRIOR_LOGIT_SCALE) * center_logits
        return cause_logits

    def _encode_cause_view(
        self,
        cause_input_emb: torch.Tensor,
        concat_input_emb: Optional[torch.Tensor],
        cause_padding_mask: Optional[torch.Tensor],
        concat_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.cause_input_mode == "event":
            cause_seq = self.cause_proj(cause_input_emb)
            cause_feats = self.cause_encoder(cause_seq, cause_padding_mask)
            cause_state = self.cause_attn_pool(cause_feats, cause_padding_mask)
            return cause_feats, cause_padding_mask, cause_state

        if self.cause_input_mode == "concat":
            context_emb = concat_input_emb if concat_input_emb is not None else cause_input_emb
            context_mask = concat_padding_mask
            if context_mask is None:
                context_mask = cause_padding_mask
            cause_seq = self.cause_proj(context_emb)
            cause_feats = self.cause_encoder(cause_seq, context_mask)
            cause_state = self.cause_attn_pool(cause_feats, context_mask)
            return cause_feats, context_mask, cause_state

        if concat_input_emb is None:
            concat_input_emb = cause_input_emb
        if concat_padding_mask is None:
            concat_padding_mask = cause_padding_mask

        event_seq = self.cause_event_proj(cause_input_emb)
        event_feats = self.cause_event_encoder(event_seq, cause_padding_mask)
        event_state = self.cause_event_attn_pool(event_feats, cause_padding_mask)

        context_seq = self.cause_context_proj(concat_input_emb)
        context_feats = self.cause_context_encoder(context_seq, concat_padding_mask)
        context_state = self.cause_context_attn_pool(context_feats, concat_padding_mask)

        cause_feats = torch.cat([event_feats, context_feats], dim=1)
        if cause_padding_mask is None and concat_padding_mask is None:
            fused_mask = None
        else:
            if cause_padding_mask is None:
                cause_padding_mask = torch.zeros_like(concat_padding_mask)
            if concat_padding_mask is None:
                concat_padding_mask = torch.zeros_like(cause_padding_mask)
            fused_mask = torch.cat([cause_padding_mask, concat_padding_mask], dim=1)
        cause_state = self.cause_state_fusion(event_state, context_state)
        return cause_feats, fused_mask, cause_state
