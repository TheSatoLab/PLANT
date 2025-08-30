

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, EsmConfig, PreTrainedModel
from transformers.utils import ModelOutput
from safetensors.torch import load_file as safe_load

# ============== ここだけ最小追加：OHE を外部からセットするための仕組み ==============
OHE_virus = None
OHE_ref = None
OHE_vp = None
OHE_rp = None

def set_encoders(ohe_virus, ohe_ref, ohe_vp, ohe_rp):
    """
    学習済みの OneHotEncoder 群を外部から注入します。
    semanticESM の forward 内では元コード通り OHE_virus / OHE_ref など
    グローバル変数を参照します。
    """
    global OHE_virus, OHE_ref, OHE_vp, OHE_rp
    OHE_virus, OHE_ref, OHE_vp, OHE_rp = ohe_virus, ohe_ref, ohe_vp, ohe_rp
# ==========================================================================


class semanticESM(PreTrainedModel):
    config_class = EsmConfig

    def __init__(
        self,
        config,
        esm_model_name,
        effects_len = None,
        virus_effects_len = None,
        embed_scale_factor = 1,
        latent_dim=3,
        intermediate_dim = 256,
        intermediate_dim_encoder = 64,
        dropout = 0.05,
        dropout_encoder=0.1,
        MAIN_W=1,
        CSE_W=0,
        CSE_ALPHA=0,
        SEMANTIC_W=0.2,
        CSE_W_VIRUS_ONLY=0,
        SEMANTIC_W_VIRUS_ONLY=0.2,
        CART_W=0.05,
        LG_W=0.01
    ):
        super().__init__(config)

        # OHE が未セットだと元コードの forward で参照できないため、ここで検査
        if any(x is None for x in (OHE_virus, OHE_ref, OHE_vp, OHE_rp)):
            raise RuntimeError("Encoders are not set. Call plant.model.set_encoders(...) before model init.")

        # 既存コードでは num_* は外で計算していましたが、モジュール化にあたり自動推定も許容
        if virus_effects_len is None:
            virus_effects_len = len(OHE_virus.categories_[0])
        if effects_len is None:
            effects_len = len(OHE_ref.categories_[0]) + len(OHE_vp.categories_[0]) + len(OHE_rp.categories_[0])

        # ESM モデルの初期化
        self.esm_model = AutoModel.from_pretrained(esm_model_name, add_pooling_layer=False)
        self.esm_model_original = self._initialize_frozen_esm_model(esm_model_name)

        self.embedding_dim = self.esm_model.config.hidden_size

        # 回帰モデル
        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, latent_dim),
        )

        # systematic error
        self.virus_effects = nn.Linear(virus_effects_len, 1, bias=False)
        self.systematic_error_effects = nn.Sequential(
            nn.Linear(effects_len, intermediate_dim_encoder),
            nn.ReLU(),
            nn.Dropout(dropout_encoder),
            nn.Linear(intermediate_dim_encoder, 1, bias=False),
        )

        # スケールパラメータ
        self.embed_scale = nn.Parameter(torch.tensor(1.0))

        # **MSE Loss をクラス内で保持**
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_mean = nn.MSELoss(reduction="none")


        # **ハイパーパラメータを保存**
        self.MAIN_W = MAIN_W
        self.CSE_W = CSE_W
        self.SEMANTIC_W = SEMANTIC_W
        self.CSE_ALPHA=CSE_ALPHA
        self.CART_W = CART_W
        self.CSE_W_VIRUS_ONLY = CSE_W_VIRUS_ONLY
        self.SEMANTIC_W_VIRUS_ONLY = SEMANTIC_W_VIRUS_ONLY
        self.LG_W = LG_W

        self.embed_scale_factor = embed_scale_factor

    def _initialize_frozen_esm_model(self, esm_model_name):
        """事前学習済みの ESM モデルを読み込み、フリーズする"""
        esm_model = AutoModel.from_pretrained(esm_model_name, add_pooling_layer=False)
        for param in esm_model.parameters():
            param.requires_grad = False
        return esm_model

    def encode_sequence(self, model, input_ids, attention_mask):
        """ESM モデルでエンコードし、特徴量を取得"""
        encoder_out = model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        return encoder_out.last_hidden_state[:, 0, :]

    def encode_one_hot(self, encoder, input_tensor):
        """One-hot encoding を適用"""
        model_dtype = next(self.parameters()).dtype
        return torch.tensor(
            encoder.transform(input_tensor.cpu().numpy().reshape(-1, 1)).toarray(),
            dtype=model_dtype,
        ).to(self.device)

    def extract_pooled_embeddings(self, model, input_ids, attention_mask):
        """オリジナル ESM での特徴抽出 (Mean Pooling + Max Pooling)"""
        with torch.no_grad():
            encoder_out = model(input_ids, attention_mask).last_hidden_state

        masked_sum = (encoder_out * attention_mask.unsqueeze(-1)).sum(dim=1)
        mask_count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled = masked_sum / mask_count

        max_pooled = torch.max(encoder_out, dim=1)[0]
        pooled_embedding = torch.cat([mean_pooled, max_pooled], dim=-1)

        return pooled_embedding


    def compute_semantic_loss(self, latents, latents_original, embed_scale, embed_scale_factor):

        # 埋め込み間のペアワイズ距離を計算
        pairwise_distances = torch.cdist(latents, latents, p=2)
        upper_triangle_indices = torch.triu_indices(
            pairwise_distances.size(0), pairwise_distances.size(1), offset=1
        )

        pairwise_distances_original = torch.cdist(
            latents_original,
            latents_original,
            p=2
        )
        pairwise_distances_original = pairwise_distances_original / embed_scale_factor

        # MSE Loss の計算（重みを適用）
        semantic_loss = self.mse_loss_wo_mean(
            pairwise_distances_original[upper_triangle_indices],
            pairwise_distances[upper_triangle_indices] * embed_scale
        ).mean()

        return semantic_loss


    def contrastive_loss_semantic(self, embeddings1, embeddings2, margin=1.0, alpha=1):
        """
        Contrastive Loss Semantic1 を改良し、近い負例により大きなペナルティを与える
        - `alpha`: 近い負例に対するペナルティの強度（大きいほどHard Negativeに強くなる）
        """
        batch_size = embeddings1.shape[0]

        # L2距離の計算
        distance_matrix = torch.cdist(embeddings1, embeddings2, p=2)  # shape: (batch_size, batch_size)

        # 正例ペア（対角成分）は距離を最小化
        positive_loss = torch.mean(torch.diag(distance_matrix))  # D[i, i] の距離を最小化

        # 近い負例ほど強いペナルティを与えるように設計
        # (1) 通常の margin-based clamping を適用
        margin_loss = torch.clamp(margin - distance_matrix, min=0)  # D[i, j] for i≠j を margin 以上に

        # (2) 近い負例ほど影響が強くなるように、exp(-α * distance) を適用
        weight = torch.exp(-alpha * distance_matrix)  # 近いほど重みが大きい
        negative_loss = torch.mean(weight * margin_loss)  # 重み付きペナルティを計算

        return positive_loss + negative_loss

    def local_global_loss(self, latents, k_local=3, margin_global=0.125):
        """
        - 近傍 k_local 個を attract（局所密度を維持）
        - margin_global より離れていない遠方ペアは repel（クラスタ分離性を促す）
        """
        # 距離行列: shape (N, N)
        dists = torch.cdist(latents, latents, p=2)
        N = latents.size(0)

        # 対角成分（自分自身）を除外
        eye_mask = torch.eye(N, device=dists.device).bool()
        dists_no_self = dists.masked_fill(eye_mask, float('inf'))

        ### 1. Local attraction（近傍を近づける）
        # k_local を N-1 以下に制限
        k_safe = min(k_local, N - 1)

        if k_safe > 0:
            knn_dists, _ = torch.topk(dists_no_self, k=k_safe, largest=False, dim=1)
            local_loss = torch.mean(knn_dists)
        else:
            local_loss = torch.tensor(0.0, device=latents.device)

        ### 2. Global repulsion（離れたクラスタを広げる）
        margin_mask = dists_no_self < margin_global
        repel_loss = torch.clamp(margin_global - dists_no_self, min=0.0)
        global_loss = torch.sum(repel_loss * margin_mask) / (margin_mask.sum() + 1e-8)  # avoid div by 0

        return local_loss + global_loss


    def custom_loss(
        self,
        predictions,
        predictions_cart,
        targets,
        censors,
        virus_regressor_out,
        virus_regressor_out2,
        reference_regressor_out,
        reference_regressor_out2,
        virus_embedding_original,
        reference_embedding_original,
        weight
    ):

        uncensored_loss = self.mse_loss_wo_mean(predictions, targets) * (1 - censors)
        censored_loss = self.mse_loss_wo_mean(predictions, torch.minimum(predictions, targets)) * censors

        uncensored_loss = uncensored_loss * weight
        censored_loss = censored_loss * weight

        uncensored_loss_cart = self.mse_loss_wo_mean(predictions_cart, targets) * (1 - censors)
        censored_loss_cart = self.mse_loss_wo_mean(predictions_cart, torch.minimum(predictions_cart, targets)) * censors

        uncensored_loss_cart = uncensored_loss_cart * weight
        censored_loss_cart = censored_loss_cart * weight

        combined_latents = torch.cat([virus_regressor_out, reference_regressor_out], dim=0)
        combined_latents2 = torch.cat([virus_regressor_out2, reference_regressor_out2], dim=0)
        combined_latents_original = torch.cat([virus_embedding_original, reference_embedding_original], dim=0)

        # **Semantic CSE のように「自分以外を遠ざける」Contrastive Loss**
        contrastive_loss_value = self.contrastive_loss_semantic(combined_latents, combined_latents2, alpha=self.CSE_ALPHA)
        local_global_loss_value = self.local_global_loss(combined_latents, k_local=3, margin_global=0.125)


        semantic_loss = self.compute_semantic_loss(combined_latents,combined_latents_original,self.embed_scale,self.embed_scale_factor)

        total_loss = (
            torch.mean(uncensored_loss) * self.MAIN_W
            + torch.mean(censored_loss) * self.MAIN_W
            + torch.mean(uncensored_loss_cart) * self.CART_W
            + torch.mean(censored_loss_cart) * self.CART_W
            + contrastive_loss_value * self.CSE_W  # repel_gain を Semantic CSE のように変更
            + semantic_loss * self.SEMANTIC_W
            + local_global_loss_value * self.LG_W
        )

        return total_loss



    def custom_loss_only_semantic(
        self,
        virus_regressor_out,
        virus_regressor_out2,
        virus_embedding_original
    ):
        """
        - `semantic_loss` で実験データの距離を再現
        - `contrastive_loss_semantic` で全体のバランスを調整
        """
        # **1 Contrastive Loss (Semantic CSE) でバランスの良い埋め込みを作る**
        contrastive_loss_value = self.contrastive_loss_semantic(virus_regressor_out, virus_regressor_out2, alpha=self.CSE_ALPHA)
        local_global_loss_value = self.local_global_loss(virus_regressor_out, k_local=3, margin_global=0.125)

        semantic_loss = self.compute_semantic_loss(virus_regressor_out,virus_embedding_original,self.embed_scale,self.embed_scale_factor)

        # **Total Loss**
        total_loss = (
            semantic_loss * self.SEMANTIC_W_VIRUS_ONLY  # 実験データの距離を反映
            + contrastive_loss_value * self.CSE_W_VIRUS_ONLY  # Semantic CSE で全体のバランス調整
            + local_global_loss_value * self.LG_W
        )
        return total_loss

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        **kwargs,
    ):
        from transformers import EsmConfig
        from safetensors.torch import load_file
        from pathlib import Path

        if config is None:
            config = EsmConfig.from_pretrained(pretrained_model_name_or_path)

        model = cls(config, *model_args, **kwargs)

        checkpoint_dir = Path(pretrained_model_name_or_path)
        safetensor_files = sorted(checkpoint_dir.glob("model-*-of-*.safetensors"))

        print(f"[INFO] Found {len(safetensor_files)} safetensors.")

        state_dict = {}
        for f in safetensor_files:
            print(f"[INFO] Loading {f}")
            part = load_file(str(f), device="cpu")
            state_dict.update(part)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("[INFO] Model state_dict loaded.")
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        return model

    def forward(
        self,
        input_ids_virus: torch.Tensor,
        attention_mask_virus: torch.Tensor,
        input_ids_reference: Optional[torch.Tensor] = None,
        attention_mask_reference: Optional[torch.Tensor] = None,
        censors=None,
        labels=None,
        virus=None,
        reference=None,
        dates=None,
        virus_passage=None,
        reference_passage=None,
        weight=None,
        **kwargs,
    ):
        
        
        # labelsがNoneかどうかをサンプル単位で判定
        has_labels_mask = labels.ne(float(-10.0)) if labels is not None else torch.zeros(input_ids_virus.shape[0], dtype=torch.bool, device=self.device)
        # `input_ids_reference` が None の場合の処理（DataLoaderのKeyErrorを回避）
        if input_ids_reference is None:
            input_ids_reference = torch.zeros_like(input_ids_virus, device=self.device)
            attention_mask_reference = torch.zeros_like(attention_mask_virus, device=self.device)
        
        # ラベルあり・なしでデータを分割
        virus_regressor_out = self.regressor(self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus))
        virus_regressor_out2 = self.regressor(self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus))

        if self.training == False and labels is None:
            return ModelOutput(hidden_state_virus=virus_regressor_out)

        virus_embedding_original = self.extract_pooled_embeddings(self.esm_model_original, input_ids_virus, attention_mask_virus)

        # labelsがNoneのサンプルに対してはcustom_loss_only_semanticを計算
        virus_regressor_out_no_labels = virus_regressor_out[~has_labels_mask]
        virus_regressor_out2_no_labels = virus_regressor_out2[~has_labels_mask]
        virus_embedding_original_no_labels = virus_embedding_original[~has_labels_mask]
        
        if virus_regressor_out_no_labels.shape[0] > 0:
            loss_no_labels = self.custom_loss_only_semantic(virus_regressor_out_no_labels, virus_regressor_out2_no_labels, virus_embedding_original_no_labels)
        else:
            loss_no_labels = torch.tensor(0.0, device=self.device)
        
        # labelsがあるサンプルのみreferenceの処理を行う
        if has_labels_mask.any():
            input_ids_reference = input_ids_reference[has_labels_mask]
            attention_mask_reference = attention_mask_reference[has_labels_mask]
            reference_regressor_out = self.regressor(self.encode_sequence(self.esm_model, input_ids_reference, attention_mask_reference))
            reference_regressor_out2 = self.regressor(self.encode_sequence(self.esm_model, input_ids_reference, attention_mask_reference))
            
            distance = torch.norm(virus_regressor_out[has_labels_mask] - reference_regressor_out, p=2, dim=1, keepdim=True)
            
            virus_encoding = self.encode_one_hot(OHE_virus, virus[has_labels_mask])
            reference_encoding = self.encode_one_hot(OHE_ref, reference[has_labels_mask])
            virus_passage_encoding = self.encode_one_hot(OHE_vp, virus_passage[has_labels_mask])
            reference_passage_encoding = self.encode_one_hot(OHE_rp, reference_passage[has_labels_mask])
            
            combined_encoding = torch.cat([reference_encoding, virus_passage_encoding, reference_passage_encoding], dim=-1)
            systematic_error1 = self.virus_effects(virus_encoding)
            systematic_error2 = self.systematic_error_effects(combined_encoding)
            systematic_error = systematic_error1 + systematic_error2
            observed_distance = distance + systematic_error
            
            logits = torch.cat((observed_distance, distance), dim=1)
            reference_embedding_original = self.extract_pooled_embeddings(self.esm_model_original, input_ids_reference, attention_mask_reference)
            
            combined_loss = self.custom_loss(
                observed_distance,
                distance,
                labels[has_labels_mask].to(self.device).view(-1, 1),
                censors[has_labels_mask].to(self.device).view(-1, 1),
                virus_regressor_out[has_labels_mask],
                virus_regressor_out2[has_labels_mask],
                reference_regressor_out,
                reference_regressor_out2,
                virus_embedding_original[has_labels_mask],
                reference_embedding_original,
                weight[has_labels_mask].to(self.device).view(-1, 1)
            )
            systematic_error1_L2 = torch.mean(systematic_error1 ** 2)
            systematic_error2_L2 = torch.mean(systematic_error2 ** 2)
            combined_loss = combined_loss + (systematic_error1_L2 + systematic_error2_L2) * 1.0E-4
        else:
            combined_loss = torch.tensor(0.0, device=self.device)
            logits = None
        
        total_loss = combined_loss + loss_no_labels
        
        return ModelOutput(loss=total_loss, logits=logits, hidden_state_virus=virus_regressor_out)

