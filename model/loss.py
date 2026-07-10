import torch
import torch.nn as nn
import torch.nn.functional as F


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, alignment_hard, alignment_soft):
        log_sum = torch.log(torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)).sum()
        return -log_sum / alignment_hard.sum()

class MSE1D(nn.Module):
    """Mean Squared Error Loss for 1D sequences with masking (batch_size, seq_len)."""

    def __init__(self):
        super(MSE1D, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Squared Error loss between predictions and ground truth for 1D sequences,
        considering a mask to exclude certain entries from the loss computation.

        Parameters:
            x (torch.Tensor): Predictions of shape (batch_size, seq_len).
            y (torch.Tensor): Ground truth of shape (batch_size, seq_len).
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len), where True means excluded (invalid).

        Returns:
            torch.Tensor: Computed MSE loss for valid elements.
        """

        assert x.shape == y.shape, f"Shape mismatch between predictions and ground truth, {x.size()}, vs {y.size()}"
        assert mask.shape == x.shape, "Shape mismatch between mask and predictions/ground_truth"

        # Apply the mask by selecting elements where mask is False
        valid_x = x[~mask]
        valid_y = y[~mask]

        # Calculate MSE for the valid elements
        mse_loss = F.mse_loss(valid_x, valid_y, reduction='mean')  # Calculate mean only over the unmasked elements
        return mse_loss

class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = torch.nn.functional.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss = total_loss + loss

        total_loss = total_loss / attn_logprob.shape[0]
        return total_loss


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]



        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.mse2_loss = MSE1D()
        self.forward_sum = ForwardSumLoss()
        self.bin_loss = BinLoss()

        self.bin_loss_start_epoch = model_config["aligner"]["bin_loss_start_epoch"]
        self.bin_loss_warmup_epochs = model_config["aligner"]["bin_loss_warmup_epochs"]

    def forward(self, inputs, predictions, epoch=0):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,

            input_lengths,
            output_lengths,
            attn_logprob,
            attn_hard,
            attn_soft,
            attn_hard_dur,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(attn_hard_dur.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        #log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        #log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)

        duration_loss = self.mse2_loss(
            log_duration_predictions,
            log_duration_targets,
            ~src_masks,
        )

        output_lengths = torch.clamp_max(output_lengths, attn_logprob.size(2))
        al_forward_sum = self.forward_sum(attn_logprob=attn_logprob, in_lens=input_lengths, out_lens=output_lengths)

        total_attn_loss = al_forward_sum

        if epoch > self.bin_loss_start_epoch:
            bin_loss_scale = min((epoch - self.bin_loss_start_epoch) / self.bin_loss_warmup_epochs, 1.0)
            al_match_loss = self.bin_loss(attn_hard, attn_soft) * bin_loss_scale
            total_attn_loss += al_match_loss



        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + total_attn_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            total_attn_loss
        )
