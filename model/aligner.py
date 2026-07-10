import torch
from torch import nn
import torch.nn.functional as F
from monotonic_alignment_search import maximum_path

# From https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/tts/layers/generic/aligner.py
# Modifications inspierd by ZDisket's FastSpeech2 fork
class AlignmentNetwork(torch.nn.Module):
    """Aligner Network for learning alignment between the input text and the model output with Gaussian Attention.

    ::

        query -> conv1d -> relugt -> conv1d -> relugt -> conv1d -> L2_dist -> softmax -> alignment
        key   -> conv1d -> relugt -> conv1d -----------------------^

    Args:
        in_query_channels (int): Number of channels in the query network. Defaults to 80.
        in_key_channels (int): Number of channels in the key network. Defaults to 512.
        attn_channels (int): Number of inner channels in the attention layers. Defaults to 80.
        temperature (float): Temperature for the softmax. Defaults to 0.0005.
    """

    def __init__(
        self,
        in_query_channels=80,
        in_key_channels=512,
        attn_channels=80,
        temperature=0.0005,
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_layer = nn.Sequential(
            nn.Conv1d(
                in_key_channels,
                in_key_channels * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            ReLUGT(),
            nn.Conv1d(in_key_channels * 2, attn_channels, kernel_size=1, padding=0, bias=True),
        )

        self.query_layer = nn.Sequential(
            nn.Conv1d(
                in_query_channels,
                in_query_channels * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            ReLUGT(),
            nn.Conv1d(in_query_channels * 2, in_query_channels, kernel_size=1, padding=0, bias=True),
            ReLUGT(),
            nn.Conv1d(in_query_channels, attn_channels, kernel_size=1, padding=0, bias=True),
        )

        self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(self.key_layer[0].weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self.key_layer[2].weight, gain=torch.nn.init.calculate_gain("linear"))
        torch.nn.init.xavier_uniform_(self.query_layer[0].weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self.query_layer[2].weight, gain=torch.nn.init.calculate_gain("linear"))
        torch.nn.init.xavier_uniform_(self.query_layer[4].weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(
        self, queries: torch.tensor, keys: torch.tensor, mask: torch.tensor = None, attn_prior: torch.tensor = None
    ) -> tuple[torch.tensor, torch.tensor]:
        """Forward pass of the aligner encoder.
        Shapes:
            - queries: :math:`[B, C, T_de]`
            - keys: :math:`[B, C_emb, T_en]`
            - mask: :math:`[B, T_de]`
        Output:
            attn (torch.tensor): :math:`[B, 1, T_en, T_de]` soft attention mask.
            attn_logp (torch.tensor): :math:`[ßB, 1, T_en , T_de]` log probabilities.
        """
        key_out = self.key_layer(keys)
        query_out = self.query_layer(queries)
        attn_factor = (query_out[:, :, :, None] - key_out[:, :, None]) ** 2
        attn_logp = -self.temperature * attn_factor.sum(1, keepdim=True)
        if attn_prior is not None:
            attn_logp = self.log_softmax(attn_logp) + torch.log(attn_prior[:, None] + 1e-8)

        if mask is not None:
            attn_logp.data.masked_fill_(~mask.bool().unsqueeze(2), -float("inf"))

        attn = self.softmax(attn_logp)
        return attn, attn_logp

# https://github.com/ZDisket/FastSpeech2/blob/isolate-preencoder/model/subatts.py
class ReLUGT(nn.Module):
    """
    ReLU GT: Leaky squared ReLU with trainable positive alpha, slope, and static negative alpha.
    Early experiments show near parity with APTx S1 with faster initial fitting. Only squares positive part.
    """
    def __init__(self, initial_slope=0.05, initial_alpha_neg=2.5, initial_alpha_pos=1.0):
        super(ReLUGT, self).__init__()
        self.slope = nn.Parameter(torch.tensor(initial_slope))
        self.alpha_neg = initial_alpha_neg
        self.alpha_pos = nn.Parameter(torch.tensor(initial_alpha_pos))

    def forward(self, x):
        return torch.where(x < 0, self.alpha_neg * self.slope * x, self.alpha_pos * x ** 2)


# My own shitty code
class Aligner(nn.Module):
    """
    Wrapper for MAS and the aligner network.
    """

    def __init__(self, in_query_channels=80, in_key_channels=512, attn_channels=80, temperature=0.0005):
        super().__init__()
        self.aligner_network = AlignmentNetwork(in_query_channels, in_key_channels, attn_channels, temperature)
    
    def forward(self, input_seq, output_seq, input_seq_mask, output_seq_mask):
        
        # Might Change because ming024's FS2 implementation is diferent from CoquiAI
        attn_mask = torch.unsqueeze(input_seq_mask, -1) * torch.unsqueeze(output_seq_mask, 2) 
        alignment_soft, alignment_logprob = self.aligner_network(output_seq.transpose(1, 2), input_seq.transpose(1, 2), input_seq_mask, None)
        alignment_mas = maximum_path(
            alignment_soft.squeeze(1).transpose(1, 2).contiguous(), attn_mask.squeeze(1).contiguous()
        )
        o_alignment_dur = torch.sum(alignment_mas, -1).int()
        alignment_soft = alignment_soft.squeeze(1).transpose(1, 2)
        return o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas