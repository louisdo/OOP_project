import torch

class CustomCrossEntropyLoss(torch.nn.Module):
    """
    Custom cross entropy loss
    """
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

        self.crossentropy = torch.nn.CrossEntropyLoss(reduction = 'none')

    def forward(self, 
                predictions: torch.tensor, 
                target: torch.tensor, 
                tgt_key_padding_mask: torch.ByteTensor):

        """
        input:
        + predictions: torch.tensor of shape [B, V, L], with 'B' be the batchsize
        'V' be the vocab size and 'L' is the max training sequence length;
        the predictions of the model
        + target: torch.tensor of shape [B, L], with 'B' be the batchsize, 'L' be the
        max training sequence length; 
        the ground-truth sequence
        + tgt_key_padding_mask: torch.ByteTensor of shape [B, L], with 'B' be the batchsize, 
        'L' be the max training sequence length; the mask for padding masking
        when computing the the loss function
        """
        loss = self.crossentropy(predictions, target)
        loss = torch.mul(loss, tgt_key_padding_mask.float())
        return torch.sum(loss) / torch.sum(tgt_key_padding_mask.float())
