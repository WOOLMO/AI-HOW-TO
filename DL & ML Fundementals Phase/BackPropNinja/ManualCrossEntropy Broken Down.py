import torch 
logits = torch.randn(3, 6)
# we go our logits after the non linearity function , so we need to apply the softmax function to get the probabilities! and the loss function generally.
max_logits = logits.max(dim=1,keepdim=True).values
stable_logits = logits - max_logits

exponentials = torch.exp(stable_logits)