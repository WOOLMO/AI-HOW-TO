import torch 
labels = torch.tensor([1 , 0 , 5])
logits = torch.randn(3, 6)
# we go our logits after the non linearity function , so we need to apply the softmax function to get the probabilities! and the loss function generally.
max_logits = logits.max(dim=1,keepdim=True).values
stable_logits = logits - max_logits

exponentials = torch.exp(stable_logits)
sum_exponentials = exponentials.sum(dim=1, keepdim=True)
inv_sum_exponentials = sum_exponentials ** -1
probabilities = exponentials * inv_sum_exponentials
# Softmax function is now applied and we are ready to calculate the loss function. 
logprobabilities = torch.log(probabilities)
loss = -logprobabilities[range(len(labels)), labels ].mean()
# Now As You See Those Two Lines Are The Implementation Of The Cross Entropy Loss Function.
print(loss) # Just Curious To See The Loss Value.  