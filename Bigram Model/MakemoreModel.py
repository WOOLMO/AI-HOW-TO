import torch # we import the pytorch
import torch.nn.functional as F
N = torch.zeros((54,54),dtype=torch.int32) # we create the 54 by 54 matrix P.S : 54 bcuz the number of existing characters is 54

words =  open("01.txt","r").read().splitlines() # listing the words in a nice splitlines list

chars = sorted(list(set(''.join(words)))) # exporting the needed characters ! 54!
stoi = {s:x+1 for x,s in enumerate(chars)} # creating a mapping for each character string to index and vice versa index to string
itos = {x+1:s for x,s in enumerate(chars)}
stoi['.'] = 0 # the end and start token of the ord is a . we map it on both mappings
itos[0] = "."
for w in words:
    chs = ["."] + list(w) + ["."]    
    for w1,w2 in zip(chs,chs[1:]): # 
        ich1=stoi[w1]                
        ich2=stoi[w2]
        N[ich1,ich2] += 1 # we compute how often a combination accures by adding one on each duo lol
s = torch.Generator().manual_seed(859674)

#N = torch.ones((54,54),dtype=torch.int32)  -> to see untrained values
p = (N+1).float() # making the matrix and how often the number appears floating numbers for precision for sure and plus 1 to kinda smooth the model so it assumes there are no impossible combination between characters even the "." aka model smothing
p /= p.sum(1 , keepdim=True) # to surely  keep the dimension of the matrix to 1 so it becomes broadcastable !
out = []
for x in range (10):
    ix = 0
   
    while True:
    
        z = p[ix] # we get for each row its propabilities while the sum per example of p[x] is forcely a 1 bcuz we normalize it !
        ix = torch.multinomial(z,num_samples=1,replacement=True,generator=s).item() # using the multinomial function and based on a seed. we determine the most likely character to follow up (by its index for sure ) so that index got recycled to the loop and get its follow up with the same way till it hit a wall "." lol 
        out.append(itos[ix]) # we append it to a list 
        if ix == 0: # if its an end char the loop ends and give us the names
            out.append("\n")
            break
print("".join(out))
log_likelyhood=0
h=0
for w in words:
    chs = ["."] + list(w) + ["."]    
    for w1,w2 in zip(chs,chs[1:]): # 
        ich1=stoi[w1]                
        ich2=stoi[w2]
        propability = p[ich1,ich2]
        logprob = torch.log(propability)
        log_likelyhood += logprob
        h+=1
negativeloglikelyhood = -log_likelyhood
loss = negativeloglikelyhood / h
print(loss.item())




a,b = [],[]
for w in words:
    chs = ["."]+ list(w) +["."]
    for w1,w2 in zip(chs,chs[1:]):
        ich1 = stoi[w1]
        ich2 = stoi[w2]
        a.append(ich1)
        b.append(ich2)
a = torch.tensor(a)
b = torch.tensor(b)
enca = F.one_hot(a, num_classes=54).float()
p = torch.Generator().manual_seed(5165156)
w = torch.randn((54,54),generator=p,requires_grad=True)
lol = 0
while lol < 10.9:
    for zb in range (1,10):
        
        #forward pass 
            nlogits = enca @ w
            nprobsb = torch.exp(nlogits)
            nprobs = nprobsb / nprobsb.sum(dim = 1 , keepdim=True)
        # print(b)
        # loss Function initialization
            floss = -(nprobs[torch.arange(0,len(a)),b]).log().mean() + 0.001 * (w**2).mean()
        # Backpropagation But Dont Forgot To Zero The Old Grads It Affect New Grads
            w.grad = None     
            floss.backward()
        # Actual Learning And Stepping UPDATE!
            # print(f"trained...{floss.item()}")
            w.data += -50 * w.grad
            lol = ((torch.exp(-(floss))*100).mean()).item()
print(f"here is the neuron model loss {floss}")
print()
out = []
s2 = torch.Generator().manual_seed(859674)
for x in range (10):
    ix = 0
  
    while True:
        xenc0 = F.one_hot( torch.tensor([ix]) , num_classes=54 ).float()  
        logits0 = xenc0 @ w # we get for each row its propabilities while the sum per example of p[x] is forcely a 1 bcuz we normalize it !
        counts = logits0.exp()
        shitx = counts / counts.sum(dim=1,keepdim=True)
        ix = torch.multinomial(shitx,num_samples=1,replacement=True,generator=s2).item() # using the multinomial function and based on a seed. we determine the most likely character to follow up (by its index for sure ) so that index got recycled to the loop and get its follow up with the same way till it hit a wall "." lol 
        out.append(itos[ix]) # we append it to a list 
        if ix == 0: # if its an end char the loop ends and give us the names
            out.append("\n")
            break

print("".join(out))

