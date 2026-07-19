import torch

N = torch.zeros((54,54),dtype=torch.int32)

words =  open("01.txt","r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:x+1 for x,s in enumerate(chars)}
itos = {x+1:s for x,s in enumerate(chars)}
stoi['.'] = 0
itos[0] = "."
for w in words:
    chs = ["."] + list(w) + ["."]    
    for w1,w2 in zip(chs,chs[1:]): # 
        ich1=stoi[w1]                
        ich2=stoi[w2]
        N[ich1,ich2] += 1
s = torch.Generator().manual_seed(156416)

#N = torch.ones((54,54),dtype=torch.int32)  -> to see untrained values
for x in range (10):
    ix = 0
    out = []
    while True:
    
        p = N[ix].float()
        p = p / p.sum()
        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=s).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))

