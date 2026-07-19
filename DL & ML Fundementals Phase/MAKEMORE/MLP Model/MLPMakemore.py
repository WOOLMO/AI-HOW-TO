import torch
import matplotlib.pyplot as plt
# here we generally make our dataset ready !

with open("01.txt","r") as o:
    data = o.read()
data = data.splitlines()
characters = sorted(list(set("".join(data))))
stoi = {s:i+1 for i,s in enumerate(characters)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}





# clearing our dataset so we can make it useful by setting the inputs and labels
blocksize = 3
def dataset_Builder(words):
    
    X , Y = [], []
    for w in words:
        #print(w)
        context = [0] * blocksize
        for ch in w+".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print("".join(itos[o] for o in context), "=>" , itos[ix])
            context = context[1:] + [ix]
            # print(context)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X , Y
import random
random.seed(42)
random.shuffle(data)
n1 = int(0.8*len(data))
n2 = int(0.9*len(data))
xtr , ytr = dataset_Builder(data[:n1])
xdev , ydev = dataset_Builder(data[n1:n2])
xte , yte = dataset_Builder(data[n2:])

o = torch.Generator().manual_seed(4654675)
C = torch.randn((54,2),generator=o)
W1 = torch.randn((6,20),generator=o)
B1 = torch.randn(20,generator=o)
W2 = torch.randn((20,54),generator=o)
B2 = torch.randn(54,generator=o)
params = [W1,W2,B1,B2,C]

for i in params:
    i.requires_grad=True
KOs = torch.linspace(-3,0,1000)
Sos = 10 ** KOs 
lossi = []
learni = []
for i in range(28000):
    ixs = torch.randint(0, xtr.shape[0], (32,),generator=o) 
    # Forward Pass
    embeddings = C[xtr[ixs]]
    values = torch.tanh(embeddings.view(embeddings.shape[0] , 6) @ W1 + B1)
    logits = values @ W2 + B2
    # counts = logits.exp()
    # prob = (counts / counts.sum(1, keepdim = True))
    # softmaxLOSS = -prob[torch.arange(5072),Y].log().mean() # generally speaking i find out that there is that cross_entropy function which does all this softmax stuff and more fficient well behaved numerically and tada
    softmaxLOSS = torch.nn.functional.cross_entropy(logits , ytr[ixs]) 

    # Backward Pass
    
    for s in params:
        s.grad = None
    softmaxLOSS.backward()

    # lossi.append(softmaxLOSS.item())
    # learni.append(Sos[i]) plot things to see learni on function of lossi
    
    # Update !:
    lr = 0.1
    for Relaxing in params:
        Relaxing.data += - lr * Relaxing.grad #Sos[i]
# print("good Job AI Your LOSS is:" , softmaxLOSS.item())
# plt.plot(learni,lossi)
# plt.show() to plot for sure
    # Forward Pass
embeddings = C[xdev]
values = torch.tanh(embeddings.view(embeddings.shape[0] , 6) @ W1 + B1)
logits = values @ W2 + B2
    # counts = logits.exp()
    # prob = (counts / counts.sum(1, keepdim = True))
    # softmaxLOSS = -prob[torch.len(Y.item),Y].log().mean() # generally speaking i find out that there is that cross_entropy function which does all this softmax stuff and more fficient well behaved numerically and tada
softmaxLOSS = torch.nn.functional.cross_entropy(logits , ydev)
# print("good Job AI Your LOSS is:" , softmaxLOSS.item())


# Sampling From The Neural Net
gen = torch.Generator().manual_seed(15151515 + 10)

for _ in range(20):
    Context = [0] * blocksize
    out = []
    while True:
        emb = C[torch.tensor(Context)]
        fst = torch.tanh(emb.view(1 , 6) @ W1 + B1)
        logits = fst @ W2 + B2
        probs = torch.softmax(logits,dim=1)
        ix = torch.multinomial(probs , num_samples=1 , generator=gen)
        Context = Context[1:] + [ix]
        out.append(itos[ix.item()])
        if ix == 0:
            break
    print("".join(out))
