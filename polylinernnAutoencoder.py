from apiParser import loadTrainingData
import matplotlib.pyplot as plt
import torch
import polyline as pl
import pickle as pkl
from math import floor
data = loadTrainingData()
filtered = list(filter(lambda x: "map" in x and "polyline" in x["map"] and len(x["map"]["polyline"]) != 0, data))

def polylineToTensor(s, length):
    x = [float(ord(c)/255) for c in s]
    x += [0 for c in range(length - len(x))]
    return torch.tensor(x)

X = [polylineToTensor(x["map"]["polyline"], 2630) for x in filtered]
print(len(X))

# Creating a PyTorch class(From G2G)
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2630, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2630),
            torch.nn.Tanh()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Model Initialization
model = AE()
 
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# # Using an Adam Optimizer with lr = 0.1
# optimizer = torch.optim.Adam(model.parameters(),
#                              lr = .2,
#                              weight_decay = 1e-8)

optimizer = torch.optim.SGD(model.parameters(),
                            lr = .1,
                            weight_decay = 1e-20)
epochs = 2000
outputs = []
losses = []
for epoch in range(epochs):
    for polyline in X:
        # Output of Autoencoder
        reconstructed = model(polyline)
        
        # Calculating the loss function
        loss = loss_function(reconstructed, polyline)
        
        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Storing the losses in a list for plotting
    losses.append(loss)
    outputs.append((epochs, polyline, reconstructed))
    print(epoch, loss)

# plt.style.use('fivethirtyeight')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
 
# Plotting the last 100 values
# plt.plot(losses[-100:])
    
# with open("aeOut1.pkl", "rb") as file:
#     outputs = pkl.load(file)

# print("loaded")

with open("aeModel2.pkl", "wb") as file:
    pkl.dump(model, file)

points = []
# for e, p, tensorpolyline in outputs[-130:]:
#     polyline = "".join([chr(floor(c*255)) for c in tensorpolyline.tolist()])
    
#     try:
#         decoded = pl.decode(polyline)
#         points += decoded
#     except:
#         print("rip")
#         pass

# pointsf = list(filter(lambda p: abs(p[0])<300 and abs(p[1]) < 300, points))
# x, y = zip(*pointsf)
# plt.scatter(list(x), list(y))