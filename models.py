import torch
import torchvision.models

model = torchvision.models.vgg16(weights='DEFAULT')
torch.save(model.state_dict(), 'model_weights.pth')

model = torchvision.models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

torch.save(model, 'model.pth')
model = torch.load('model.pth')
