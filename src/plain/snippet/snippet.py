import torch
import torch.nn as nn

torch.manual_seed(0)
i = torch.tensor([1, 2, 3]).float()
linear = nn.Linear(3, 9)
y = nn.Linear(3, 3)


print(i)
# print(linear.weight.split(3, dim=1))
# print(linear(i))
# print(linear.weight)
print(linear.weight.shape)

with torch.no_grad():
    for k, v in y.named_parameters():
        if "weight" in k:
            v.copy_(linear.weight[:3, :])
        elif "bias" in k:
            print(v.shape, linear.bias.shape)
            v.copy_(linear.bias[:3])


print(y(i))
print(linear(i))


print(y.weight @ i.t())
print(linear.weight @ i.t())
