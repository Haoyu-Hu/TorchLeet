"""B.5i 参数初始化（评测来自仓库的 `cnn-param-init` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.5i
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim


def train_test_loop(model, train_loader, test_loader, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for image, label in train_loader:
            pred = model(image)
            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
        print(f"Training loss at epoch {epoch} = {loss.item()}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image_test, label_test in test_loader:
            pred_test = model(image_test)
            _, pred_test_vals = torch.max(pred_test, dim=1)
            total += label_test.size(0)
            correct += (pred_test_vals == label_test).sum().item()
    print(f"Test Accuracy = {(correct * 100)/total}")


class VanillaCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def config_init(init_type="kaiming"):
    # TODO: Add Kaiming initialization according to a CNN or a Linear layer
    def kaiming_init(m):
        ...
            
    # TODO: Implement Xavier initialization        
    def xavier_init(m):
        ...
    
    # TODO: Initialize weights and biases to zero          
    def zeros_init(m):
        ...
    
    # TODO: Initialize weights and biases from a normal distribution          
    def random_init(m):
        ...
    

    initializer_dict = {"kaiming": kaiming_init,
                        "xavier": xavier_init,
                        "zeros": zeros_init,
                        "random": random_init}
    
    return initializer_dict.get(init_type)
