import torch.nn as nn
from torchvision import transforms
import torch
from PIL import Image
class ASLClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(ASLClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifierCNN(num_classes=27).to(device)
model.load_state_dict(torch.load('../main model/best_model.pth', map_location=device))
model.eval()  # Set the model to evaluation mode


# Define a function to make an inference
def make_inference(image_path):
    # Load and preprocess the image
    transform = transforms.Compose([
        # transforms.Resize((128, 128)),  # Resize image to match input dimensions
        transforms.ToTensor(),  # Convert image to tensor
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])

    image = Image.open(image_path).convert('RGB')  # Open the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move input tensor to device
    input_tensor = input_tensor.to(device)

    # Perform inference
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)  # Get the predicted class label

    return predicted_class.item()


if __name__ == "__main__":
    image_path = 'isolated_hand.png'
    predicted_label = make_inference(image_path)
    print(f'Predicted Class: {predicted_label}')
