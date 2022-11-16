from torchvision import datasets
data_path = r"C:\Users\Vadim\Desktop\folders\For myself\Python\datasets\cv"

cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)