import os
import shutil
import time
import copy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

# ------------------------------ CONFIGURATION ------------------------------ #
DATA_DIR = 'Rock-Paper-Scissors'
CLASSES = ['rock', 'paper', 'scissors']
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 5
LR = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1


# -------------------------- UTILITY FUNCTIONS ------------------------------ #
def prepare_directories(data_dir: str, val_dir_name: str = 'val') -> None:
	"""
	Создает папки для валидации, если они не существуют.
	"""
	val_dir = os.path.join(data_dir, val_dir_name)
	if not os.path.exists(val_dir):
		os.makedirs(val_dir)
		for class_name in CLASSES:
			os.makedirs(os.path.join(val_dir, class_name))
	return


def move_validation_images(data_dir: str, src_folder: str, dst_folder: str) -> None:
	"""
	Перемещает изображения для валидации из папки src_folder в dst_folder.
	"""
	for class_name in CLASSES:
		src_class_dir = os.path.join(data_dir, src_folder, class_name)
		dst_class_dir = os.path.join(data_dir, dst_folder, class_name)
		if os.path.exists(src_class_dir):
			for img in os.listdir(src_class_dir):
				shutil.move(os.path.join(src_class_dir, img), os.path.join(dst_class_dir, img))
	return


def get_transforms() -> Dict[str, transforms.Compose]:
	"""
	Возвращает словарь трансформаций для обучения и валидации.
	"""
	return {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}


def get_dataloaders(data_dir: str, data_transforms: Dict[str, transforms.Compose],
					batch_size: int, num_workers: int) -> Tuple[Dict[str, torch.utils.data.DataLoader],
Dict[str, int],
list]:
	"""
	Загружает датасеты и возвращает словарь DataLoader-ов, размеров датасетов и список имён классов.
	"""
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
	dataloaders = {
		x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
									   shuffle=True, num_workers=num_workers)
		for x in ['train', 'val']
	}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes
	return dataloaders, dataset_sizes, class_names


def imshow(inp: torch.Tensor, title: str = None) -> None:
	"""
	Функция для отображения тензорного изображения.
	"""
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	plt.imshow(inp)
	plt.xticks([])  # Убираем подписи по оси X
	plt.yticks([])  # Убираем подписи по оси Y
	if title:
		plt.title(title)
	plt.pause(0.001)


def show_batch(dataloader: torch.utils.data.DataLoader, class_names: list) -> None:
	"""
	Отображает пакет изображений из DataLoader-а.
	"""
	inputs, classes = next(iter(dataloader))
	out = torchvision.utils.make_grid(inputs)
	titles = [class_names[x] for x in classes]
	imshow(out, title=", ".join(titles))
	plt.show()


def build_resnet18(num_classes: int, pretrained: bool, device: torch.device) -> Tuple[torch.nn.Module,
nn.CrossEntropyLoss,
optim.Optimizer,
lr_scheduler._LRScheduler]:
	"""
	Создает модель ResNet18.
	Если pretrained=True, замораживаются все слои, кроме последнего.
	"""
	if pretrained:
		# Предобученная модель
		model = models.resnet18(weights='IMAGENET1K_V1')
		for param in model.parameters():
			param.requires_grad = False
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, num_classes)
		optimizer = optim.SGD(model.fc.parameters(), lr=LR, momentum=MOMENTUM)
	else:
		# Обучение с нуля
		model = models.resnet18(pretrained=False)
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, num_classes)
		optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
	return model, criterion, optimizer, scheduler


def train_model(model: torch.nn.Module, dataloaders: Dict[str, torch.utils.data.DataLoader],
				dataset_sizes: Dict[str, int], device: torch.device,
				criterion: nn.Module, optimizer: optim.Optimizer,
				scheduler: lr_scheduler._LRScheduler, num_epochs: int = 5) -> torch.nn.Module:
	"""
	Функция для обучения модели.
	"""
	since = time.time()
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print(f'Epoch {epoch}/{num_epochs - 1}')
		print('-' * 10)

		for phase in ['train', 'val']:
			model.train() if phase == 'train' else model.eval()

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

			# Сохраняем лучшую модель на валидации
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since
	print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
	print(f'Best val Acc: {best_acc:.4f}')

	model.load_state_dict(best_model_wts)
	return model


# ------------------------------- MAIN LOGIC -------------------------------- #
def main() -> None:
	# Подготовка директорий
	prepare_directories(DATA_DIR, val_dir_name='val')
	# Если требуется переместить изображения (например, из папки 'validation'), раскомментируйте следующую строку:
	move_validation_images(DATA_DIR, src_folder='validation', dst_folder='val')

	# Задаем трансформации и загружаем датасеты
	data_transforms = get_transforms()
	dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR, data_transforms, BATCH_SIZE, NUM_WORKERS)

	# Определяем устройство (GPU, если доступно)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Running model on', device)

	# Визуализируем пакет изображений из обучающего датасета
	print("Displaying a batch of training images...")
	show_batch(dataloaders['train'], class_names)

	# ---------------- Обучение модели с нуля ---------------- #
	print("\nTraining ResNet-18 from scratch")
	model_scratch, criterion_scratch, optimizer_scratch, scheduler_scratch = build_resnet18(
		num_classes=len(class_names), pretrained=False, device=device)
	model_scratch = train_model(model_scratch, dataloaders, dataset_sizes, device,
								criterion_scratch, optimizer_scratch, scheduler_scratch, num_epochs=NUM_EPOCHS)

	# ---------------- Дообучение предобученной модели ---------------- #
	print("\nFine-tuning pretrained ResNet-18")
	model_pretrained, criterion_pretrained, optimizer_pretrained, scheduler_pretrained = build_resnet18(
		num_classes=len(class_names), pretrained=True, device=device)
	model_pretrained = train_model(model_pretrained, dataloaders, dataset_sizes, device,
								   criterion_pretrained, optimizer_pretrained, scheduler_pretrained,
								   num_epochs=NUM_EPOCHS)


if __name__ == '__main__':
	main()
