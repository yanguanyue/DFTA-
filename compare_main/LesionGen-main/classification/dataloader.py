import csv
import os
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None, label_mapping=None, file_extension=".jpg"):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.label_mapping = label_mapping
        self.file_extension = file_extension

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations[index]
        if 'img_path' in row:
            img_path = row['img_path']
        else:
            img_path = os.path.join(self.img_dir, row['image_id'] + self.file_extension)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image {img_path} not found.")
            return None, None

        label_key = 'dx' if 'dx' in row else 'class'
        label = row[label_key]
        label = self.label_mapping[label]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations (avoid numpy -> torch.from_numpy path)
def _pil_to_tensor(image):
    image = image.convert("RGB")
    data = torch.tensor(list(image.getdata()), dtype=torch.float32)
    data = data.view(image.size[1], image.size[0], 3).permute(2, 0, 1)
    return data / 255.0

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(_pil_to_tensor),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Updated get_dataloaders function with target class filtering and oversampling
def get_dataloaders(dataset_name, batch_size=32, img_dir=None, transform=transform, return_dataset=False, target_classes=None, augment_underrepresented=True):
    label_mapping = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    }

    def _read_csv_rows(csv_file):
        with open(csv_file, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    def _filter_rows(rows, target_classes):
        if not target_classes:
            return rows
        filtered = []
        for row in rows:
            label_key = 'dx' if 'dx' in row else 'class'
            if row[label_key] in target_classes:
                filtered.append(row)
        return filtered

    def load_and_balance_data(csv_file, target_classes, augment_underrepresented):
        rows = _read_csv_rows(csv_file)
        rows = _filter_rows(rows, target_classes)

        if augment_underrepresented:
            label_key = 'dx' if rows and 'dx' in rows[0] else 'class'
            class_counts = Counter(row[label_key] for row in rows)
            if class_counts:
                max_count = max(class_counts.values())
                data_aug_rate = {label: (max_count // count) for label, count in class_counts.items()}
                print("Data Augmentation Rates:", data_aug_rate)

                augmented_rows = []
                for row in rows:
                    rate = data_aug_rate.get(row[label_key], 1)
                    augmented_rows.extend([row] * max(1, rate))
                rows = augmented_rows

        return rows

    if dataset_name == 'HAM':
        # Assume CSV files are in a splits folder
        if img_dir and os.path.isdir(os.path.join(img_dir, 'splits')):
            base_dir = img_dir
        else:
            base_dir = os.path.dirname(img_dir) if img_dir else '.'
        train_csv_file = os.path.join(base_dir, 'splits', 'train.csv')
        val_csv_file = os.path.join(base_dir, 'splits', 'val.csv')
        test_csv_file = os.path.join(base_dir, 'splits', 'test.csv')

        # Load, filter, and balance training data
        train_annotations = load_and_balance_data(train_csv_file, target_classes, augment_underrepresented= False)
        val_annotations = load_and_balance_data(val_csv_file, target_classes, augment_underrepresented=False)
        test_annotations = load_and_balance_data(test_csv_file, target_classes, augment_underrepresented=False)

    elif dataset_name == 'DMF':
        base_dir = os.path.dirname(img_dir) if img_dir else '.'
        train_csv_file = os.path.join(base_dir, 'splits', 'train.csv')
        val_csv_file = os.path.join(base_dir, 'splits', 'val.csv')
        test_csv_file = os.path.join(base_dir, 'splits', 'test.csv')
        train_annotations = load_and_balance_data(train_csv_file, target_classes, augment_underrepresented)
        val_annotations = load_and_balance_data(val_csv_file, target_classes, augment_underrepresented=False)
        test_annotations = load_and_balance_data(test_csv_file, target_classes, augment_underrepresented=False)

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    # Create datasets
    train_dataset = CustomDataset(annotations=train_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")
    val_dataset = CustomDataset(annotations=val_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")
    test_dataset = CustomDataset(annotations=test_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if return_dataset:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_loader, val_loader, test_loader


# import os
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from torchvision import transforms
# from PIL import Image

# class CustomDataset(Dataset):
#     def __init__(self, annotations, img_dir, transform=None, label_mapping=None, file_extension=".jpg"):
#         self.annotations = annotations
#         self.img_dir = img_dir
#         self.transform = transform
#         self.label_mapping = label_mapping
#         self.file_extension = file_extension

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.img_dir, self.annotations.iloc[index]['image_id'] + self.file_extension)
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except FileNotFoundError:
#             print(f"Warning: Image {img_path} not found.")
#             return None, None

#         label = self.annotations.iloc[index]['dx']
#         label = self.label_mapping[label]
#         label = torch.tensor(label, dtype=torch.long)

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # Updated get_dataloaders function with target class filtering and WeightedRandomSampler
# def get_dataloaders(dataset_name, batch_size=32, img_dir=None, transform=transform, return_dataset=False, target_classes=None):
#     label_mapping = {
#         'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
#     }

#     def load_data(csv_file, target_classes):
#         # Load dataset
#         annotations = pd.read_csv(csv_file)

#         # Filter for target classes
#         if target_classes:
#             annotations = annotations[annotations['dx'].isin(target_classes)]

#         return annotations

#     if dataset_name == 'HAM':
#         train_csv_file = 'Datasets/HAM/splits/train.csv'
#         val_csv_file = 'Datasets/HAM/splits/val.csv'
#         test_csv_file = 'Datasets/HAM/splits/test.csv'

#         train_annotations = load_data(train_csv_file, target_classes)
#         val_annotations = load_data(val_csv_file, target_classes)
#         test_annotations = load_data(test_csv_file, target_classes)

#     elif dataset_name == 'DMF':
#         train_csv_file = 'Datasets/DMF/splits/train.csv'
#         val_csv_file = 'Datasets/DMF/splits/val.csv'
#         test_csv_file = 'Datasets/DMF/splits/test.csv'

#         train_annotations = load_data(train_csv_file, target_classes)
#         val_annotations = load_data(val_csv_file, target_classes)
#         test_annotations = load_data(test_csv_file, target_classes)

#     else:
#         raise ValueError(f"Dataset '{dataset_name}' not supported.")

#     # Create datasets
#     train_dataset = CustomDataset(annotations=train_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")
#     val_dataset = CustomDataset(annotations=val_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")
#     test_dataset = CustomDataset(annotations=test_annotations, img_dir=img_dir, transform=transform, label_mapping=label_mapping, file_extension=".jpg" if dataset_name == 'HAM' else ".png")

#     # Calculate class weights for the sampler
#     class_counts = train_annotations['dx'].value_counts()
#     class_weights = {label_mapping[label]: 1.0 / count for label, count in class_counts.items()}
#     sample_weights = [class_weights[label_mapping[label]] for label in train_annotations['dx']]
    
#     # Create sampler
#     sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

#     # Create DataLoaders
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
#     val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#     if return_dataset:
#         return train_dataset, val_dataset, test_dataset
#     else:
#         return train_loader, val_loader, test_loader

