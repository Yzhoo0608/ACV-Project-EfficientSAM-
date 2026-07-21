import os
import cv2
from torch.utils.data import Dataset


class CamVidDataset(Dataset):

    def __init__(self, root, split="val"):

        self.root = root
        self.split = split

        self.image_dir = os.path.join(
            root,
            split
        )

        self.label_dir = os.path.join(
            root,
            split + "_labels"
        )

        self.images = sorted(
            [
                f for f in os.listdir(self.image_dir)
                if f.endswith(".png")
            ]
        )

        print("Loaded images:", len(self.images))


    def __len__(self):

        return len(self.images)


    def __getitem__(self, idx):

        img_name = self.images[idx]


        image_path = os.path.join(
            self.image_dir,
            img_name
        )


        # Convert image filename to label filename
        label_name = img_name.replace(
            ".png",
            "_L.png"
        )


        label_path = os.path.join(
            self.label_dir,
            label_name
        )


        image = cv2.imread(
            image_path
        )


        if image is None:
            raise FileNotFoundError(
                f"Image missing: {image_path}"
            )


        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        label = cv2.imread(
            label_path
        )


        if label is None:
            raise FileNotFoundError(
                f"Label missing: {label_path}"
            )


        label = cv2.cvtColor(
            label,
            cv2.COLOR_BGR2RGB
        )


        return image, label