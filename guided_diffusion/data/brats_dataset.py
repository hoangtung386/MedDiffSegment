import os
import os.path

import nibabel
import numpy as np
import torch
import torch.nn
import torchvision.utils as vutils


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag = test_flag
        if test_flag:
            self.seqtypes = ["t1", "t1ce", "t2", "flair"]
        else:
            self.seqtypes = ["t1", "t1ce", "t2", "flair", "seg"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            if not dirs:
                files.sort()
                files = [f for f in files if f.endswith(".nii.gz") or f.endswith(".nii")]

                if len(files) == 0:
                    continue

                datapoint = dict()
                for f in files:
                    try:
                        seqtype = f.split("_")[2].split(".")[0]
                        datapoint[seqtype] = os.path.join(root, f)
                    except IndexError:
                        print(f"Warning: Cannot parse filename {f}, skipping...")
                        continue

                if self.seqtypes_set.issubset(set(datapoint.keys())):
                    self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path = filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image = out
            image = image[..., 8:-8, 8:-8]
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:
            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]
            label = label[..., 8:-8, 8:-8]
            label = torch.where(label > 0, 1, 0).float()
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)


class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag = test_flag
        if test_flag:
            self.seqtypes = ["t1", "t1ce", "t2", "flair"]
        else:
            self.seqtypes = ["t1", "t1ce", "t2", "flair", "seg"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        # Collect all .nii/.nii.gz files recursively
        all_files = {}
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                if f.endswith(".nii.gz") or f.endswith(".nii"):
                    full_path = os.path.join(root, f)
                    all_files[full_path] = f

        print(f"Found {len(all_files)} .nii/.nii.gz files in total")

        # Group files by patient
        patients = {}
        for filepath, filename in all_files.items():
            # Extract patient identifier from path
            # e.g., /path/patient001/... -> patient001
            path_parts = filepath.split(os.sep)
            patient_id = None
            for part in path_parts:
                if part.startswith("patient"):
                    patient_id = part
                    break

            if not patient_id:
                print(f"Warning: Cannot identify patient for {filepath}")
                continue

            if patient_id not in patients:
                patients[patient_id] = {}

            # Extract modality from filename
            fname_lower = filename.lower()
            seqtype = None

            for mod in ["flair", "t1ce", "t1", "t2", "seg"]:
                if mod in fname_lower:
                    if mod == "t1" and "t1ce" in fname_lower:
                        continue
                    seqtype = mod
                    break

            if seqtype:
                patients[patient_id][seqtype] = filepath
                print(f"✓ {patient_id}: {filename} → {seqtype}")
            else:
                print(f"✗ Cannot identify modality: {filename}")

        # Validate and add complete patients to database
        print(f"\n{'='*60}")
        print("Patient validation:")
        print(f"{'='*60}")

        for patient_id in sorted(patients.keys()):
            datapoint = patients[patient_id]
            found_mods = set(datapoint.keys())

            print(f"\n{patient_id}:")
            print(f"  Found: {found_mods}")
            print(f"  Required: {self.seqtypes_set}")

            if self.seqtypes_set.issubset(found_mods):
                self.database.append(datapoint)
                print(f"  ✓ ADDED to dataset")
            else:
                missing = self.seqtypes_set - found_mods
                print(f"  ✗ SKIPPED - Missing: {missing}")

        print(f"\n{'='*60}")
        print(f"Loaded {len(self.database)} complete patient scans")
        print(f"Total slices: {len(self.database) * 155}")
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.database) * 155

    def __getitem__(self, x):
        n = x // 155
        slice_idx = x % 155
        filedict = self.database[n]
        path = filedict[self.seqtypes[0]]

        # Load full 3D volumes
        volumes = {}
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            volumes[seqtype] = torch.tensor(nib_img.get_fdata())

        # Create 2D data (center slice with all 4 modalities)
        image_2d_modalities = []
        for s in self.seqtypes:
            if s != "seg":
                image_2d_modalities.append(volumes[s][..., slice_idx])
        image_2d = torch.stack(image_2d_modalities)

        # Create 2.5D data (3 consecutive slices from flair)
        vol_2_5d = volumes.get("flair", volumes[self.seqtypes[0]])
        num_slices_2_5d = 3
        half_slices = num_slices_2_5d // 2

        slices_for_stack = []
        for i in range(slice_idx - half_slices, slice_idx + half_slices + 1):
            clamped_idx = np.clip(i, 0, vol_2_5d.shape[2] - 1)
            slices_for_stack.append(vol_2_5d[..., clamped_idx])

        # Shape: (H, W, 3) -> (1, H, W, 3) for Conv3D
        image_2_5d = torch.stack(slices_for_stack, dim=-1).unsqueeze(0)

        # Handle label
        if self.test_flag:
            label_2d = image_2d
        else:
            label_vol = volumes["seg"]
            label_2d = label_vol[..., slice_idx].unsqueeze(0)
            label_2d = torch.where(label_2d > 0, 1, 0).float()

        # Apply transformations
        if self.transform:
            state = torch.get_rng_state()
            image_2d = self.transform(image_2d)
            image_2_5d = self.transform(image_2_5d)
            if not self.test_flag:
                torch.set_rng_state(state)
                label_2d = self.transform(label_2d)

        batch_image = (image_2d, image_2_5d)
        virtual_path = path.split(".nii")[0] + "_slice" + str(slice_idx) + ".nii"

        if self.test_flag:
            return (batch_image, batch_image, virtual_path)

        return (batch_image, label_2d, virtual_path)
