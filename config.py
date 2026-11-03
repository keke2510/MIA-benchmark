""" 
configurations for this project
"""

# Imports here
import os
from datetime import datetime

# On Windows ``log_files\model`` and ``log_files/model`` are the same place,
# but on Linux a backslash is an ordinary filename character.  Translate the
# default on POSIX so the same relative tree is used everywhere.  Set
# ``MIA_CHECKPOINT_PATH`` to point at an existing tree instead (for example
# when restoring a backup whose directory literally contains a backslash).
CHECKPOINT_PATH = os.environ.get("MIA_CHECKPOINT_PATH") or r"log_files\model"
if os.name != "nt" and "MIA_CHECKPOINT_PATH" not in os.environ:
    CHECKPOINT_PATH = CHECKPOINT_PATH.replace("\\", "/")

# total training epochs

# Training parameters for the tasks; milestones are when the learning rate gets lowered
PinsFaceRecognition_EPOCHS = 200
PinsFaceRecognition_MILESTONES = [60, 120, 160]

Cifar100_EPOCHS = 50
Cifar100_MILESTONES = [15, 30, 40]

Cifar10_EPOCHS = 20
Cifar10_MILESTONES = [8, 12, 16]

Cifar20_EPOCHS = 40
Cifar20_MILESTONES = [15, 30, 35]

Cinic10_EPOCHS = 50
Cinic10_MILESTONES = [20, 35, 45]

TinyImageNet_EPOCHS = 100
TinyImageNet_MILESTONES = [40, 70, 90]

Cifar19_EPOCHS = 40
Cifar19_MILESTONES = [15, 30, 35]

Cifar15_EPOCHS = 40
Cifar15_MILESTONES = [15, 30, 35]

Imagenet64_EPOCHS = 200
Imagenet64_MILESTONES = [120, 180, 250]

GTSRB_EPOCHS = 50
GTSRB_MILESTONES = [30, 35]

Cifar10_ViT_EPOCHS = 8
Cifar10_ViT_MILESTONES = [7]

Cifar20_ViT_EPOCHS = 9
Cifar20_ViT_MILESTONES = [8]

Cinic10_ViT_EPOCHS = 8
Cinic10_ViT_MILESTONES = [7]

TinyImageNet_ViT_EPOCHS = 30
TinyImageNet_ViT_MILESTONES = [20, 27]

Cifar19_ViT_EPOCHS = 9
Cifar19_ViT_MILESTONES = [8]

Cifar100_ViT_EPOCHS = 8
Cifar100_ViT_MILESTONES = [7]

Imagenet64_ViT_EPOCHS = 8
Imagenet64_ViT_MILESTONES = [7]

GTSRB_ViT_EPOCHS = 15
GTSRB_ViT_MILESTONES = [12]

DATE_FORMAT = "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
# time of script run
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# log dir
LOG_DIR = "runs"

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
