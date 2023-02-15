import os
import shutil
import random

def split_dataset(src_dir, dst_dir, train_ratio, val_ratio):
    """
    Splits the images in `src_dir` into training, validation, and test sets,
    each with their respective classes, and saves the sets to `dst_dir`.
    """
    if not os.path.exists(src_dir):
        raise Exception("Source directory does not exist")

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Create directories for training, validation, and test sets
    train_dir = os.path.join(dst_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    val_dir = os.path.join(dst_dir, "valid")
    os.makedirs(val_dir, exist_ok=True)

    test_dir = os.path.join(dst_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    # Split images into training, validation, and test sets
    for subdir, dirs, files in os.walk(src_dir):
        if subdir == src_dir:
            continue

        class_name = os.path.basename(subdir)

        # Create subdirectories for each class
        train_class_dir = os.path.join(train_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)

        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(val_class_dir, exist_ok=True)

        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(test_class_dir, exist_ok=True)

        for file in files:
            src_file = os.path.join(subdir, file)
            random_value = random.random()
            if random_value < train_ratio:
                dst_file = os.path.join(train_class_dir, file)
            elif random_value < train_ratio + val_ratio:
                dst_file = os.path.join(val_class_dir, file)
            else:
                dst_file = os.path.join(test_class_dir, file)
            shutil.copy(src_file, dst_file)

    print("Dataset split into training, validation, and test sets")


split_dataset("../datasets/CKPLUS_RMP", "../datasets/CKPLUS", 0.6, 0.2)
