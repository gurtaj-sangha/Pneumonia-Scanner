import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from config import DATA_DIR

class PneumoniaDataPipeline:
    """
    Data pipeline for pneumonia classification with proper preprocessing,
    augmentation, and data loading capabilities.
    """
    
    def __init__(self, img_size=(224, 224), batch_size=32, seed=42):
        self.data_dir = Path(DATA_DIR)
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        self.classes = ["NORMAL", "bacterial_pneumonia", "viral_pneumonia"]
        
        # Verify data directories exist
        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {split_dir}")
                
        print(f"Initialized data pipeline with image size: {self.img_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Classes: {self.classes}")
        
    def create_data_generators(self):
        """
        Create train, validation, and test data generators with appropriate
        preprocessing and augmentation.
        """
        
        # Training data augmentation for better generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values to [0,1]
            rotation_range=15,  # Random rotation up to 15 degrees
            width_shift_range=0.1,  # Random horizontal shift
            height_shift_range=0.1,  # Random vertical shift
            horizontal_flip=True,  # Random horizontal flip
            zoom_range=0.1,  # Random zoom
            brightness_range=[0.8, 1.2],  # Random brightness adjustment
            fill_mode='nearest'  # Fill strategy for transformations
        )
        
        # Validation and test data (no augmentation, only normalization)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        print("Creating data generators...")
        
        train_generator = train_datagen.flow_from_directory(
            self.data_dir / "train",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.classes,
            seed=self.seed,
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            self.data_dir / "val",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.classes,
            seed=self.seed,
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.data_dir / "test",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.classes,
            seed=self.seed,
            shuffle=False
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Test samples: {test_generator.samples}")
        print(f"Class indices: {train_generator.class_indices}")
        
        return train_generator, val_generator, test_generator
    
    def calculate_class_weights(self):
        """
        Calculate class weights to handle class imbalance during training.
        This gives more weight to underrepresented classes.
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        print("Calculating class weights...")
        
        # Count samples per class in training set
        class_counts = {}
        train_dir = self.data_dir / "train"
        
        for class_name in self.classes:
            class_dir = train_dir / class_name
            if class_dir.exists():
                # Count all image files
                count = (len(list(class_dir.glob("*.jpeg"))) + 
                        len(list(class_dir.glob("*.jpg"))) + 
                        len(list(class_dir.glob("*.png"))))
                class_counts[class_name] = count
            else:
                class_counts[class_name] = 0
        
        print("Class distribution in training set:")
        total_samples = sum(class_counts.values())
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Calculate balanced class weights
        class_weights = {}
        n_classes = len(self.classes)
        
        for i, class_name in enumerate(self.classes):
            if class_counts[class_name] > 0:
                # Inverse frequency weighting
                class_weights[i] = total_samples / (n_classes * class_counts[class_name])
            else:
                class_weights[i] = 1.0
        
        print("Calculated class weights:")
        for i, class_name in enumerate(self.classes):
            print(f"  {class_name} (index {i}): {class_weights[i]:.3f}")
        
        return class_weights
    
    def get_dataset_statistics(self):
        """
        Get comprehensive statistics about the dataset.
        """
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        splits = ["train", "val", "test"]
        total_counts = {class_name: 0 for class_name in self.classes}
        split_counts = {}
        
        for split in splits:
            print(f"\n{split.upper()} SET:")
            split_dir = self.data_dir / split
            split_counts[split] = {}
            
            if not split_dir.exists():
                print(f"  Directory not found: {split_dir}")
                continue
            
            split_total = 0
            for class_name in self.classes:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    count = (len(list(class_dir.glob("*.jpeg"))) + 
                            len(list(class_dir.glob("*.jpg"))) + 
                            len(list(class_dir.glob("*.png"))))
                    print(f"  {class_name}: {count} images")
                    split_counts[split][class_name] = count
                    total_counts[class_name] += count
                    split_total += count
                else:
                    print(f"  {class_name}: Directory not found")
                    split_counts[split][class_name] = 0
            
            print(f"  Total {split} images: {split_total}")
        
        print(f"\nTOTAL DATASET:")
        grand_total = sum(total_counts.values())
        for class_name, count in total_counts.items():
            percentage = (count / grand_total) * 100 if grand_total > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        print(f"  Total images: {grand_total}")
        
        return split_counts, total_counts
    
    def visualize_samples(self, num_samples=9):
        """
        Visualize sample images from each class for data inspection.
        """
        print("Generating sample visualizations...")
        
        # Create a simple generator for visualization
        datagen = ImageDataGenerator(rescale=1./255)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle('Sample Images from Each Class', fontsize=16)
        
        for i, class_name in enumerate(self.classes):
            class_dir = self.data_dir / "train" / class_name
            
            if class_dir.exists():
                # Get first few images from this class
                image_files = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg"))
                
                for j in range(min(3, len(image_files))):
                    if i * 3 + j < 9:
                        img = tf.keras.preprocessing.image.load_img(
                            image_files[j], 
                            target_size=self.img_size
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                        
                        ax = axes[i, j]
                        ax.imshow(img_array)
                        ax.set_title(f'{class_name}\n{image_files[j].name}', fontsize=10)
                        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Sample visualization saved as 'sample_images.png'")
    
    def preview_augmentations(self, num_augmentations=4):
        """
        Preview data augmentations on a sample image.
        """
        print("Generating augmentation preview...")
        
        # Get a sample image
        train_dir = self.data_dir / "train"
        sample_image = None
        
        for class_name in self.classes:
            class_dir = train_dir / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg"))
                if image_files:
                    sample_image = image_files[0]
                    break
        
        if sample_image is None:
            print("No sample image found for augmentation preview")
            return
        
        # Load and prepare image
        img = tf.keras.preprocessing.image.load_img(sample_image, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Create augmentation generator
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Generate augmented images
        fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(15, 3))
        fig.suptitle('Data Augmentation Preview', fontsize=16)
        
        # Original image
        axes[0].imshow(img_array[0] / 255.0)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Augmented images
        aug_iter = datagen.flow(img_array, batch_size=1)
        for i in range(num_augmentations):
            aug_img = next(aug_iter)
            axes[i + 1].imshow(aug_img[0])
            axes[i + 1].set_title(f'Augmented {i + 1}')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('augmentation_preview.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Augmentation preview saved as 'augmentation_preview.png'")


def main():
    """
    Main function to test the data pipeline.
    """
    print("Testing Pneumonia Data Pipeline")
    print("="*50)
    
    # Initialize pipeline
    pipeline = PneumoniaDataPipeline(batch_size=16)
    
    # Get dataset statistics
    pipeline.get_dataset_statistics()
    
    # Calculate class weights
    class_weights = pipeline.calculate_class_weights()
    
    # Create data generators
    train_gen, val_gen, test_gen = pipeline.create_data_generators()
    
    # Visualize samples
    pipeline.visualize_samples()
    
    # Preview augmentations
    pipeline.preview_augmentations()
    
    print("\nData pipeline setup complete!")
    return pipeline, train_gen, val_gen, test_gen, class_weights


if __name__ == "__main__":
    main()
