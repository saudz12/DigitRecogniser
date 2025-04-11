import subprocess
import sys
import os

def install_requirements():
    print("Installing required packages...")
    
    required_packages = [
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "pandas",
        "tqdm"  
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        print("Error: pip is not installed or not working properly.")
        print("Please install pip and try again.")
        sys.exit(1)
    
    for package in required_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Error: Failed to install {package}")
            sys.exit(1)
    
    print("\nAll required packages have been installed successfully!")
    print("You can now run the training script with: python train.py")
    print("After training, run the GUI with: python gui.py")

def check_data_directory():
    if not os.path.exists("./data"):
        print("Creating data directory...")
        os.makedirs("./data")
        print("Please place the MNIST CSV files in the './data/' directory:")
        print("  - mnist_train.csv")
        print("  - mnist_test.csv")
    else:
        train_path = "./data/mnist_train.csv"
        test_path = "./data/mnist_test.csv"
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("Warning: MNIST data files not found.")
            print("Please ensure the following files are in the './data/' directory:")
            print("  - mnist_train.csv")
            print("  - mnist_test.csv")

if __name__ == "__main__":
    print("Setting up MNIST Digit Recognizer application...")
    install_requirements()
    check_data_directory()
    print("\nSetup complete!")