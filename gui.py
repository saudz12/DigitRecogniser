import tkinter as tk
from PIL import Image, ImageDraw
import torch
import numpy as np
from model import MultinomialLogisticRegression
from config import *

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, 
                               bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.recognize_btn = tk.Button(root, text="Recognize", command=self.recognize)
        self.recognize_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.clear_btn = tk.Button(root, text="Clear", command=self.clear)
        self.clear_btn.grid(row=1, column=1, padx=5, pady=5)
        
        self.result_var = tk.StringVar()
        self.result_var.set("Draw a digit and click 'Recognize'")
        self.result_label = tk.Label(root, textvariable=self.result_var, font=("Arial", 14))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.load_model()
        
        self.last_x, self.last_y = None, None
        
    def load_model(self):
        self.model = MultinomialLogisticRegression(input_size=INPUT_SIZE, k=NUM_CLASSES)
        
        try:
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
            self.model.eval()  
            print(f"Model loaded from {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            self.result_var.set(f"Error: Failed to load model")
            
    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                   fill="white", width=LINE_WIDTH, 
                                   capstyle=tk.ROUND, smooth=True)
            
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=LINE_WIDTH)
            
        self.last_x, self.last_y = x, y
    
    def clear(self):
        self.canvas.delete("all")
        
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.result_var.set("Draw a digit and click 'Recognize'")
        
        self.last_x, self.last_y = None, None
    
    def recognize(self):
        img_resized = self.image.resize((MNIST_SIZE, MNIST_SIZE), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        # img_resized.show()
        
        img_tensor = torch.FloatTensor(img_array.flatten())
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 0)
            digit = predicted.item()
        
        self.result_var.set(f"Recognized digit: {digit}")

def main():
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()