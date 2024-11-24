import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions as decode_predictions_mobilenetv2
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load MobileNetV2 model
mobilenetv2_model = MobileNetV2(weights='imagenet')

# Basic categories (subset of ImageNet classes for demo purposes)
BASIC_CATEGORIES = {
    'animals': ['dog', 'cat', 'horse', 'elephant', 'giraffe', 'lion'],
    'vehicles': ['car', 'train', 'airplane', 'bus', 'bicycle'],
    'objects': ['tv', 'laptop', 'mouse', 'phone', 'book', 'knife', 'chair']
}

class ImageClassifierApp:
    def __init__(self, root, window_title="Image Classification App", window_size="1000x800", bg_color="#F2F2F2"):
        self.root = root
        self.root.title(window_title)
        self.root.geometry(window_size)
        self.root.config(bg=bg_color)

        # Customizations for the app's appearance
        self.bg_color = bg_color
        self.font_style = ("Helvetica", 12)
        self.button_color = "#FF6F61"
        self.image_width = 300  # Width of the displayed image

        self.img_path = None
        self.confidence_threshold = 0.5  # Set the threshold for minimum confidence for a valid prediction

        # Initialize the UI components
        self.create_widgets()

    def create_widgets(self):
        """Create the layout of the app, including buttons, image display area, and results."""
        
        # Frame for buttons and options
        button_frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        button_frame.pack(pady=20, side="left", fill="y")

        # Image selection button
        self.select_image_button = tk.Button(button_frame, text="Select Image", command=self.select_image, 
                                             bg=self.button_color, fg="white", font=self.font_style, relief="solid", width=15)
        self.select_image_button.grid(row=0, column=0, padx=10, pady=5)

        # Clear button
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear, 
                                      bg=self.button_color, fg="white", font=self.font_style, relief="solid", width=15)
        self.clear_button.grid(row=1, column=0, padx=10, pady=5)

        # Reset button
        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset, 
                                      bg=self.button_color, fg="white", font=self.font_style, relief="solid", width=15)
        self.reset_button.grid(row=2, column=0, padx=10, pady=5)

        # Result display area (label and scrollable frame)
        self.result_label_frame = tk.Frame(self.root, bg=self.bg_color)
        self.result_label_frame.pack(pady=20, fill="both", expand=True)

        self.result_label_canvas = tk.Canvas(self.result_label_frame)
        self.result_label_canvas.pack(side="left", fill="both", expand=True)

        self.result_label_scroll = tk.Scrollbar(self.result_label_frame, orient="vertical", command=self.result_label_canvas.yview)
        self.result_label_scroll.pack(side="right", fill="y")

        self.result_label_canvas.config(yscrollcommand=self.result_label_scroll.set)

        self.result_label_content = tk.Frame(self.result_label_canvas, bg=self.bg_color)
        self.result_label_canvas.create_window((0, 0), window=self.result_label_content, anchor="nw")

        # Display result text area
        self.result_text = tk.Label(self.result_label_content, text="", wraplength=600, bg=self.bg_color, font=self.font_style, justify="left")
        self.result_text.grid(row=0, column=0, padx=20, pady=10)

        # Frame for image display and prediction chart (side-by-side)
        self.image_frame = tk.Frame(self.root, bg=self.bg_color)
        self.image_frame.pack(pady=20, side="top", fill="x")

        self.image_display_frame = tk.Frame(self.image_frame, bg=self.bg_color, width=self.image_width, height=300)
        self.image_display_frame.pack(side="left", padx=20)

        self.prediction_frame = tk.Frame(self.image_frame, bg=self.bg_color, width=300, height=300)
        self.prediction_frame.pack(side="left", padx=20)

        # Prediction label
        self.prediction_label = tk.Label(self.prediction_frame, text="Predictions", bg=self.bg_color, font=("Helvetica", 14, "bold"))
        self.prediction_label.pack(pady=10)

        # Image dimensions label
        self.image_dimensions_label = tk.Label(self.image_frame, text="Image: N/A", bg=self.bg_color, font=("Helvetica", 10))
        self.image_dimensions_label.pack(side="top", padx=10)

        # Progress label
        self.progress_label = tk.Label(self.root, text="Classifying Image... Please Wait.", bg=self.bg_color, font=("Helvetica", 14))
        self.progress_label.pack(pady=10)
        self.progress_label.place_forget()  # Hide the label initially

    def load_and_preprocess_image(self, img_path):
        """Load and preprocess the image for MobileNetV2 model."""
        img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 expects 224x224 input
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = preprocess_mobilenetv2(x)  # Preprocess for MobileNetV2
        return x

    def classify_image(self, img_path):
        """Classify the image using MobileNetV2 and return the top 3 predictions."""
        processed_image = self.load_and_preprocess_image(img_path)
        preds = mobilenetv2_model.predict(processed_image)
        decoded_preds = decode_predictions_mobilenetv2(preds, top=3)[0]
        return decoded_preds

    def select_image(self):
        """Open a file dialog to select an image and classify it."""
        img_path = filedialog.askopenfilename(title="Select an Image", 
                                               filetypes=[("All Image Files", "*.*")])  # Accept all image formats
        if img_path:  # If a file was selected
            try:
                self.img_path = img_path
                self.progress_label.place(relx=0.5, rely=0.2, anchor="center")  # Show progress text
                self.root.update_idletasks()  # Force the UI to update
                predictions = self.classify_image(img_path)
                self.progress_label.place_forget()  # Hide progress text after processing
                self.display_predictions(predictions, img_path)

                # Display the input image on the GUI
                self.display_input_image(img_path)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

    def display_input_image(self, img_path):
        """Display the selected image on the Tkinter window."""
        img = Image.open(img_path)
        img_width, img_height = img.size
        self.image_dimensions_label.config(text=f"Image: {img_width} x {img_height}")

        img = img.resize((self.image_width, int(img.height * (self.image_width / img.width))))  # Resize to fit user-defined width
        img_tk = ImageTk.PhotoImage(img)

        # If there's already an image displayed, remove it first
        for widget in self.image_display_frame.winfo_children():
            widget.destroy()  # Remove the previous image if any

        # Display the image in the frame
        label = tk.Label(self.image_display_frame, image=img_tk)
        label.image = img_tk  # Keep a reference to avoid garbage collection
        label.grid(row=0, column=0, padx=20, pady=20)

    def display_predictions(self, predictions, img_path):
        """Display the predictions along with the input image and prediction bar chart."""
        prediction_text = ""
        categorized_preds = {'animals': [], 'vehicles': [], 'objects': []}

        for _, label, score in predictions:
            prediction_text += f"{label}: {score * 100:.2f}%\n"
            if score < self.confidence_threshold:
                prediction_text += f"(Confidence is below threshold of {self.confidence_threshold * 100}%)\n"
            if label in BASIC_CATEGORIES['animals']:
                categorized_preds['animals'].append((label, score))
            elif label in BASIC_CATEGORIES['vehicles']:
                categorized_preds['vehicles'].append((label, score))
            elif label in BASIC_CATEGORIES['objects']:
                categorized_preds['objects'].append((label, score))

        self.result_text.config(text=f"Predictions for {img_path}:\n{prediction_text}")
        self.display_prediction_histogram(predictions)

        # Display categorized predictions
        categorized_text = "\nCategorized Predictions:\n"
        for category, items in categorized_preds.items():
            if items:
                categorized_text += f"\n{category.capitalize()}:\n"
                for label, score in items:
                    categorized_text += f"  - {label} ({score * 100:.2f}%)\n"

        self.result_text.config(text=f"{self.result_text.cget('text')}\n{categorized_text}")

    def display_prediction_histogram(self, predictions):
        """Display a histogram for predictions."""
        labels = [label for _, label, _ in predictions]
        scores = [score for _, _, score in predictions]

        # Create a matplotlib figure and axes for the histogram
        fig, ax = plt.subplots(figsize=(4, 3))  # Adjust the size to fit well in the layout
        ax.bar(labels, scores, color="#FF6F61")
        ax.set_xlabel("Labels")
        ax.set_ylabel("Confidence")
        ax.set_title("Prediction Confidence")

        # Display the histogram in Tkinter
        chart_canvas = FigureCanvasTkAgg(fig, self.prediction_frame)
        chart_canvas.get_tk_widget().pack(pady=10)
        chart_canvas.draw()

    def clear(self):
        """Clear the image, predictions, and reset the UI."""
        self.img_path = None
        self.result_text.config(text="")
        self.image_dimensions_label.config(text="Image: N/A")
        for widget in self.image_display_frame.winfo_children():
            widget.destroy()  # Clear the image display
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()  # Clear the prediction histogram

    def reset(self):
        """Reset the entire app to the initial state."""
        self.clear()
        self.progress_label.place_forget()  # Hide progress text
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root, window_title="MobileNetV2 Image Classifier", window_size="1000x800", bg_color="#F2F2F2")
    root.mainloop()
