import tkinter as tk
from tkinter import filedialog, ttk, Frame, Label, Button, BOTTOM, TOP, LEFT, RIGHT, X, Y, BOTH, END
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import load
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import json
import os
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Model Loading ---
try:
    base_model = InceptionV3(weights='imagenet')
    vgg_model = Model(base_model.input, base_model.layers[-2].output)
    model = load_model('new-model-1.h5')

    # Load tokenizer files
    with open("wordtoix.pkl", "rb") as f:
        wordtoix = load(f)
    with open("ixtoword.pkl", "rb") as f:
        ixtoword = load(f)
    max_length = 74

except FileNotFoundError as e:
    print(f"Error loading model or pickle file: {e}")
    print("Please ensure 'new-model-1.h5', 'wordtoix.pkl', and 'ixtoword.pkl' are in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    exit()

# --- Image Preprocessing and Encoding ---
def preprocess_img(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image_path):
    image = preprocess_img(image_path)
    vec = vgg_model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

# --- Caption Generation Functions ---
def greedy_search(pic_features):
    start = 'startseq'
    for _ in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([pic_features, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword.get(yhat, None)
        if word is None or word == 'endseq':
            break
        start += ' ' + word
    final = start.split()
    final = final[1:] if len(final) > 1 else [] # Only remove 'startseq'
    return ' '.join(final)

def beam_search(pic_features, beam_index=3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            preds = model.predict([pic_features, par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += np.log(preds[0][w] + 1e-9)
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, key=lambda l: l[1], reverse=True)
        start_word = start_word[:beam_index]

    best_seq_indices = start_word[0][0]
    intermediate_caption = [ixtoword.get(i, '') for i in best_seq_indices]

    final_caption = []
    for i in intermediate_caption:
        if i == 'endseq':
            break
        if i != 'startseq':
            final_caption.append(i)

    return ' '.join(final_caption)

# --- Evaluation Functions ---
def load_test_references(test_references_path='test_references.json'):
    """
    Load reference captions for test images from a JSON file.
    """
    try:
        if os.path.exists(test_references_path):
            with open(test_references_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Reference file {test_references_path} not found.")
            return {}
    except Exception as e:
        print(f"Error loading reference captions: {e}")
        return {}

def evaluate_caption(predicted_caption, reference_captions):
    """
    Calculate BLEU scores for a single caption compared to its references.
    """
    if not reference_captions:
        return None
    
    # Tokenize the captions (split into words)
    tokenized_references = [caption.lower().split() for caption in reference_captions]
    tokenized_prediction = predicted_caption.lower().split()
    
    # Calculate BLEU scores
    bleu1 = sentence_bleu(tokenized_references, tokenized_prediction, weights=(1.0, 0, 0, 0))
    bleu2 = sentence_bleu(tokenized_references, tokenized_prediction, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(tokenized_references, tokenized_prediction, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(tokenized_references, tokenized_prediction, weights=(0.25, 0.25, 0.25, 0.25))
    
    return {
        'bleu1': bleu1 * 100,
        'bleu2': bleu2 * 100,
        'bleu3': bleu3 * 100,
        'bleu4': bleu4 * 100
    }

# --- GUI Setup ---
BG_COLOR = "#F0F0F0"
TITLE_COLOR = "#FF6347"
BUTTON_COLOR = "#4682B4"
BUTTON_TEXT_COLOR = "white"
TEXT_COLOR = "#333333"

top = tk.Tk()
top.geometry('900x700')
top.title('Image Caption Generator')
top.configure(background=BG_COLOR)

# --- Style Configuration ---
style = ttk.Style(top)
style.theme_use('clam')

style.configure('TFrame', background=BG_COLOR)
style.configure('TButton', background=BUTTON_COLOR, foreground=BUTTON_TEXT_COLOR, font=('Helvetica', 12), padding=6)
style.map('TButton', background=[('active', '#5A9BD5')])
style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 12))
style.configure('Title.TLabel', background=BG_COLOR, foreground=TITLE_COLOR, font=('Helvetica', 24, 'bold'))
style.configure('Caption.TLabel', background=BG_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 14), wraplength=400)

# --- Main Frame ---
main_frame = ttk.Frame(top, padding="20 20 20 20")
main_frame.pack(expand=True, fill=BOTH)

# --- Title ---
heading = ttk.Label(main_frame, text="Image Caption Generator", style='Title.TLabel')
heading.pack(pady=(0, 20))

# --- Image Display Area ---
image_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=1)
image_frame.pack(pady=10, padx=20, fill=BOTH, expand=True)

sign_image_label = ttk.Label(image_frame, text="Image will appear here", anchor='center')
sign_image_label.pack(fill=BOTH, expand=True, padx=10, pady=10)

# --- Results Area ---
results_frame = ttk.Frame(main_frame)
results_frame.pack(pady=(10, 10), fill=X, padx=20)

greedy_label = ttk.Label(results_frame, text="Greedy Search:", style='Caption.TLabel', anchor='w')
greedy_label.pack(fill=X, pady=2)

beam3_label = ttk.Label(results_frame, text="Beam Search (k=3):", style='Caption.TLabel', anchor='w')
beam3_label.pack(fill=X, pady=2)

beam5_label = ttk.Label(results_frame, text="Beam Search (k=5):", style='Caption.TLabel', anchor='w')
beam5_label.pack(fill=X, pady=2)

# --- Evaluation Results Frame ---
eval_frame = ttk.Frame(main_frame)
eval_frame.pack(pady=(10, 10), fill=X, padx=20)

eval_heading = ttk.Label(eval_frame, text="Evaluation Metrics (when references available)", 
                         font=('Helvetica', 12, 'bold'))
eval_heading.pack(fill=X, pady=(5, 10))

greedy_bleu_label = ttk.Label(eval_frame, text="Greedy Search BLEU: N/A", anchor='w')
greedy_bleu_label.pack(fill=X, pady=2)

beam3_bleu_label = ttk.Label(eval_frame, text="Beam Search (k=3) BLEU: N/A", anchor='w')
beam3_bleu_label.pack(fill=X, pady=2)

beam5_bleu_label = ttk.Label(eval_frame, text="Beam Search (k=5) BLEU: N/A", anchor='w')
beam5_bleu_label.pack(fill=X, pady=2)

# --- Control Buttons Frame ---
button_frame = ttk.Frame(main_frame)
button_frame.pack(pady=(10, 0), fill=X, padx=20)

# Function to toggle evaluation display
def toggle_eval_frame():
    if eval_frame.winfo_viewable():
        eval_frame.pack_forget()
        toggle_eval_btn.configure(text="Show Evaluation")
    else:
        eval_frame.pack(pady=(10, 10), fill=X, padx=20, after=results_frame)
        toggle_eval_btn.configure(text="Hide Evaluation")

toggle_eval_btn = ttk.Button(button_frame, text="Show Evaluation", command=toggle_eval_frame)
toggle_eval_btn.pack(side=LEFT)

classify_button = ttk.Button(button_frame, text="Generate Captions", state=tk.DISABLED)
classify_button.pack(side=RIGHT, padx=(10, 0))

upload_button = ttk.Button(button_frame, text="Upload Image")
upload_button.pack(side=RIGHT)

# Initially hide the evaluation frame
eval_frame.pack_forget()

# --- Functions ---
current_file_path = None

def generate_captions():
    global current_file_path
    if not current_file_path:
        return

    try:
        upload_button.config(state=tk.DISABLED)
        classify_button.config(state=tk.DISABLED)
        greedy_label.configure(text="Greedy Search: Generating...")
        beam3_label.configure(text="Beam Search (k=3): Generating...")
        beam5_label.configure(text="Beam Search (k=5): Generating...")
        top.update_idletasks()

        encoded_features = encode(current_file_path).reshape(1, 2048)

        # Generate captions
        pred_greedy = greedy_search(encoded_features)
        print("Greedy:", pred_greedy)
        greedy_label.configure(text=f'Greedy Search: {pred_greedy}')

        pred_beam3 = beam_search(encoded_features, beam_index=3)
        print("Beam_3:", pred_beam3)
        beam3_label.configure(text=f'Beam Search (k=3): {pred_beam3}')

        pred_beam5 = beam_search(encoded_features, beam_index=5)
        print("Beam_5:", pred_beam5)
        beam5_label.configure(text=f'Beam Search (k=5): {pred_beam5}')

        # Evaluate captions if references are available
        references = load_test_references()
        if references:
            image_filename = os.path.basename(current_file_path)
            if image_filename in references:
                ref_captions = references[image_filename]
                
                # Evaluate and display metrics
                greedy_bleu = evaluate_caption(pred_greedy, ref_captions)
                beam3_bleu = evaluate_caption(pred_beam3, ref_captions)
                beam5_bleu = evaluate_caption(pred_beam5, ref_captions)
                
                if greedy_bleu:
                    greedy_bleu_label.configure(
                        text=f"Greedy Search BLEU: BLEU-1: {greedy_bleu['bleu1']:.2f}, BLEU-4: {greedy_bleu['bleu4']:.2f}")
                
                if beam3_bleu:
                    beam3_bleu_label.configure(
                        text=f"Beam Search (k=3) BLEU: BLEU-1: {beam3_bleu['bleu1']:.2f}, BLEU-4: {beam3_bleu['bleu4']:.2f}")
                
                if beam5_bleu:
                    beam5_bleu_label.configure(
                        text=f"Beam Search (k=5) BLEU: BLEU-1: {beam5_bleu['bleu1']:.2f}, BLEU-4: {beam5_bleu['bleu4']:.2f}")
                
                # Make the evaluation frame visible if we have metrics
                if not eval_frame.winfo_viewable():
                    toggle_eval_frame()
            else:
                # Reset labels if no references for this image
                greedy_bleu_label.configure(text="Greedy Search BLEU: No references available")
                beam3_bleu_label.configure(text="Beam Search (k=3) BLEU: No references available")
                beam5_bleu_label.configure(text="Beam Search (k=5) BLEU: No references available")

    except Exception as e:
        print(f"Error during caption generation: {e}")
        greedy_label.configure(text="Greedy Search: Error")
        beam3_label.configure(text="Beam Search (k=3): Error")
        beam5_label.configure(text="Beam Search (k=5): Error")
    finally:
        upload_button.config(state=tk.NORMAL)
        classify_button.config(state=tk.NORMAL)

def upload_image_action():
    global current_file_path
    try:
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            return

        current_file_path = file_path

        greedy_label.configure(text="Greedy Search:")
        beam3_label.configure(text="Beam Search (k=3):")
        beam5_label.configure(text="Beam Search (k=5):")
        
        # Reset BLEU score labels
        greedy_bleu_label.configure(text="Greedy Search BLEU: N/A")
        beam3_bleu_label.configure(text="Beam Search (k=3) BLEU: N/A")
        beam5_bleu_label.configure(text="Beam Search (k=5) BLEU: N/A")

        img = Image.open(file_path)

        img_w, img_h = img.size
        frame_w = image_frame.winfo_width() - 20
        frame_h = image_frame.winfo_height() - 20
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = 400, 300

        ratio = min(frame_w / img_w, frame_h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        img.thumbnail((new_w, new_h))
        im = ImageTk.PhotoImage(img)

        sign_image_label.configure(image=im, text="")
        sign_image_label.image = im

        classify_button.config(state=tk.NORMAL)

    except FileNotFoundError:
        print("Error: Selected file not found.")
        current_file_path = None
        classify_button.config(state=tk.DISABLED)
    except Exception as e:
        print(f"Error loading image: {e}")
        sign_image_label.configure(text="Error loading image", image=None)
        sign_image_label.image = None
        current_file_path = None
        classify_button.config(state=tk.DISABLED)

upload_button.configure(command=upload_image_action)
classify_button.configure(command=generate_captions)

top.mainloop()