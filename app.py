import tkinter as tk
import customtkinter as ctk
from tkinter import ttk
from PIL import Image, ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import threading
import numpy as np
import subprocess  # Added to run UGS
import pyautogui  # Added to automate UGS
import time
import cv2
import pyvips
import svgwrite

app = tk.Tk()
app.geometry("532x800")
app.title("Text to Image")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512, text="")
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

progress_bar = ttk.Progressbar(app, mode='determinate', length=500)
progress_bar.place(x=16, y=632)

message_label = tk.Label(app, text="", font=("Arial", 16))
message_label.place(x=10, y=710)

def generate_image():
    with autocast(device):
        image = pipe(str(prompt.get())).images[0]
    image.save('generatedimage.png')
    img = ctk.CTkImage(image, size=(512, 512))
    lmain.configure(image=img)

def update_progress():
    progress = progress_bar["value"]
    if progress < 100:
        progress += 1
        progress_bar["value"] = progress
        app.after(100, update_progress)
def generate_with_progress():
    progress_bar['value'] = 0
    thread = threading.Thread(target=generate_image)
    thread.start()
    app.after(100, update_progress)

generate_button = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate_with_progress)
generate_button.configure(text='Generate')
generate_button.place(x=206, y=60)

def generate_laser_gcode():
    try:
        input_image = cv2.imread('generatedimage.png', cv2.IMREAD_COLOR)

        feed_rate = 1000 
        laser_power_max = 255 
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        svg_file = svgwrite.Drawing('output_image.svg', profile='tiny', size=(input_image.shape[1], input_image.shape[0]))

        with open('output.gcode', 'w') as gcode_file:
            gcode_file.write('; G-code for Laser Engraving\n')
            gcode_file.write('G21 ; Set units to millimeters\n')
            gcode_file.write('G90 ; Set to absolute positioning\n')

            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    for point in contour:
                        x, y = point[0]
                        gcode_file.write(f'G1 X{x/10} Y{y/10} F{feed_rate}\n')
                        gcode_file.write(f'M106 S{laser_power_max}\n')
                        gcode_file.write(f'G4 P50\n') 
            gcode_file.write('M107 ; Turn off the laser\n')
            gcode_file.write('G0 X0 Y0 ; Return to home position\n')

        svg_file.save()

        message_label.config(text="G-code has been generated")

        print("G-code generation for laser engraving complete. Check 'output.gcode' for the result.")
    except Exception as e:
        message_label.config(text="Error generating G-code")
        print("Error generating G-code:", e)


app.mainloop()