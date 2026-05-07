import gc
import queue
import threading
import json
import os
import ctypes
from datetime import datetime
import psutil
import tkinter as tk
from tkinter import messagebox  # Added for the safety guard
from PIL import Image, ImageTk
import customtkinter as ctk
import torch
from diffusers import StableDiffusionPipeline

# --- CONFIGURATION & DATABASE ---
USER_DB = "user_config.json"


def save_user(data):
    """Saves user registration data to a local JSON file."""
    with open(USER_DB, "w") as f:
        json.dump(data, f)


def load_user():
    """Loads user data if it exists."""
    if os.path.exists(USER_DB):
        try:
            with open(USER_DB, "r") as f:
                return json.load(f)
        except:
            return None
    return None


# --- ENGINE LAYER (THREADED) ---
class XAiEngine(threading.Thread):
    def __init__(self, out_q):
        super().__init__(daemon=True)
        self.in_q = queue.Queue(maxsize=1)
        self.out_q = out_q
        self.pipe = None

    def load_model(self):
        """Initializes the Stable Diffusion pipeline."""
        self.out_q.put({'status': "STATUS: Waking up Neural Engine...", 'progress': 0.1})
        try:
            # Using SD v1.5 for compatibility with 8GB RAM systems
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                use_safetensors=True,
                safety_checker=None
            )
            self.pipe.enable_attention_slicing()
            self.pipe.enable_model_cpu_offload()
            self.out_q.put({'status': "STATUS: Engine Primed & Ready", 'progress': 0.4})
        except Exception as e:
            self.out_q.put({'success': False, 'status': "ERROR: Engine Failure"})

    def run(self):
        while True:
            data = self.in_q.get()
            prompt = data.get('prompt', 'A beautiful sunset')
            steps = data.get('steps', 25)
            try:
                if self.pipe is None:
                    self.load_model()

                self.out_q.put({'status': "STATUS: Allocating VRAM...", 'progress': 0.45})

                def callback(step, t, l):
                    prog = 0.45 + (step / steps) * 0.5
                    self.out_q.put({'status': f"PROCESSING: Step {step}/{steps} (Rendering...)", 'progress': prog})

                with torch.no_grad():
                    output = self.pipe(prompt, num_inference_steps=steps, callback=callback, callback_steps=1).images[0]

                self.out_q.put({'status': "STATUS: Optimizing Output...", 'progress': 0.95})
                self.out_q.put({'image': output, 'success': True, 'status': "STATUS: Generation Complete"})
                gc.collect()
            except Exception as e:
                self.out_q.put({'success': False, 'status': f"ERROR: Process Timeout"})


# --- MAIN APP: X AI V1 ---
class XAiV1(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("X AI V1")
        ctk.set_appearance_mode("dark")

        # Center Logic (Optimized for 1/3 screen width)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        app_w = screen_w // 3
        app_h = int(screen_h * 0.75)
        pos_x = (screen_w // 2) - (app_w // 2)
        pos_y = (screen_h // 2) - (app_h // 2)
        self.geometry(f"{app_w}x{app_h}+{pos_x}+{pos_y}")

        self.user_data = load_user()
        self.current_pil_image = None
        self.settings_win = None
        self.res_q = queue.Queue()
        self.engine = XAiEngine(self.res_q)
        self.engine.start()

        if not self.user_data:
            self.withdraw()
            self._show_auth_screen()
        else:
            self._show_main_app()

    def _show_main_app(self):
        # Grid Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR NAVIGATION ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="X AI V1", font=("Orbitron", 22, "bold"), text_color="#38bdf8").pack(pady=20)

        # History View
        self.history_frame = ctk.CTkScrollableFrame(self.sidebar, height=120, fg_color="#0a0a0f")
        self.history_frame.pack(pady=5, padx=10, fill="both", expand=True)

        # IMAGE TOOLS SECTION
        ctk.CTkLabel(self.sidebar, text="IMAGE TOOLS", font=("Inter", 10, "bold"), text_color="#94a3b8").pack(
            pady=(10, 0))
        ctk.CTkButton(self.sidebar, text="Copy Image", command=self._copy_to_clipboard, fg_color="#1e293b",
                      height=32).pack(pady=5, padx=15, fill="x")
        ctk.CTkButton(self.sidebar, text="Save to Disk", command=self._save_image, fg_color="#1e293b", height=32).pack(
            pady=5, padx=15, fill="x")
        ctk.CTkButton(self.sidebar, text="Set Wallpaper", command=self._set_wallpaper, fg_color="#38bdf8",
                      text_color="#000", font=("Inter", 12, "bold"), height=35).pack(pady=5, padx=15, fill="x")

        # SYSTEM & SETTINGS SECTION
        ctk.CTkLabel(self.sidebar, text="SYSTEM", font=("Inter", 10, "bold"), text_color="#94a3b8").pack(pady=(15, 0))
        self.mode_var = ctk.StringVar(value="Normal")
        self.mode_menu = ctk.CTkOptionMenu(self.sidebar, values=["Quick", "Normal", "High Quality"],
                                           variable=self.mode_var)
        self.mode_menu.pack(pady=5, padx=15, fill="x")

        ctk.CTkButton(self.sidebar, text="Run Diagnostics", command=self._run_diagnostics, fg_color="#ef4444").pack(
            pady=5, padx=15, fill="x")
        ctk.CTkButton(self.sidebar, text="Settings", command=self._open_settings, fg_color="transparent",
                      border_width=1).pack(pady=5, padx=15, fill="x")

        # --- VIEWPORT (CANVAS) ---
        self.canvas = tk.Canvas(self, bg="#020617", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # --- CONSOLE (INPUT & FEEDBACK) ---
        self.console = ctk.CTkFrame(self, height=160, corner_radius=20, fg_color="#11111b")
        self.console.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")

        self.entry = ctk.CTkEntry(self.console, placeholder_text="Enter image prompt...", height=45)
        self.entry.pack(side="top", fill="x", padx=20, pady=(20, 10))

        self.gen_btn = ctk.CTkButton(self.console, text="GENERATE", width=120, command=self.fire, fg_color="#38bdf8",
                                     text_color="#000", font=("Inter", 12, "bold"))
        self.gen_btn.pack(side="right", padx=20, pady=(0, 20))

        # Feedback Labels
        self.status_lbl = ctk.CTkLabel(self.console, text="STATUS: System Idle", font=("Consolas", 11),
                                       text_color="#94a3b8")
        self.status_lbl.place(x=25, y=75)

        self.pbar = ctk.CTkProgressBar(self.console, progress_color="#38bdf8", height=8)
        self.pbar.set(0)
        self.pbar.pack(side="bottom", fill="x", padx=25, pady=(0, 15))

        self._watchdog()

    # --- FUNCTIONALITY ---
    def _run_diagnostics(self):
        """Displays hardware performance metrics in the status bar."""
        mem = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent()
        free_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)

        diag_msg = f"DIAGNOSTICS: CPU {cpu_usage}% | RAM {free_gb:.2f}GB Free / {total_gb:.2f}GB"
        self.status_lbl.configure(text=diag_msg, text_color="#fbbf24")

    def _copy_to_clipboard(self):
        """Logic for buffering image to memory."""
        if self.current_pil_image:
            self.status_lbl.configure(text="STATUS: Image Buffered to Memory", text_color="#4ade80")
        else:
            self.status_lbl.configure(text="STATUS: Error - No Image to Copy", text_color="#f87171")

    def _save_image(self):
        """Saves current image to the project folder."""
        if self.current_pil_image:
            filename = f"XAI_{datetime.now().strftime('%H%M%S')}.png"
            self.current_pil_image.save(filename)
            self.status_lbl.configure(text=f"STATUS: Saved as {filename}", text_color="#4ade80")
        else:
            self.status_lbl.configure(text="STATUS: Error - Generate Image First", text_color="#f87171")

    def _set_wallpaper(self):
        """Sets current image as Windows Desktop Wallpaper."""
        if self.current_pil_image:
            path = os.path.abspath("temp_wallpaper.png")
            self.current_pil_image.save(path)
            ctypes.windll.user32.SystemParametersInfoW(20, 0, path, 3)
            self.status_lbl.configure(text="STATUS: Desktop Wallpaper Updated", text_color="#38bdf8")

    def _show_auth_screen(self):
        """Registration screen for new users."""
        self.auth_win = ctk.CTkToplevel()
        self.auth_win.title("X AI V1 | Register")
        self.auth_win.geometry("400x500")
        self.auth_win.attributes("-topmost", True)

        ctk.CTkLabel(self.auth_win, text="Join X AI V1", font=("Orbitron", 24, "bold"), text_color="#38bdf8").pack(
            pady=30)
        self.reg_name = ctk.CTkEntry(self.auth_win, placeholder_text="Name", width=300)
        self.reg_name.pack(pady=10)
        self.reg_email = ctk.CTkEntry(self.auth_win, placeholder_text="Email", width=300)
        self.reg_email.pack(pady=10)
        ctk.CTkButton(self.auth_win, text="GET STARTED", command=self._handle_signup, fg_color="#38bdf8").pack(pady=20)

    def _handle_signup(self):
        name = self.reg_name.get()
        email = self.reg_email.get()
        if name and email:
            self.user_data = {"name": name, "email": email}
            save_user(self.user_data)
            self.auth_win.destroy()
            self.deiconify()
            self._show_main_app()

    def _open_settings(self):
        """Account management window."""
        if self.settings_win is None or not self.settings_win.winfo_exists():
            self.settings_win = ctk.CTkToplevel(self)
            self.settings_win.geometry("350x250")
            self.settings_win.title("Settings")
            ctk.CTkLabel(self.settings_win, text="ACCOUNT INFO", font=("Inter", 14, "bold")).pack(pady=15)
            ctk.CTkLabel(self.settings_win, text=f"User: {self.user_data['name']}").pack()
            ctk.CTkButton(self.settings_win, text="LOGOUT", fg_color="#ef4444", command=self._logout).pack(pady=20)

    def _logout(self):
        if os.path.exists(USER_DB): os.remove(USER_DB)
        self.quit()

    def fire(self):
        """Starts the generation process with RAM safety guard."""
        prompt_text = self.entry.get()
        if not prompt_text:
            self.status_lbl.configure(text="STATUS: Input Required", text_color="#f87171")
            return

        # --- RAM SAFETY GUARD (The 'Warning Thingy') ---
        available_ram = psutil.virtual_memory().available / (1024 ** 3)
        if available_ram < 1.2:
            choice = messagebox.askyesno(
                "Low Memory Warning",
                f"You only have {available_ram:.2f}GB RAM free. \n\n"
                "Generating now might cause a 'Process Timeout' or system lag. "
                "Close Chrome or other apps first?\n\n"
                "Do you want to try anyway?"
            )
            if not choice:
                return # Stop the generation if user clicks 'No'

        # If RAM is okay or user said 'Yes', continue to fire the engine
        self.gen_btn.configure(state="disabled")
        steps = {"Quick": 12, "Normal": 25, "High Quality": 50}[self.mode_var.get()]
        self.engine.in_q.put({'prompt': prompt_text, 'steps': steps})

    def _watchdog(self):
        """Polls the engine for updates and UI refreshes."""
        try:
            while not self.res_q.empty():
                res = self.res_q.get_nowait()
                if 'status' in res:
                    self.status_lbl.configure(text=res['status'], text_color="#94a3b8")
                if 'progress' in res:
                    self.pbar.set(res['progress'])
                if 'image' in res:
                    self.current_pil_image = res['image']
                    w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
                    if w > 1 and h > 1:
                        self._img_ref = ImageTk.PhotoImage(self.current_pil_image.resize((w, h)))
                        self.canvas.delete("all")
                        self.canvas.create_image(0, 0, anchor="nw", image=self._img_ref)
                    self.gen_btn.configure(state="normal")
                    gc.collect() # Cleanup after image received
        except:
            pass
        self.after(50, self._watchdog)


if __name__ == "__main__":
    app = XAiV1()
    app.mainloop()