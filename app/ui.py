from tkinter.ttk import Label

import customtkinter as ctk


def create_window(
        master,
        window_width,
        window_height,
        is_width_resizable=False,
        is_height_resizable=False,
):
    master.title("Система распознавания лиц")
    master.resizable(is_width_resizable, is_height_resizable)
    screen_width = master.winfo_screenwidth()
    screen_height = master.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    master.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")


def get_web_cam_label(master):
    web_cam_label = Label(master)
    web_cam_label.grid(row=0, column=0, columnspan=2)
    return web_cam_label


def get_status_label(master, text):
    verification_label = ctk.CTkLabel(
        master,
        text=text,
        font=ctk.CTkFont(size=16, weight="bold"),
    )
    verification_label.grid(row=1, column=0, padx=20, columnspan=2, pady=(20, 10))
    return verification_label


def display_verification_button(master, command):
    verification_button = ctk.CTkButton(
        master,
        text="Подтвердить",
        command=command,
        width=150,
        height=30,
        font=ctk.CTkFont(size=16, weight="bold"),
    )
    verification_button.grid(row=3, column=1, columnspan=1, padx=5, pady=10)


def display_exit_button(master, command):
    exit_button = ctk.CTkButton(
        master,
        text="Выйти",
        command=command,
        width=150,
        height=30,
        font=ctk.CTkFont(size=16, weight="bold"),
        fg_color="brown2",
        hover_color="brown4",
    )
    exit_button.grid(row=3, column=0, columnspan=1, padx=5, pady=10)


def display_progress_bar(master):
    master.progress_bar.start()
    master.progress_bar.grid(row=2, column=0, padx=20, columnspan=2, pady=(10, 15))


def hide_progress_bar(master):
    master.progress_bar.stop()
    master.progress_bar.grid_forget()


def get_info_label(master, text):
    info_label = ctk.CTkLabel(
        master,
        text=text,
        font=ctk.CTkFont(size=16, weight="bold"),
    )
    info_label.grid(row=2, column=0, padx=20, columnspan=2, pady=(5, 10))
    return info_label
