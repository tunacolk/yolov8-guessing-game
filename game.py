import random
import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# YOLO model yükle
model = YOLO("best.pt")

# sözlük oluştur
hints = {
    "Pen": "Düşünceleri kağıda dökmenin aracı, yazı yazmada olmazsa olmazdır.",
    "Door": "Her yeni başlangıcın anahtarıdır; içeri girmek ya da çıkmak için açarsın.",
    "Plastic Bottles": "Genellikle su veya içecek taşır, geri dönüşüm için uygundur.",
    "Book": "Bilgiyi keşfetmenin sessiz bir yolu, sayfaları çevirdikçe dünyalar açılır.",
    "Backpack": "Genellikle seyahatlerde veya okulda kullanılır, sırtında taşırsın.",
    "Notebook ": "Fikirlerini, çizimlerini veya planlarını saklamak için kullanırsın. Sayfaları genellikle beyazdır.",
    "Mobile Phone": "İletişimin ve bilgiye ulaşmanın küçük ama güçlü bir aracı, genelde cebinde taşınır."
}

# Kameradan görüntü al
cap = cv2.VideoCapture(0)  # Kamera açılır

# klasörler oluştur
output_folder = "detected_objects"
output_folder_original = "detected_objects_original"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_original, exist_ok=True)

# nesneler için liste oluştur
recognized_objects = []

# Fotoğraf bankası için liste
photo_bank = []

# Nesneleri tanıma ve fotoğraf toplama
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    detections = results[0].boxes
    if detections is not None:
        for box in detections:
            cls = int(box.cls[0])  # Sınıf indeksini al
            label = model.names[cls]  # Sınıf ismi
            conf = box.conf[0]  # Güven skoru

            if conf > 0.5:
                if label not in recognized_objects:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])# Koordinatları al


                    original_output_path = os.path.join(output_folder_original, f"{label}_{random.randint(1000, 9999)}.jpg")
                    roi = frame[y1:y2, x1:x2]
                    cv2.imwrite(original_output_path, roi)# Orijinal bölgeyi kaydet


                    blurred_roi = cv2.GaussianBlur(roi, (301, 301), 0)# Bölgeyi bulanıklaştır
                    frame[y1:y2, x1:x2] = blurred_roi

                    # Bulanık görüntüyü kaydet
                    output_path = os.path.join(output_folder, f"{label}_{random.randint(1000, 9999)}.jpg")
                    cv2.imwrite(output_path, frame)

                    recognized_objects.append(label)
                    photo_bank.append({
                        "label": label,
                        "blurred_path": output_path,
                        "original_path": original_output_path
                    })

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Oyun kontrol
if len(photo_bank) < 4:
    print("Oyun başlatılamıyor. En az 4 farklı nesne algılanmalı.")
    messagebox.showerror("Hata", "Oyun başlatılamıyor. En az dört farklı nesne algılanmalı.")
    exit()  # Programı sonlandır

# Oyun Arayüzü
class GameApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Nesne Tanıma Oyunu")

        self.question_label = tk.Label(master, text="Nesneyi Tanıyın!", font=("Arial", 14))
        self.question_label.pack(pady=20)

        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

        self.hint_label = tk.Label(master, text="", font=("Arial", 12), wraplength=400)
        self.hint_label.pack(pady=10)

        self.buttons_frame = tk.Frame(master)
        self.buttons_frame.pack()

        self.correct_label = None
        self.correct_image_path = None
        self.correct_original_path = None
        self.start_game()

    def start_game(self):
        # rastgele bir nesne seç
        correct_choice = random.choice(photo_bank)
        self.correct_label = correct_choice["label"]
        self.correct_image_path = correct_choice["blurred_path"]
        self.correct_original_path = correct_choice["original_path"]

        # İpucu ekle
        hint = hints.get(self.correct_label, "Bu nesne hakkında bir ipucu yok.")
        self.hint_label.config(text=hint)

        # Diğer 3 rastgele nesneyi seç
        options = [self.correct_label]
        while len(options) < 4:
            random_choice = random.choice(photo_bank)["label"]
            if random_choice != self.correct_label and random_choice not in options:
                options.append(random_choice)

        random.shuffle(options)
        self.display_image(self.correct_image_path)

        self.clear_buttons()
        for i, option in enumerate(options):
            button = tk.Button(self.buttons_frame, text=option, font=("Arial", 12),
                               command=lambda option=option: self.check_answer(option))
            button.grid(row=i // 2, column=i % 2, padx=10, pady=10)

    def clear_buttons(self):
        for widget in self.buttons_frame.winfo_children():
            widget.grid_forget()

    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def check_answer(self, selected):
        if selected == self.correct_label:
            # Doğru tahminde bulanıksız görüntüyü göster
            self.display_image(self.correct_original_path)
            messagebox.showinfo("Tebrikler!", "Doğru bildiniz!")
            self.start_game()
        else:
            messagebox.showerror("Yanlış Cevap", "Tekrar deneyin!")
            self.display_image(self.correct_image_path)
            self.clear_buttons()

            options = [self.correct_label]
            while len(options) < 4:
                random_choice = random.choice(photo_bank)["label"]
                if random_choice != self.correct_label and random_choice not in options:
                    options.append(random_choice)

            random.shuffle(options)
            for i, option in enumerate(options):
                button = tk.Button(self.buttons_frame, text=option, font=("Arial", 12),
                                   command=lambda option=option: self.check_answer(option))
                button.grid(row=i // 2, column=i % 2, padx=10, pady=10)

# Tkinter GUI başlat
root = tk.Tk()
app = GameApp(root)
root.mainloop()
