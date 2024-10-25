import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

class DrawDataset(Dataset):
    def __init__(self, file_path, data_max_size=None):
        # Загрузка данных из файла формата JSON
        self.data = pd.read_json(file_path, lines=True)
        # Отбор только распознанных рисунков
        self.data = self.data[self.data['recognized'] == True]
        # Ограничение размера набора данных, если указано
        if data_max_size and len(self.data) > data_max_size:
            self.data = self.data[:data_max_size]
        
        self.images = self.data['drawing'].values
        self.processed_images = []

        # Преобразование набора в изображения и нормализация
        for raw_drawing in self.images:
            img = self.stroke_to_image(raw_drawing)
            img = np.array(img).astype(np.float32) / 255.0  # Нормализация изображения в диапазон [0, 1]
            img = torch.from_numpy(img)  # Преобразование в тензор PyTorch
            self.processed_images.append(img.unsqueeze(0))  # Добавление оси канала (1, 64, 64)

    def stroke_to_image(self, raw_drawing):
        # Коэффициенты для изменения размера изображения и его улучшения
        scale_factor = 0.22  # Масштаб для уменьшения координат рисунков
        upscale_factor = 8  # Коэффициент увеличения для получения плавных линий
        original_size = 64  # Окончательный размер изображения
        large_size = original_size * upscale_factor  # Увеличенный размер для рисования линий

        # Преобразование координат линий с масштабированием и смещением
        polylines = (
            zip([(x + 25) * scale_factor * upscale_factor for x in polyline[0]],
                [(y + 25) * scale_factor * upscale_factor for y in polyline[1]])
            for polyline in raw_drawing if len(polyline) == 2
        )

        # Преобразуем набор линий в список для последующего рисования
        polylines_list = [list(polyline) for polyline in polylines]
        
        # Создание пустого увеличенного изображения
        pil_img = Image.new("L", (large_size, large_size), 255)  # Черно-белое изображение, белый фон
        d = ImageDraw.Draw(pil_img)
        
        # Рисование линий с учетом масштабирования и увеличенной толщины
        for polyline in polylines_list:
            d.line(polyline, fill=0, width=int(1.5 * upscale_factor))  # Линии черного цвета
        
        # Масштабирование изображения обратно до 64x64 с использованием LANCZOS для сглаживания
        pil_img = pil_img.resize((original_size, original_size), Image.Resampling.LANCZOS)
        
        return pil_img

    def __len__(self):
        # Возвращает количество изображений в наборе данных
        return len(self.images)

    def __getitem__(self, idx):
        # Возвращает обработанное изображение по индексу
        return self.processed_images[idx]
