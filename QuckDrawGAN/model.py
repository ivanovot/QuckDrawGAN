import argparse
import torch
import numpy as np
import torchvision.utils as vutils
from .utils.models import Generator, Discriminator, latent_dim
import hashlib
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Model:
    def __init__(self, generator_path, discriminator_path=None):
        # Определяем устройство для выполнения (GPU или CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Инициализация и загрузка генератора
        self.generator = Generator(latent_dim).to(self.device)
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device, weights_only=True))  # Загружаем веса генератора
        self.generator.eval()  # Переводим генератор в режим оценки

        # Инициализация дискриминатора, если задан путь к его весам
        if discriminator_path:
            self.discriminator = Discriminator().to(self.device)
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device, weights_only=True))  # Загружаем веса дискриминатора
            self.discriminator.eval()  # Переводим дискриминатор в режим оценки
        else:
            self.discriminator = None  # Если дискриминатор не используется

    def generate(self, n=1, seed=None):
        """Генерирует n изображений. Если дискриминатор загружен, возвращает изображение с наибольшей оценкой дискриминатора."""
        with torch.no_grad():  # Отключаем градиенты для режима оценки
            # Установка сида для воспроизводимости, если задан
            if seed is not None:
                seed_number = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)  # Преобразуем текстовый сид в число
                torch.manual_seed(seed_number)  # Устанавливаем сид для генерации

            # Генерация случайного латентного вектора
            z = torch.randn(n, latent_dim).to(self.device)

            # Генерация изображений
            gen_imgs = self.generator(z)

            # Если дискриминатор загружен, выбираем изображение с наилучшей оценкой
            if self.discriminator:
                predictions = self.discriminator(gen_imgs).cpu().numpy().flatten()  # Получаем оценки дискриминатора
                max_pred_idx = predictions.argmax()  # Находим индекс изображения с максимальной оценкой
                best_img = gen_imgs[max_pred_idx].cpu().squeeze().numpy()  # Преобразуем изображение в формат (H, W)
                return best_img  # Возвращаем лучшее изображение
            else:
                # Если дискриминатор не загружен, возвращаем первое сгенерированное изображение
                return gen_imgs[0].cpu().squeeze().numpy()  # Преобразуем изображение в формат (H, W)

if __name__ == "__main__":
    # Определение аргументов командной строки
    parser = argparse.ArgumentParser(description="Generate image using pretrained GAN model")
    parser.add_argument('--generator_path', type=str, required=True, help='Path to generator model weights')
    parser.add_argument('--discriminator_path', type=str, help='Path to discriminator model weights (optional)')
    parser.add_argument('--output_path', type=str, default='result.png', help='Path to save the generated image')
    parser.add_argument('--n', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--seed', type=str, help='Seed for random generation (optional)')

    args = parser.parse_args()

    # Инициализация модели и генерация изображения
    model = Model(args.generator_path, args.discriminator_path)
    generated_image = model.generate(n=args.n, seed=args.seed)

    # Нормализация изображения
    min_val = np.min(generated_image)
    max_val = np.max(generated_image)

    # Применяем нормализацию
    normalized_image = (generated_image - min_val) / (max_val - min_val) * 255

    # Приводим к 8-битному формату
    normalized_image = normalized_image.astype(np.uint8)

    # Проверяем размерность и преобразуем в RGB, если это необходимо
    if normalized_image.ndim == 2:  # Если изображение в градациях серого
        # Преобразуем в RGB (64, 64) -> (64, 64, 3)
        normalized_image = np.stack([normalized_image] * 3, axis=-1)
    elif normalized_image.shape[2] == 1:  # Если изображение с одним каналом
        # Удаляем канал и преобразуем в RGB
        normalized_image = np.squeeze(normalized_image, axis=2)

    # Создаем изображение в формате RGB
    img = Image.fromarray(normalized_image, mode='RGB')
    img.save(args.output_path)
