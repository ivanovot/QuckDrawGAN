import argparse
import os
import logging
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils.models import Generator, Discriminator, latent_dim
from .utils.data import DrawDataset


def train(epochs, batch_size, data_path, output_path='output', lr_g=0.001, lr_d=0.002, data_max_size=None):
    # Создание директорий для сохранения изображений и моделей
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'models'), exist_ok=True)
    
    # Инициализация логирования
    log_file = os.path.join(output_path, 'training_logs.log')
    with open(log_file, 'w'):
        pass  # Очищаем файл логов
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # Определение устройства для обучения
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading dataset")
    dataset = DrawDataset(data_path, data_max_size)

    # Инициализация генератора и дискриминатора
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Оптимизаторы для генератора и дискриминатора
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))

    # Функция потерь
    adversarial_loss = nn.L1Loss()

    # Фиксированные векторы шума для генерации изображений в каждом эпохе
    fix_z = torch.randn(64, latent_dim).to(device)

    # Загрузка данных
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    logging.info("Training started")

    # Основной цикл обучения
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", bar_format='{l_bar}{bar:12}{r_bar}')
        generator.train()

        for i, real_imgs in enumerate(progress_bar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Создание меток для реальных и поддельных изображений
            valid_labels = torch.full((batch_size, 1), random.uniform(0.7, 1)).to(device)
            fake_labels = torch.full((batch_size, 1), random.uniform(0, 0.3)).to(device)

            # Обновление дискриминатора
            optimizer_D.zero_grad()

            # Генерация поддельных изображений
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z)

            # Вычисление потерь для реальных и поддельных изображений
            real_preds = discriminator(real_imgs)
            fake_preds = discriminator(gen_imgs.detach())

            loss_real = adversarial_loss(real_preds, valid_labels)
            loss_fake = adversarial_loss(fake_preds, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # Обновление генератора
            optimizer_G.zero_grad()

            # Генерация новых поддельных изображений
            gen_imgs = generator(z)

            # Потери генератора на основе предсказаний дискриминатора
            fake_preds_for_gen = discriminator(gen_imgs)
            loss_G = adversarial_loss(fake_preds_for_gen, valid_labels)

            loss_G.backward()
            optimizer_G.step()

            # Обновление информации в прогресс-баре
            progress_bar.set_postfix(Loss_D=loss_D.item(), Loss_G=loss_G.item())

        # Логирование итогов эпохи
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

        # Сохранение изображений и модели
        with torch.no_grad():
            generator.eval()
            gen_imgs = generator(fix_z)
            vutils.save_image(gen_imgs.data, os.path.join(output_path, 'images', f'{epoch+1}.png'), nrow=8, normalize=True)
            torch.save(generator.state_dict(), os.path.join(output_path, 'models', 'generator.pt'))
            torch.save(discriminator.state_dict(), os.path.join(output_path, 'models', 'discriminator.pt'))


def discriminator_fine_tune(generator_file, discriminator_file, data_path, batch_size=64, fine_tune_epochs=10, lr_d=0.002, data_max_size=None):
    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Загрузка сохранённых моделей
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.load_state_dict(torch.load(generator_file, map_location=device, weights_only=True))
    discriminator.load_state_dict(torch.load(discriminator_file, map_location=device, weights_only=True))

    # Оптимизатор для дискриминатора
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))
    adversarial_loss = nn.L1Loss()

    # Загрузка данных
    dataset = DrawDataset(data_path, data_max_size)
    fine_tune_dataloader = DataLoader(dataset, batch_size=batch_size//2, shuffle=True, pin_memory=True)

    logging.info(f"Fine-tuning discriminator for {fine_tune_epochs} epochs")

    for epoch in range(fine_tune_epochs):
        progress_bar = tqdm(fine_tune_dataloader, desc=f"Fine-tuning Discriminator [{epoch+1}/{fine_tune_epochs}]", bar_format='{l_bar}{bar:12}{r_bar}')
        discriminator.train()

        for i, real_imgs in enumerate(progress_bar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Создание меток для реальных и поддельных изображений
            valid_labels = torch.full((batch_size, 1), random.uniform(0.7, 1)).to(device)
            fake_labels = torch.full((batch_size // 2, 1), random.uniform(0, 0.3)).to(device)

            # Обновление дискриминатора
            optimizer_D.zero_grad()

            # Генерация поддельных изображений
            z = torch.randn(batch_size // 2, latent_dim).to(device)
            gen_imgs = generator(z)

            # Потери для реальных и поддельных изображений
            real_preds = discriminator(real_imgs)
            fake_preds = discriminator(gen_imgs.detach())

            loss_real = adversarial_loss(real_preds, valid_labels)
            loss_fake = adversarial_loss(fake_preds, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # Обновление информации в прогресс-баре
            progress_bar.set_postfix(Loss_D=loss_D.item())

        # Логирование результатов дообучения
        logging.info(f"Fine-tune Epoch [{epoch+1}/{fine_tune_epochs}], Loss_D: {loss_D.item():.4f}")

    # Сохранение обновлённого дискриминатора
    torch.save(discriminator.state_dict(), os.path.join(os.path.dirname(discriminator_file), 'discriminator_fine_tuned.pt'))


# Определение аргументов командной строки
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN model with specified parameters.")
    
    # Основные аргументы для функции train
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_path', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--lr_g', type=float, default=0.001, help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=0.002, help='Learning rate for discriminator')
    parser.add_argument('--data_max_size', type=int, default=None, help='Maximum size of data to use')

    # Аргументы для дообучения дискриминатора
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune discriminator')
    parser.add_argument('--generator_file', type=str, help='Path to generator weights for fine-tuning')
    parser.add_argument('--discriminator_file', type=str, help='Path to discriminator weights for fine-tuning')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning discriminator')
    
    args = parser.parse_args()
    
    if args.fine_tune:
        # Запуск функции дообучения дискриминатора
        discriminator_fine_tune(args.generator_file, args.discriminator_file, args.data_path, args.batch_size, args.fine_tune_epochs, args.lr_d, args.data_max_size)
    else:
        # Запуск основной функции тренировки
        train(args.epochs, args.batch_size, args.data_path, args.output_path, args.lr_g, args.lr_d, args.data_max_size)