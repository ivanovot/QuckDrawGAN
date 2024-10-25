import torch.nn as nn

latent_dim = 100  # Размерность латентного пространства (входного шума для генератора)

# Класс генератора для генерации изображений из латентного пространства
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Размеры начального изображения и количество каналов для начала транспонированных операций
        self.init_size = 4  # Размер изображения после первой линейной трансформации
        self.start_channels = 512  # Количество каналов на первом этапе

        # Последовательная модель генератора
        self.model = nn.Sequential(
            # Линейное преобразование латентного вектора в развернутую форму для дальнейшего увеличения
            nn.Linear(latent_dim, self.start_channels * self.init_size ** 2),
            nn.BatchNorm1d(self.start_channels * self.init_size ** 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Преобразование в 4D тензор для начала операций с изображениями
            nn.Unflatten(1, (self.start_channels, self.init_size, self.init_size)),

            # Начало операций с изображением (увеличение размера)
            nn.Upsample(scale_factor=2),  # Увеличение размера изображения в 2 раза

            # Сверточные слои с уменьшением количества каналов и последующими нелинейностями
            nn.Conv2d(self.start_channels, self.start_channels // 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.start_channels // 3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # Еще одно увеличение

            nn.Conv2d(self.start_channels // 3, self.start_channels // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.start_channels // 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # Третье увеличение

            nn.Conv2d(self.start_channels // 4, self.start_channels // 6, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.start_channels // 6, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # Четвертое увеличение

            nn.Conv2d(self.start_channels // 6, self.start_channels // 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.start_channels // 8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Последний сверточный слой для вывода изображения размером 1xWxH с функцией активации Тангенс
            nn.Conv2d(self.start_channels // 8, 1, 3, stride=1, padding=1),
            nn.Tanh()  # Приведение значений пикселей в диапазон [-1, 1]
        )

    def forward(self, z):
        # Прямое распространение через сеть генератора
        out = self.model(z)
        return out  # Возвращаем сгенерированное изображение


# Класс дискриминатора для различения реальных и сгенерированных изображений
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Последовательная модель дискриминатора
        self.model = nn.Sequential(
            # Первый сверточный блок
            nn.Conv2d(1, 64, 3, stride=2, padding=1),  # Уменьшает размер изображения до (64, 32, 32)
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Вероятность выключения нейронов для регуляризации

            # Второй сверточный блок
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Уменьшает размер до (128, 16, 16)
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Третий сверточный блок
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # Уменьшает размер до (256, 8, 8)
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Четвертый сверточный блок
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # Поддерживает размер (256, 8, 8)
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # Уменьшает размер до (256, 4, 4)
            nn.Dropout(0.3),

            # Пятый сверточный блок
            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # Уменьшает размер до (512, 2, 2)
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # Размер до (512, 1, 1)

            # Преобразование в плоский вектор для классификации
            nn.Flatten(),  # Преобразует изображение в вектор (512 * 2 * 2 = 2048)
            nn.Linear(512 * 2 * 2, 1)  # Полносвязный слой для получения одного скалярного выхода
        )

    def forward(self, img):
        # Прямое распространение через сеть дискриминатора
        out = self.model(img)
        return out  # Возвращаем вероятность принадлежности к классу "реальное" или "сгенерированное"
