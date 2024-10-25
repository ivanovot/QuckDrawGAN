# QuckDrawGAN


## Возможности

**QuckDrawGAN** - это генеративная состязательная сеть (GAN), разработанная для обучения на данных из проекта **Quick, Draw!** (https://github.com/googlecreativelab/quickdraw-dataset). Позволяет обучить и испоьльзовать ее для генерации изображений

![Процесс обучения модели](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYmlqcTd6ZmVzYnE1dG92OGQ0enZ6ZGt6endxbmtlcmJyeWh1dGpnbSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jsBFynL23hcIWOYDvQ/giphy.gif)

**QuckDrawGAN** основан на архитектуре генеративной состязательной сети (GAN), состоящей из двух основных компонентов: генератора и дискриминатора.

---

### Как происходит обучение модели

Обучение **GAN** включает две основные части: **генератор** и **дискриминатор**. Генератор создает изображения, а дискриминатор оценивает, насколько они похожи на настоящие. Генератор пытается улучшить свои изображения, чтобы обмануть дискриминатор, а дискриминатор, в свою очередь, учится лучше различать подделки и настоящие. Этот процесс повторяется много раз, пока генератор не начнет создавать изображения, которые трудно отличить от реальных. 

### Генерация изображений

Генерация изображений происходит в два этапа. Сначала **генератор** создает заданное количество изображений на основе случайного вектора. Затем **дискриминатор** анализирует все сгенерированные изображения и выбирает то, которое наиболее похоже на настоящие. Этот процесс позволяет находить наиболее качественные изображения.

[Предобученные данные для демонстрации работы пакета](pretrained_output/)

## Установка

Для установки проекта выполните следующую команду:

```bash
git clone https://github.com/ivanovot/QuckDrawGAN.git
cd QuckDrawGAN
pip install -r requirements.txt
```

## Применение

Использовать QuckDrawGAN можно двумя основными способами: через командную строку (терминал Bash) или как Python-пакет, подключив его напрямую в коде.

[Руководство по применению в терминале](usage.md)
[Туториал по применению python-пакета](tutorial.ipynb)

#### Обучение модели
Создание обученных моделей
```bash
python -m QuckDrawGAN.train --data_path duck.ndjson --output_path output --epochs 100 --batch_size 64 --data_max_size 30000
```

#### Дообучение дискриминатора
Это нужно для более лучшего распознования плохих изображений на этапе генерации
```bash
python -m QuckDrawGAN.train --fine_tune --generator_file pretrained_output/models/generator.pt --discriminator_file pretrained_output/models/discriminator_fine_tuned.pt --data_path duck.ndjson --fine_tune_epochs 15
```

#### Генерация изображений
Генерация нового изображения на основе случайного шума или заданного ключа
```bash
python -m QuckDrawGAN.model --generator_path pretrained_output/models/generator.pt --discriminator_path pretrained_output/models/discriminator_fine_tuned.pt --output_path result.png --n 16
```

---

## Использованые технологи
Для корректной работы проекта вам потребуется:

- **Python 1.12** 
- **PyTorch**
