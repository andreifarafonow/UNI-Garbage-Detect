# UNI-Garbage-Detect
## Разработка системы видео-аналитики подсчета твердых бытовых отходов (ТБО)

## Описание
Этот проект посвящен разработке системы видео-аналитики, которая способна автоматически подсчитывать количество твердых бытовых отходов на видеозаписи. Данная система использует технологии *Python* и архитектуру *YOLOv8*, реализованную с использованием фреймворка *PyTorch*.

## Скринкаст
![](https://github.com/MaxTube-dot/Asserts/blob/master/train_batch3682.jpg)
![](https://github.com/MaxTube-dot/Asserts/blob/master/ezgif.com-optimize.gif)


## Архитектура решения

![](https://github.com/MaxTube-dot/Asserts/blob/master/image_2023-10-28_20-04-35.png)

## Состав проекта
Проект состоит из следующих частей:

1. Python: Для разработки системы видео-аналитики используется язык программирования Python. Python широко используется для анализа данных, машинного обучения и глубокого обучения, поэтому оказывается идеальным выбором для этого проекта.

2. YOLOv8: YOLOv8 - это одна из самых эффективных архитектур для обнаружения объектов на изображении или видео. Она используется в данном проекте для анализа видеозаписей и подсчета твердых бытовых отходов.

3. PyTorch: PyTorch - это открытая библиотека глубокого обучения, разработанная для Python. Она предоставляет удобный интерфейс для обучения нейронных сетей и обработки данных, что делает ее идеальным выбором для реализации архитектуры YOLOv8.



## Установка и настройка
1. Для правильной работы алгоритма, требуется версия Python 3.10.11
2. Также, для запуска необходимо скачать и установить [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/) со следующими пакетами:

![Build Tools](https://raw.githubusercontent.com/MaxTube-dot/Asserts/master/photo_2023-10-29_08-11-12.jpg)

3. Изменить версию PIP
```
python -m pip install pip==21.1.1
```
4. В корне каталога решения выполнить:
```
pip install -r requirements.txt
```
5. Перейти в libraries/cython_bbox-0.1.3 и выполнить:
```
python setup.py -q develop
```
6. Перейти в libraries/ByteTrack и выполнить:
```
python setup.py -q develop
```


## Использование

У решения 2 варианта выполнения:

1. Заменить папку **frames_rgb** на необходимую и запустить файл **track.py**
2. Удалить папку **frames_rgb** и при запуске **track.py** указать абсолютный путь к директории семпла (датасета)

## Авторы
Проект разработан командой разработчиков:

- [Левин Илья](https://github.com/MaxTube-dot "Левин Илья") - Full-stack  C# developer
- [Фарафонов Андрей](https://github.com/andreifarafonow "Фарафонов Андрей") - Full-stack  developer, ML engineer
- [Меркулов Андрей](https://github.com/Dead-CLu8ku "Меркулов Андрей") -  Python developer
- [Паневин Даниил](https://www.behance.net/daniilpanevin "Паневин Даниил") - Product Designer, Manager
