import os
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from typing import List
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import cv2



INPUT_CHECK_DATASET_PATH =input("Enter the full path to the test images:")

MODEL = "model/last.pt"
CLASS_ID = [0, 1, 2, 3]
TARGET_VIDEO_PATH = "result.mp4"
RGB_IMAGES_FOLDER = "frames_rgb"
FRAMES_OUTPUT_FOLDER = "frames_output"
STEP_BY_PICTURE = 0.01

FPS = 20


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1
    mot20: bool = False


# Преобразует обнаружения в формат, который может быть использован функцией match_detections_with_tracks
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# Преобразует List[STrack] в формат, который может быть использован функцией match_detections_with_tracks
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# Сопоставляет наши ограничивающие рамки с предсказаниями
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


image_list = []
image_names = []

# Цикл по файлам в папке
for filename in os.listdir(os.path.join(INPUT_CHECK_DATASET_PATH, RGB_IMAGES_FOLDER)):
    # Полный путь к файлу
    file_path = os.path.join(INPUT_CHECK_DATASET_PATH, RGB_IMAGES_FOLDER, filename)
    
    # Проверка, что файл является изображением
    if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # Открываем изображение и добавляем его в список
        image = cv2.imread(file_path)

        image_list.append(image)
        image_names.append(filename)

model = YOLO(MODEL)
model.fuse()

# dict сопостовления class_id с class_name
CLASS_NAMES_DICT = model.model.names



# Создание BYTETracker сущности
byte_tracker = BYTETracker(BYTETrackerArgs())


# Создание сущности BoxAnnotator и LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4)

#TODO переписать 
# Получение размеров первой картинки
height, width, _ = image_list[0].shape

# Расчет времени кадра на основе частоты кадров
frame_interval = 1 / FPS  # Частота кадров в секунду

# Расчет продолжительности видео на основе числа кадров и времени кадра
num_frames = len(image_list)
duration = num_frames * frame_interval

# Создание VideoInfo объекта
video_info = VideoInfo(
    width = width,
    height = height,
    fps=FPS,  # Частота кадров в секунду
    total_frames=num_frames
)

if not os.path.exists(os.path.join(INPUT_CHECK_DATASET_PATH, FRAMES_OUTPUT_FOLDER)):
    os.makedirs(os.path.join(INPUT_CHECK_DATASET_PATH, FRAMES_OUTPUT_FOLDER))


objects_dict = {}

common_classes_count = {
    'wood': 0,
    'glass': 0,
    'plastic': 0,
    'metal': 0
}

# Открытие видео файла
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    frame_number = 0

    # цикл по кадрам
    for frame in tqdm(image_list, total=len(image_list)):
        # Прогнозирование модели по одному кадру
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # Отфильтровывание обнаружений с нежелательными классами
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        ordered_classes = sorted(detections.class_id.tolist())
        
        frame_out = [ordered_classes.count(0), ordered_classes.count(1), ordered_classes.count(2), ordered_classes.count(3)]

        # Открытие файла для записи
        with open(os.path.join(INPUT_CHECK_DATASET_PATH, FRAMES_OUTPUT_FOLDER, os.path.splitext(image_names[frame_number])[0] + ".txt"), "w") as file:
            # Запись каждого элемента массива на новой строке
            file.write('\n'.join(str(item) for item in frame_out))

        # Отслеживание обнаружений
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=(height,width),
            img_size=(height,width)
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        # Фильтрация обнаружений без использования трекеров
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        t = detections.filter(mask=mask, inplace=True)

        # Форматирование подписи
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        detect_id = 0

        # Заполнение objects_dict уникальными объектами
        for label in labels:
            elements = label.split()
            key = elements[0][1:]  # Получение ключа без символа '#'
            value = elements[1]    # Получение значения класса

            x1 = detections.xyxy[detect_id][0]
            x2 = detections.xyxy[detect_id][3]

            # Определение границ
            left = 0
            right = 0
            
            if x1 > x2:
                right = x1
                left = x2
            else:
                right = x2
                left = x1


            # Область трекинга
            if value=='metal' or (left > width * STEP_BY_PICTURE):
                if key not in objects_dict:
                    objects_dict[key] = value

            detect_id += 1

        # Отображение подписи и вывод картинки
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        sink.write_frame(frame)

        frame_number += 1
        

for value in objects_dict.values():
    common_classes_count[value] += 1

material_order = ['wood', 'glass', 'plastic', 'metal']
classes_count_array = [common_classes_count[material] for material in material_order]

with open(os.path.join(INPUT_CHECK_DATASET_PATH, "output.txt"), "w") as file:
    # Запись общего кол-ва объектов
    file.write('\n'.join(str(item) for item in classes_count_array))