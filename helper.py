import torch.cuda
from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube

import settings
import random
import time
import io
import requests
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path: The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def get_vlm_result(image):
    answer = ("Я памятник себе воздвиг нерукотворный,К нему не зарастет народная тропа, Вознесся выше он главою "
              "непокорной Александрийского столпа. Нет, весь я не умру — душа в заветной лире Мой прах переживет и "
              "тленья убежит — И славен буду я, доколь в подлунном мире Жив будет хоть один пиит.")

    return answer


# def get_vlm_result(image):
#     json_data = {'text': settings.PROMPT}
#     cv2.imwrite("image.jpg", image)
#
#     with open("image.jpg", 'rb') as file:
#         files = {'image': ("image.jpg", file, 'image/jpeg')}
#         data = {'json_data': json.dumps(json_data)}
#
#         response = requests.post(settings.VLM_URL, files=files, data=data)
#
#     return response.text


def wrap_text(text, max_width, font):
    lines = []
    words = text.split(' ')

    current_line = ''
    for word in words:
        test_line = current_line + word + ' '
        line_width, _ = font.getsize(test_line)
        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line[:-1])
            current_line = word + ' '

    lines.append(current_line[:-1])
    return lines


def add_text_to_image(image, sentence):
    # Load a suitable font with the desired size
    image = Image.fromarray(image)
    font_size = 14
    font = ImageFont.truetype("times.ttf", font_size)

    # Apply line wrap if the text is longer than the image width
    max_width = int(image.width * 0.9)
    wrapped_text = wrap_text(sentence, max_width, font)

    draw = ImageDraw.Draw(image)
    line_spacing = 1.3  # Adjust this value based on your needs
    x = (image.width - max_width) // 2
    y = (image.height - int(font_size * len(wrapped_text) * line_spacing)) // 2

    for line in wrapped_text:
        line_width, _ = font.getsize(line)
        draw.text(((image.width - line_width) // 2, y), line, "white", font=font)
        y += int(font_size * line_spacing)

    # save image to disk
    return np.asarray(image)


def save_image(image, directory="processed_images"):
    count = str(len(os.listdir(directory)))
    count_str = "0" * (6 - len(count)) + count
    Image.fromarray(image).save(f"{directory}/{count_str}.jpg")


def _display_detected_frames(model, st_frame, st_for_text, image, print_text=False):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Parameters:
        model_path: Path to the `YOLOv8` class containing the YOLOv8 model.

    Returns:
    None
    """
    # # Plot the detected objects on the video frame
    result = model.predict(image, imgsz=512, conf=0.55)[0]
    res_plotted = result.plot()

    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    save_image(res_plotted)

    if print_text and len(result.boxes.cls) >= 2:
        result = get_vlm_result(image)
        current_ind = 0
        while current_ind < len(result):
            part_result = result[:current_ind]
            # st_for_text.text(part_result)
            text_image = add_text_to_image(res_plotted, part_result)
            for _ in range(2):
                st_frame.image(text_image,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
                save_image(text_image)
            current_ind += 1
        time.sleep(5)

        return True

    return False


def play_youtube_video(model_path):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        model_path: Path to the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    processing_preparation()

    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            delete_processed_images()
            model = load_model(model_path)
            if (not torch.cuda.is_available() or
                    torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 < 1500):
                model.to(torch.device("cpu"))
                print("using cpu")
            else:
                print("using gpu")
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            st_for_text = st.empty()
            process_video(vid_cap, st_frame, st_for_text, model)
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(model_path):
    """
    Plays a rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        model_path: Path to the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    processing_preparation()
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    if st.sidebar.button('Detect Objects'):
        try:
            delete_processed_images()
            model = load_model(model_path)
            if (not torch.cuda.is_available() or
                    torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 < 1500):
                model.to(torch.device("cpu"))
                print("using cpu")
            else:
                print("using gpu")
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            st_for_text = st.empty()
            process_video(vid_cap, st_frame, st_for_text, model)
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(model_path):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        model_path: Path to the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    processing_preparation()
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        try:
            delete_processed_images()
            model = load_model(model_path)
            if (not torch.cuda.is_available() or
                    torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 < 1500):
                model.to(torch.device("cpu"))
                print("using cpu")
            else:
                print("using gpu")
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            st_for_text = st.empty()
            process_video(vid_cap, st_frame, st_for_text, model)
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(model_path):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        model_path: Path to the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    processing_preparation()
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    if uploaded_file:
        video_bytes = io.BytesIO(uploaded_file.read())
        temporary_location = "videos/video"

        with open(temporary_location, 'wb') as out:
            out.write(video_bytes.read())

        out.close()

    if st.sidebar.button('Detect Video Objects') and uploaded_file:
        try:
            delete_processed_images()
            model = load_model(model_path)
            if (not torch.cuda.is_available() or
                    torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 < 1500):
                model.to(torch.device("cpu"))
                print("using cpu")
            else:
                print("using gpu")
            vid_cap = cv2.VideoCapture(temporary_location)
            st_frame = st.empty()
            st_for_text = st.empty()
            process_video(vid_cap, st_frame, st_for_text, model)
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def process_video(vid_cap, st_frame, st_for_text, model):
    count = 0
    while vid_cap.isOpened():
        count += 1
        print_text = (count > 120)
        success, image = vid_cap.read()
        if success:
            null_count = _display_detected_frames(model,
                                                  st_frame,
                                                  st_for_text,
                                                  image,
                                                  print_text
                                                  )
            if null_count:
                count = 0
        else:
            vid_cap.release()
            break


def generate_video(image_folder, output_path):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith("jpg")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_path, 0, 23, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        os.remove(os.path.join(image_folder, image))

    cv2.destroyAllWindows()
    video.release()


def processing_preparation():
    if len(os.listdir("processed_images")) > 1:
        st.button("Generate video from results", on_click=generate_and_download)


def generate_and_download():
    generate_video("processed_images", "result.avi")
    with open('result.avi', "rb") as f:
        st.download_button("Load processed video", f, file_name="result.avi")


def delete_processed_images():
    images = [img for img in sorted(os.listdir("processed_images")) if img.endswith("jpg")]

    for image in images:
        os.remove(os.path.join("processed_images", image))
