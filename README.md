# Проект по дорожным знакам с использованием VLM

## Описание проекта 
Целью этого проекта является исследование возможностей VLM-моделей в задачах, связанных с анализом дорожных знаков и ситуации на дороге. Исследования проводились для VLM модели – **Llava**. 
## Запуск модели
В рамках эксперимента, модель **Llava** была запущена разными способами:
- [x] При помощи открытой [демонстрации]( https://llava-vl.github.io/) от разработчиков
- [x] Локальный запуск Llava 7b на ОС Windows на CPU (12th Gen Intel(R) Core(TM) i7-12700H   2.30 GHz, ОП 32гб)
- [x] Локальный запуск Llava 7b при помощи Google Collab на GPU
- [x] Запуск на вычислительном сервере aichem/aihub (6 GPU a6000, 256 ядер, процессор AMD EPYC 7763 64-Core Processor, 512 GB RAM)
- [ ] Локальный запуск Llava 7b на ОС Windows на GPU

Основная работа проводилась на вычислительном кластере, однако хочется отметить мысли, возникшие при альтернативных способах работы с моделью:
> Открытая демонстрация работает быстро и очень просто запускается, однако есть и существенные недостатки: неспособность модели работать с несколькими изображениями и большая нагрузка на сервер (демонстрация часто вылетала с ошибкой) 

> Модель без особых проблем можно поднять на CPU, но работа с ней станет весьма долгой (примерно от 5 минут на запрос). Кроме того,при перезагрузке или старте демонстрации модель подгружается 20 секунд 

> Модель легко поднять при помощи Google Collab. Несмотря на весьма долгое подключение, получение ответа модели занимает куда меньше времени, хотя всё ещё уступает демонстрации или полноценному запуску на мощном GPU. Кроме того, стоит быть готовым к постоянным вылетам по памяти

## Идеи и брейншторм
Рассматривая модель **Llava** для задач, связанных с анализом дорожных знаков и ситуации на дороге, команда выделила несколько основных направлений для исследований
- Ассистирование водителю в сложных дорожных ситуациях/ оценка ситуации на дороге
- Тренажёр для углубления знаний неопытных водителей
- Детектор аномалий на дороге

В результате проведения нескольких экспериментов, было решено для итоговой работы остановиться на третьем направлении (детекция аномалий на дороге), так как для реализации первого направления (ассистирование водителя) модели **Llava** не хватало быстродействия, а второе направление (тренажёр для неопытных водителей) требовало нетривиального дообучения модели, так как ответы модели не были коректными для большого числа промптов. 

## Промпты и эксперементы
Был проведён ряд экспериментов по использованию модели в двух концепциях: тренажёр для неопытных водителей и детекция аномалий на дороге. 
Для корректной работы с моделью в качестве идей для промптов использовались следующие гипотезы:
- *Если обратиться к **Llava** как к ассистенту и попросить помощи в решении относительно ситуации на дороге, модель выдаст более корректный результат*
- *Если явно указать в промпте требования к безопасности, модель выдаст более корректный результат*
- *Если добавить в промпт вероятную информацию, которую может предоставить навигатор и попросить дать совет о поведении на дороге, модель выдаст более корректный результат*
- *Если заслать в модель изображение с выделенными bounding boxes, отмеченными Yolo, модель выдаст более корректный результат*
- *Если упоминать в промптах светофоры и/или разметку, модель выдаст более корректный результат*
- *Если спросить у **Llava**, можно ли сделать вывод о ситуации на дороге по фото, модель выдаст более корректный результат*

Результаты некоторых экспериментов можно найти [здесь]( https://docs.google.com/document/d/1CkAlat0neT09ALR9ZjdIUA4YANHjSUazaYBztKVp1MY/edit)
После проверки некоторых из данных гипотез, результаты были отранжированы по корректности. Ранжированные результаты можно наблюдать [здесь]( https://docs.google.com/document/d/1P4TMRpqfw8LfZ9L26uXy3vpZu9d8sD9M-STCRi9Eeqo/edit?usp=sharing)

Интересные фотографии, использованные для экспериментов по обеим задачам, можно посмотреть [здесь]( https://drive.google.com/drive/u/1/folders/1NCDW2rE2_3dD1_IKcNiapWZkJ9ZOBInN)

## Вспомогательные средства (рука помощи Yolo)
Командой было решено, что можно повысить эффективность работы модели **Llava**, если на вход подавать изображения, на которых были бы выделены дорожные знаки. Это помогло более точно описать ситуацию на дороге. Для решения этой задачи была выбрана модель YoloV8 в качестве базового детектора. Для обучения использовался большой [датасет с российскими знаками](https://www.kaggle.com/datasets/watchman/rtsd-dataset). Было проведено несколько экспериментов по обучению модели:

1. Модель обучалась детекции 198 классов знаков, представленных в выбранном датасете. (Модель обучалась довольно долго и достигла не самого большого качества)
2. Число классов в модели ограничилось до двух. Такая модель обучилась на более высокое качество, в связи с чем ей было отдано предпочтение.

**Промежуточный результат обучения моделей:**

|               Модель               |   Число эпох   |  Recall  |  MAP50  |                                   
|:-:|:-:|:-:|:-:|
|        Multiclass_baseline         |       10       |   0.501  |  0.559  |
|           Binary _baseline         |       10       |   0.892  |  0.949  | 

## Немного о сервисе
Запуск сервиса. В корне проекта:

````
docker compose up
````
Сервис работает с:
1. Загруженными видео
2. Youtube видео
3. RTSP потоком

Чтобы запустить детектор 
и обращения к VLM, нужно нажать кнопку "Detect".
Ответы VLM будут показываться титрами на остановленном
видео.

После приостановки или завершения детекции 
появится возможность скачать результат.




