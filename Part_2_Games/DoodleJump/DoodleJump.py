import os
import time
import json
import random
import argparse
import numpy as np
from functools import wraps
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

def timeit(func):
    """Декоратор для измерения времени выполнения функции.
    
    Args:
        func: Функция, которую нужно обернуть.

    Returns:
        Функция с поведением, аналогичным оригинальному, но с измерением времени.
    """
    @wraps(func)
    def timed_function(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Функция '{func.__name__}' выполнялась {elapsed_time:.4f} секунд.")
        return result

    return timed_function

class DoodleJumpGame:
    def __init__(self, width=375, height=667, platform_width=65, platform_height=20, experiment_name='doodlejump'):
        """Инициализация объекта DoodleJumpGame.

        Args:
            width (int): Ширина игрового поля (по умолчанию 375).
            height (int): Высота игрового поля (по умолчанию 667).
            platform_width (int): Ширина платформы (по умолчанию 65).
            platform_height (int): Высота платформы (по умолчанию 20).
            experiment_name (str): Название эксперимента (по умолчанию 'doodlejump').
        """
        self.settings = self.init_settings(width, height, platform_width, platform_height)
        self.doodle, self.platforms = self.init_states()
        self.sensor_web = self.create_sensor_web()
        # Инициализация логгера
        self.writer = self.initialize_log_dir(experiment_name)

    def initialize_log_dir(self, experiment_name):
        """Создает уникальную директорию для эксперимента и инициализирует SummaryWriter.

        Args:
            experiment_name (str): Название эксперимента.

        Returns:
            SummaryWriter: Инициализированный объект SummaryWriter для логирования.
        """
        """Создает уникальную директорию для эксперимента и инициализирует SummaryWriter."""
        self.experiment_name = f'{experiment_name}'  # Сохраняем название эксперимента
        self.log_dir = f'logs/{experiment_name}'  # Базовый путь для логов
        
        i = 0
        while os.path.exists(self.log_dir):
            self.log_dir = f'logs/{experiment_name}_{i}'
            i += 1
        os.makedirs(self.log_dir)  # Создаем директорию, если она не существует
        print(f"Директория для эксперимента '{experiment_name}' инициализирована по пути: {self.log_dir}")

        self.unique_experiment_name = os.path.basename(self.log_dir)  # Сохраняем уникальное имя
        return SummaryWriter(log_dir=self.log_dir)  # Возвращаем инциализированный SummaryWriter
    
    def log_hparams_and_metrics(self, hparams, best_score):
        """Логирует гиперпараметры и лучший результат в TensorBoard с временной меткой.

        Args:
            hparams (dict): Гиперпараметры, используемые в эксперименте.
            best_score (float): Лучший результат текущего эксперимента.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

        # Логирование гиперпараметров и метрик
        self.writer.add_hparams(hparams, {'best_score': best_score}, run_name=run_name)

    def init_settings(self, width, height, platform_width, platform_height):
        """Инициализация настроек игры.

        Args:
            width (int): Ширина игрового поля.
            height (int): Высота игрового поля.
            platform_width (int): Ширина платформы.
            platform_height (int): Высота платформы.

        Returns:
            dict: Словарь с настройками игры.
        """
        return {
            'width': width,
            'height': height,
            'platformWidth': platform_width,
            'platformHeight': platform_height,
            'gravity': 0.33,
            'drag': 0.3,
            'bounceVelocity': -12.5,
            'minPlatformSpace': 15,
            'maxPlatformSpace': 20,
            'keydown': False,
            'score': 0,
            'platformStart': height - 50
        }

    def init_states(self):
        """Инициализация состояний игрока и платформ.

        Returns:
            tuple: Игрок (дудл) и список платформ в виде словарей.
        """
        settings = self.settings
        platforms = [{
            'x': settings['width'] / 2 - settings['platformWidth'] / 2,
            'y': settings['platformStart']
        }]
        y = settings['platformStart']
        while y > 0:
            y -= settings['platformHeight'] + np.random.randint(
                settings['minPlatformSpace'], settings['maxPlatformSpace'])
            while True:
                x = np.random.uniform(25, settings['width'] - 25 - settings['platformWidth'])
                if not ((y > settings['height'] / 2) and (x > settings['width'] / 2 - settings['platformWidth'] * 1.5) and (x < settings['width'] / 2 + settings['platformWidth'] / 2)):
                    break
            platforms.append({'x': x, 'y': y})

        doodle = {
            'width': 40,
            'height': 60,
            'x': settings['width'] / 2 - 20,
            'y': settings['platformStart'] - 60,
            'dx': 0,
            'dy': 0,
            'playerDir': 0,
            'prevDoodleY': settings['platformStart'] - 60
        }
        return doodle, platforms
    
    def create_sensor_web(self):
        """Создает сетку сенсоров для определения состояния игровых объектов.

        Returns:
            numpy.ndarray: Координаты сенсоров в виде массива.
        """
        sensor_web = np.meshgrid(
            np.arange(-self.settings['width'] * 2 // 3, self.settings['width'] * 2 // 3, 50),
            np.arange(-self.settings['height'] * 2 // 3, self.settings['height'] * 2 // 3, 75)
        )
        return np.concatenate([sensor_web[0].flatten()[:, None], sensor_web[1].flatten()[:, None]], axis=1)

    def restart(self):
        """Сбрасывает состояние игры к начальному.

        Сбрасывает состояние дудла и платформ.
        """
        self.doodle, self.platforms = self.init_states()

    def loop(self):
        """Основной игровой цикл, включает обновление положения объектов и проверку коллизий.

        Returns:
            int: -1 если игра закончилась (дудл вышел за пределы), 0 если игра продолжается.
        """
        doodle, platforms, settings = self.doodle, self.platforms, self.settings
        doodle['dy'] += settings['gravity']

        if doodle['y'] < settings['height'] / 2 and doodle['dy'] < 0:
            for i, _ in enumerate(platforms):
                platforms[i]['y'] -= doodle['dy']
            
            while platforms[-1]['y'] > 0:
                platforms.append({
                    'x': np.random.uniform(25, settings['width'] - 25 - settings['platformWidth']),
                    'y': np.random.uniform(platforms[-1]['y'] - (settings['platformHeight'] + np.random.uniform(settings['minPlatformSpace'], settings['maxPlatformSpace'])))
                })
                settings['minPlatformSpace'] = min(settings['minPlatformSpace'] + 0.5, settings['height'] / 2 - 0.5)
                settings['maxPlatformSpace'] = min(settings['maxPlatformSpace'] + 0.5, settings['height'] / 2)
        else:
            doodle['y'] += doodle['dy']

        if not settings['keydown']:
            if doodle['playerDir'] < 0:
                doodle['dx'] += settings['drag']
                if doodle['dx'] > 0:
                    doodle['dx'] = 0
                    doodle['playerDir'] = 0
            elif doodle['playerDir'] > 0:
                doodle['dx'] -= settings['drag']
                if doodle['dx'] < 0:
                    doodle['dx'] = 0
                    doodle['playerDir'] = 0

        doodle['x'] += doodle['dx']

        if doodle['x'] + doodle['width'] < 0:
            doodle['x'] = settings['width']
        elif doodle['x'] > settings['width']:
            doodle['x'] = -doodle['width']

        for platform in platforms:
            if (doodle['dy'] > 0 and
                doodle['prevDoodleY'] + doodle['height'] <= platform['y'] and
                doodle['x'] < platform['x'] + settings['platformWidth'] and
                doodle['x'] + doodle['width'] > platform['x'] and
                doodle['y'] < platform['y'] + settings['platformHeight'] and
                doodle['y'] + doodle['height'] > platform['y']):
                doodle['y'] = platform['y'] - doodle['height']
                doodle['dy'] = settings['bounceVelocity']

        doodle['prevDoodleY'] = doodle['y']
        platforms_cleared = len(platforms)
        platforms = [p for p in platforms if p['y'] < settings['height']]
        platforms_cleared -= len(platforms)
        settings['score'] += platforms_cleared

        if doodle['y'] > settings['height'] + doodle['height']:
            return -1
        return 0

    def apply_action(self, actionId):
        """Применяет действие игрока к дудлу.

        Args:
            actionId (int): Идентификатор действия (например, влево, вправо, ожидание).
        """
        settings, doodle = self.settings, self.doodle
        actionMap = {0: 37, 1: 39, 2: -1} # Left, Right, Wait
        key = actionMap[actionId]

        if key == 37:
            settings['keydown'] = True
            doodle['playerDir'] = -1
            doodle['dx'] = -3
        elif key == 39:
            settings['keydown'] = True
            doodle['playerDir'] = 1
            doodle['dx'] = 3
        else:
            settings['keydown'] = False

    def get_features(self):
        """Получение признаков для текущего состояния игры, необходимых для нейросети.

        Returns:
            numpy.ndarray: Список признаков, представляющих текущее состояние игры.
        """
        doodle, platforms, settings = self.doodle, self.platforms, self.settings
        points = np.array([(p['x'], p['y']) for p in platforms])

        sensor_x = (self.sensor_web[:, 0] + doodle['x']) % settings['width']
        sensor_y = np.clip(self.sensor_web[:, 1] + doodle['y'], 1, settings['height'] - 1)

        xx = sensor_x.reshape(-1, 1) - points[:, 0]
        yy = sensor_y.reshape(-1, 1) - points[:, 1]
        cond1 = (xx - settings['platformWidth']) < 0
        cond2 = (xx) > 0
        cond3 = (yy - settings['platformHeight']) < 0
        cond4 = (yy) > 0

        sensors = ((cond1 & cond2 & cond3 & cond4).any(axis=1)).astype(float)
        return np.concatenate([sensors, [doodle['dx'], doodle['dy'], 1]])

    def get_one(self, h1=5, n_classes=3):
        """Генерация случайных весов для нейросети.

        Args:
            h1 (int): Количество нейронов в скрытом слое (по умолчанию 5).
            n_classes (int): Число классов для вывода (по умолчанию 3).

        Returns:
            tuple: Две матрицы весов для нейросети.
        """
        W = np.random.normal(size=(self.sensor_web.shape[0] + 3, h1))
        W2 = np.random.normal(size=(h1, n_classes))
        return W, W2

    @staticmethod
    def softmax(x):
        """Вычисляет softmax для заданного массива.

        Args:
            x (numpy.ndarray): Входной массив.

        Returns:
            numpy.ndarray: Нормализованный выходной массив с вероятностями.
        """
        xe = np.exp(x - x.max())
        return xe / xe.sum()

    def get_action(self, weights):
        """Определение действия на основе весов и текущих признаков.

        Args:
            weights (tuple): Матрицы весов для нейросети.

        Returns:
            int: Идентификатор действия (индекс соответствующего действия).
        """
        W, W2 = weights
        logits = np.maximum(W.T.dot(self.get_features()), 0).dot(W2)
        return np.random.choice(np.arange(len(logits)), p=self.softmax(logits))

    def get_score(self, weights, patience=100, return_actions=False):
        """Получение оценки производительности текущих параметров нейросети.

        Args:
            weights (tuple): Матрицы весов для нейросети.
            patience (int): Количество шагов до уменьшения максимального счёта (по умолчанию 100).
            return_actions (bool): Если True, возвращает действия и координаты (по умолчанию False).

        Returns:
            int/tuple: Текущий счёт или кортеж (действия, координаты, минимальное расстояние до платформ).
        """
        self.restart()
        maxScore_patience = patience
        maxScore_prev = self.settings['minPlatformSpace']
        actions = []
        xcoords = []
        action = self.get_action(weights)
        for _ in range(int(5e4)):
            if self.loop() == -1:
                break
            if np.random.random() < 0.25:
                action = self.get_action(weights)
            actions.append(action)
            xcoords.append(self.doodle['x'])
            self.apply_action(action)
            if self.settings['minPlatformSpace'] > maxScore_prev:
                maxScore_prev = self.settings['minPlatformSpace']
                maxScore_patience = patience
            maxScore_patience -= 1
            if maxScore_patience < 0:
                break
        if return_actions:
            return actions, xcoords, self.settings['minPlatformSpace']
        return self.settings['minPlatformSpace']

    def mutate(self, weights, mutation_rate=0.01):
        """Мутация параметров нейросети с заданной вероятностью.

        Args:
            weights (tuple): Матрицы весов для нейросети.
            mutation_rate (float): Вероятность мутации (по умолчанию 0.01).

        Returns:
            tuple: Обновлённые матрицы весов после мутации.
        """
        W, W2 = weights
        dW, dW2 = self.get_one()
        dM, dM2 = self.get_one()
        return W + dW * (dM > 0) * mutation_rate, W2 + dW2 * (dM2 > 0) * mutation_rate

    def crossover(self, W1, W2):
        """Кроссовер параметров между двумя нейросетями.

        Args:
            W1 (numpy.ndarray): Матрица весов первой нейросети.
            W2 (numpy.ndarray): Матрица весов второй нейросети.

        Returns:
            list: Список с новой матрицей весов.
        """
        result = []
        for w1, w2 in zip(W1, W2):
            maskW = np.random.rand(*w1.shape) < 0.5
            result.append(w1 * maskW + w2 * (~maskW))
        return result

    def generate_random(self, population, size):
        """Генерация случайной популяции.

        Args:
            population (list): Текущая популяция.
            size (int): Размер для новой популяции.

        Returns:
            list: Новая популяция с случайными параметрами.
        """
        new_population = []
        for _ in range(size):
            if np.random.random() < 0.5:
                new_population.append(self.get_one())
            else:
                new_population.append(self.mutate(population[0]))
        return new_population

    def selection(self, population, scores, topK=2):
        """Отбор лучших кандидатов в популяции на основе их оценок.

        Args:
            population (list): Текущая популяция.
            scores (list): Оценки для каждого индивидуума в популяции.
            topK (int): Количество лучших кандидатов для выбора (по умолчанию 2).

        Returns:
            list: Новый список отобранных индивидов.
        """
        scores = np.array(scores).astype(np.float32)
        scores /= scores.sum()
        elitismTopK = np.argsort(scores)[::-1][:topK // 2]
        roulleteTopK = np.random.choice(len(scores), p=scores, size=topK // 2)

        new_population = [tuple(map(lambda x: np.copy(x), population[i]))
                          for i in elitismTopK] + \
                         [tuple(map(lambda x: np.copy(x), population[i]))
                          for i in roulleteTopK]
        return new_population

    def breed(self, population, scores, nChilds=10):
        """Вывод потомства на основе текущей популяции.

        Args:
            population (list): Текущая популяция.
            scores (list): Оценки для каждого индивидуума.
            nChilds (int): Количество создаваемых потомков (по умолчанию 10).

        Returns:
            list: Новый список потомства.
        """
        scores = np.array(scores).astype(np.float32)
        scores /= scores.sum()
        parents = np.random.choice(len(scores), p=scores, size=(nChilds, 2))

        new_population = []
        for parentA, parentB in parents:
            new_population.append(self.mutate(self.crossover(population[parentA], population[parentB])))
        return new_population

    def get_new_population(self, population, scores, topK=4, randomNum=10):
        """Получение новой популяции на основе текущей популяции.

        Args:
            population (list): Текущая популяция.
            scores (list): Оценки для каждого индивидуума.
            topK (int): Количество лучших кандидатов (по умолчанию 4).
            randomNum (int): Количество случайных особей в новой популяции (по умолчанию 10).

        Returns:
            list: Новая популяция.
        """
        return self.factorize(
            self.selection(population, scores, topK) +
            self.breed(population, scores, nChilds=max(0, len(population) - randomNum - topK)) +
            self.generate_random(population, randomNum)
        )

    @staticmethod
    def factorize(population, factor=3):
        """Факторизация популяции с заданным множителем.

        Args:
            population (list): Текущая популяция.
            factor (int): Множитель для сохранения формата массива (по умолчанию 3).

        Returns:
            list: Популяция с обновленными весами.
        """
        for i, p in enumerate(population):
            population[i] = tuple([np.array([[int(10 ** factor * w) / 10 ** factor for w in W] for W in pp])
                                   for pp in p])
        return population

    # @timeit
    def get_scores(self, population, patience=100):
        """Получение оценок для всей популяции.

        Args:
            population (list): Текущая популяция.
            patience (int): Количество шагов до уменьшения максимального счёта (по умолчанию 100).

        Returns:
            list: Список оценок для каждого индивидуума в популяции.
        """
        return [self.get_score(W, patience) for W in population]

    def save_thingey(self, best_thingey, score):
        """Сохранение лучших параметров нейросети в файл.

        Args:
            best_thingey (list): Лучшие параметры нейросети.
            score (float): Лучший счёт.
        """
        # Создание директории models, если она не существует
        os.makedirs('models', exist_ok=True)

        # Формирование имени файла на основе уникального имени эксперимента
        file_name = os.path.join('models', f'weights_{self.unique_experiment_name}.js')
        with open(file_name, 'w') as f:
            f.write('var sensorWeb = %s;\n\nvar W = %s;\n\nvar W2 = %s;\n' % (
                json.dumps([[int(w) for w in W] for W in self.sensor_web]),
                json.dumps([[int(1e5 * w) / 1e5 for w in W] for W in best_thingey[0]]),
                json.dumps([[int(1e5 * w) / 1e5 for w in W] for W in best_thingey[1]])
            ))
        print(f"Модель сохранена в {file_name}")

    def train(self, population_size=64, random_size=20, elite_size=4, num_generations=100, num_repeats=3, num_restarts=5):
        """Обучение с использованием генетического алгоритма.

        Args:
            population_size (int): Размер популяции (по умолчанию 64).
            random_size (int): Размер случайной популяции (по умолчанию 20).
            elite_size (int): Количество лучших особей для сохранения (по умолчанию 4).
            num_generations (int): Количество поколений для обработки (по умолчанию 100).
            num_repeats (int): Количество повторений для оценки (по умолчанию 3).
            num_restarts (int): Количество перезапусков алгоритма (по умолчанию 5).
        """
        PATIENCE = lambda x: 100 * ((x + 2) // 2)
        # Логирование гиперпараметров
        hparams = {
            'population_size': population_size,
            'random_size': random_size,
            'elite_size': elite_size,
            'num_generations': num_generations,
            'num_repeats': num_repeats,
            'num_restarts': num_restarts,
        }
        best_thingey = None
        best_score = 0

        for n_restart in range(num_restarts):
            print('=' * 50)
            print('Cтарт перезапуска №%d' % (n_restart + 1))
            print('Лучшая пока что: %.1f' % best_score)
            print('=' * 50)
            population = [self.get_one() for _ in range(population_size)]
            for generation in range(num_generations):
                scores = 0.
                for _ in range(num_repeats):
                    scores = scores + np.array(self.get_scores(population, PATIENCE(generation))) ** 4
                scores = scores / num_repeats

                population = self.get_new_population(population, scores, topK=elite_size, randomNum=random_size)
                bscore = max(scores) ** 0.25
                # Логирование точности обучения
                self.writer.add_scalar('Accuracy', bscore, n_restart * num_generations + generation)
                if bscore > best_score:
                    best_score = bscore
                    best_thingey = [np.copy(x) for x in population[0]]
                    print('Рестарт: %d\tПоколение: %d\tЗначение: %.1f' % (n_restart + 1, generation, bscore))
                    # if bscore > 100:
                    #     self.save_thingey(best_thingey, best_score)

        self.save_thingey(best_thingey, best_score)
        self.log_hparams_and_metrics(hparams, best_score)

def parse_arguments():
    """Парсит аргументы командной строки для настройки параметров запуска игры Doodle Jump.

    Returns:
        Namespace: Объект с аргументами командной строки.
    """
    parser = argparse.ArgumentParser(description='Настройки для DoodleJump AI оптимизации.')
    parser.add_argument('-p', '--population_size', type=int, default=64, help='Размер популяции.')
    parser.add_argument('-r', '--random_size', type=int, default=20, help='Размер случайной популяции.')
    parser.add_argument('-e', '--elite_size', type=int, default=4, help='Размер выборки элиты.')
    parser.add_argument('-g', '--num_generations', type=int, default=100, help='Количество поколений.')
    parser.add_argument('-t', '--num_repeats', type=int, default=3, help='Количество повторений для усреднения.')
    parser.add_argument('-s', '--num_restarts', type=int, default=5, help='Количество перезапусков алгоритма.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    game = DoodleJumpGame()
    
    game.train(
        population_size=args.population_size,
        random_size=args.random_size,
        elite_size=args.elite_size,
        num_generations=args.num_generations,
        num_repeats=args.num_repeats,
        num_restarts=args.num_restarts
    )
