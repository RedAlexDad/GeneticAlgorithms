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

class PingPongGame:
    def __init__(self, width=750, height=585, grid_size=15, experiment_name='pingpong', device=None):
        """Инициализация объекта PingPongGame.

        Args:
            width (int): Ширина игрового поля (по умолчанию 750).
            height (int): Высота игрового поля (по умолчанию 585).
            grid_size (int): Размер клетки (по умолчанию 15).
            experiment_name (str): Название эксперимента (по умолчанию 'pingpong').
            device (str, optional): Устройство ('cpu' или 'cuda') для запуска.
        """
        self.width = width
        self.height = height
        self.grid = grid_size
        self.paddle_height = self.grid * 5
        self.max_paddle_y = self.height - self.grid - self.paddle_height
        self.paddle_speed = 6
        self.ball_speed = 5

        self.left_paddle = self.init_paddle(self.grid * 2)
        self.right_paddle = self.init_paddle(self.width - self.grid * 3)
        self.ball = self.init_ball()
        
        self.device = self.get_device(device)
        print(f'Использованное устройство: {self.device}')
        # Инициализация логгера
        self.writer = self.initialize_log_dir(experiment_name)

    @staticmethod
    def get_device(select=None):
        """Определяет устройство ('cpu' или 'cuda') на основе доступности графического процессора.

        Args:
            select (str, optional): Выбор устройства ('cpu', 'cuda'). Если None, выбирается 'cuda', если доступен.

        Returns:
            torch.device: TensorFlow устройство.
        """
        return torch.device('cuda' if (select in [None, 'cuda'] and torch.cuda.is_available()) else 'cpu')

    def initialize_log_dir(self, experiment_name):
        """Создает уникальную директорию для эксперимента и инициализирует SummaryWriter.

        Args:
            experiment_name (str): Название эксперимента.

        Returns:
            SummaryWriter: Инициализированный объект SummaryWriter для логирования.
        """
        self.experiment_name = f'{experiment_name}' # Сохраняем название эксперимента
        self.log_dir = f'logs/{experiment_name}'  # Базовый путь для логов
        # Проверка существования директории и генерация нового имени, если необходимо
        i = 0
        while os.path.exists(self.log_dir):
            self.log_dir = f'logs/{experiment_name}_{i}'
            i += 1
        os.makedirs(self.log_dir)   # Создаем директорию, если она не существует
        # Вывод информации о названии эксперимента
        print(f"Директория для эксперимента '{experiment_name}' инициализирована по пути: {self.log_dir}")
        # Сохраняем уникальное имя для использования в save_model
        self.unique_experiment_name = os.path.basename(self.log_dir)  

        return SummaryWriter(log_dir=self.log_dir) # Возвращаем инициализированный SummaryWriter
  
    def log_hparams_and_metrics(self, hparams, best_score):
        """Логирует гиперпараметры и лучший результат в TensorBoard с временной меткой.

        Args:
            hparams (dict): Гиперпараметры, используемые в эксперименте.
            best_score (float): Лучший результат текущего эксперимента.
        """
        # Получение текущего времени в читаемом формате
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

        # Логирование гиперпараметров и метрик
        self.writer.add_hparams(hparams, {'best_score': best_score}, run_name=run_name)
        
    def init_paddle(self, x_position):
        """Инициализация ракетки на заданной позиции.

        Args:
            x_position (int): Позиция по оси X для ракетки.

        Returns:
            dict: Словарь с параметрами ракетки (позиция, размеры и скорость).
        """
        return {
            'x': x_position,
            'y': self.height / 2 - self.paddle_height / 2,
            'width': self.grid,
            'height': self.paddle_height,
            'dy': 0
        }

    def init_ball(self):
        """Инициализация мяча с начальным состоянием.

        Returns:
            dict: Словарь с параметрами мяча (позиция, размеры, скорость и счёт).
        """
        return {
            'x': self.width / 2,
            'y': self.height / 2,
            'width': self.grid,
            'height': self.grid,
            'resetting': False,
            'dx': self.ball_speed,
            'dy': -self.ball_speed,
            'score': 0
        }

    def collides(self, obj1, obj2):
        """Проверка столкновения между двумя объектами.

        Args:
            obj1 (dict): Первый объект.
            obj2 (dict): Второй объект.

        Returns:
            bool: True если произошло столкновение, иначе False.
        """
        return (
            obj1['x'] < obj2['x'] + obj2['width'] and
            obj1['x'] + obj1['width'] > obj2['x'] and
            obj1['y'] < obj2['y'] + obj2['height'] and
            obj1['y'] + obj1['height'] > obj2['y']
        )

    def restart(self):
        """Сбрасывает состояние игры к начальному.

        Возможные изменения:
            Сбрасывает мяча и ракетки к центральной позиции и обнулит счёт.
        """
        self.ball.update({'resetting': False, 'x': self.width / 2, 'y': self.height / 2, 'score': 0})
        self.left_paddle.update({'y': self.height / 2 - self.paddle_height / 2})
        self.right_paddle.update({'y': self.height / 2 - self.paddle_height / 2})

    def loop(self):
        """Основной игровой цикл, включает обновление положения объектов и проверку коллизий.

        Returns:
            int: -1 если игра закончилась (мяч вышел за пределы), 0 если игра продолжается.
        """
        self.left_paddle['y'] += self.left_paddle['dy']
        self.right_paddle['y'] += self.right_paddle['dy']

        self.left_paddle['y'] = max(self.left_paddle['y'], self.grid)
        self.left_paddle['y'] = min(self.left_paddle['y'], self.max_paddle_y)

        self.right_paddle['y'] = max(self.right_paddle['y'], self.grid)
        self.right_paddle['y'] = min(self.right_paddle['y'], self.max_paddle_y)

        self.ball['x'] += self.ball['dx']
        self.ball['y'] += self.ball['dy']

        if self.ball['y'] < self.grid or self.ball['y'] > self.height - self.grid:
            self.ball['dy'] *= -1

        if self.ball['x'] < 0 or self.ball['x'] > self.width:
            return -1

        if self.collides(self.ball, self.left_paddle) or self.collides(self.ball, self.right_paddle):
            self.ball['dx'] *= -1
            self.ball['score'] += 1

        return 0

    def apply_action(self, actionId):
        """Применяет действие игрока к ракеткам.

        Args:
            actionId (int): Идентификатор действия (-1 для ожидания, 0-5 для управления ракетками).
        """
        # правый игрок вверх
        # правый игрок вниз
        # левый игрок вверх
        # левый игрок вниз
        # правый игрок ожидает
        # вниз игрок ожидает
        actionMap = {0: 38, 1: 40, 2: 87, 3: 83, 4: -1, 5: -2}
        key = actionMap[actionId]

        if key == 38:
            self.right_paddle['dy'] = -self.paddle_speed
        elif key == 40:
            self.right_paddle['dy'] = +self.paddle_speed
        elif key == -1:
            self.right_paddle['dy'] = 0
        elif key == 87:
            self.left_paddle['dy'] = -self.paddle_speed
        elif key == 83:
            self.left_paddle['dy'] = +self.paddle_speed
        elif key == -2:
            self.left_paddle['dy'] = 0

    def get_features(self):
        """Получение признаков для текущего состояния игры, необходимых для нейросети.

        Returns:
            list: Список признаков, представляющих текущее состояние игры.
        """
        sensors = [
            np.sign(self.left_paddle['y'] - self.ball['y']),
            np.abs(self.left_paddle['y'] - self.ball['y']) / self.height,
            np.abs(self.left_paddle['x'] - self.ball['x']) / self.width,
            np.sign(self.right_paddle['y'] - self.ball['y']),
            np.abs(self.right_paddle['y'] - self.ball['y']) / self.height,
            np.abs(self.right_paddle['x'] - self.ball['x']) / self.width,
            np.sign(self.left_paddle['dy']),
            np.sign(self.left_paddle['dy']) == 0,
            np.sign(self.right_paddle['dy']),
            np.sign(self.right_paddle['dy']) == 0,
            np.sign(self.ball['dx']),
            np.sign(self.ball['dy']),
            np.sign(self.ball['x'] - self.width // 2),
            np.sign(self.ball['y'] - self.height // 2),
            1
        ]
        return sensors

    def get_one(self):
        """Генерация случайной матрицы весов для нейросети.

        Returns:
            torch.Tensor: Сгенерированная матрица весов.
        """
        return torch.normal(mean=0.0, std=1.0, size=(15, 6), device=self.device)

    def get_action(self, W):
        """Определение действия на основе весов и текущих признаков.

        Args:
            W (torch.Tensor): Матрица весов.

        Returns:
            int: Идентификатор действия (индекс соответствующего действия).
        """
        features = torch.tensor(self.get_features(), device=self.device, dtype=torch.float32)
        return (W.t().matmul(features)).argmax().item()

    def get_score(self, W, patience=100):
        """Получение оценки производительности текущих параметров нейросети.

        Args:
            W (torch.Tensor): Матрица весов.
            patience (int): Количество шагов до уменьшения максимального счёта (по умолчанию 100).

        Returns:
            int: Текущий счёт после завершения игры.
        """
        self.restart()
        maxScore_patience = patience
        maxScore_prev = self.ball['score']
        action = self.get_action(W)
        
        for _ in range(int(2e4)):
            if self.loop() == -1:
                break
            if np.random.random() < 0.5:
                action = self.get_action(W)
            self.apply_action(action)
            if self.ball['score'] > maxScore_prev:
                maxScore_prev = self.ball['score']
                maxScore_patience = patience
            maxScore_patience -= 1
            if maxScore_patience < 0:
                break
        return self.ball['score']

    def mutate(self, W, mutation_rate=0.02):
        """Мутация параметров нейросети с заданной вероятностью.

        Args:
            W (torch.Tensor): Матрица весов.
            mutation_rate (float): Вероятность мутации (по умолчанию 0.02).

        Returns:
            torch.Tensor: Новая матрица весов после мутации.
        """
        dW = self.get_one()
        dM = self.get_one() > 0
        return W + dW * dM * mutation_rate

    def crossover(self, W1, W2):
        """Кроссовер параметров между двумя нейросетями.

        Args:
            W1 (torch.Tensor): Матрица весов первой нейросети.
            W2 (torch.Tensor): Матрица весов второй нейросети.

        Returns:
            torch.Tensor: Новая матрица весов после кроссовера.
        """
        maskW = torch.rand_like(W1, device=W1.device) < 0.5
        return W1 * maskW + W2 * (~maskW)

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
        scores = np.array(scores) * 1.0
        scores = scores / scores.sum()
        
        elitismTopK = np.argsort(scores)[::-1][:topK // 2]
        roulleteTopK = np.random.choice(len(scores), p=scores, size=topK // 2)

        return [population[i].clone() for i in elitismTopK] + \
               [population[i].clone() for i in roulleteTopK]

    def breed(self, population, scores, nChilds=10):
        """Вывод потомства на основе текущей популяции.

        Args:
            population (list): Текущая популяция.
            scores (list): Оценки для каждого индивидуума.
            nChilds (int): Количество создаваемых потомков (по умолчанию 10).

        Returns:
            list: Новый список потомства.
        """
        scores = np.array(scores) * 1.0
        scores /= scores.sum()
        
        parents = np.random.choice(len(scores), p=scores, size=(nChilds, 2))

        return [self.mutate(self.crossover(population[pA], population[pB])) for pA, pB in parents]

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
        return (
            self.selection(population, scores, topK) +
            self.breed(population, scores, nChilds=max(0, len(population) - randomNum - topK)) +
            self.generate_random(population, randomNum)
        )

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

    def train(self, population_size=128, random_size=20, elite_size=5, num_generations=100, num_repeats=3, num_restarts=5):
        """Обучение с использованием генетического алгоритма.

        Args:
            population_size (int): Размер популяции (по умолчанию 128).
            random_size (int): Размер случайных особей в популяции (по умолчанию 20).
            elite_size (int): Количество лучших особей для сохранения (по умолчанию 5).
            num_generations (int): Количество поколений для обработки (по умолчанию 100).
            num_repeats (int): Количество повторений для оценки (по умолчанию 3).
            num_restarts (int): Количество перезапусков алгоритма (по умолчанию 5).
        """
        best_thingey = None
        best_score = 0
        PATIENCE = lambda x: 1000 * ((x + 2) // 2)

        for n_restart in range(num_restarts):
            print('=' * 50)
            print('Cтарт перезапуска №%d' % (n_restart + 1))
            print('Лучшая пока что: %.1f' % best_score)
            print('=' * 50)
            population = [self.get_one() for _ in range(population_size)]

            for generation in range(num_generations):
                scores = 1e-10

                for _ in range(num_repeats):
                    scores = scores + np.array(self.get_scores(population, PATIENCE(generation)))

                scores = scores + num_repeats
                bscore = max(scores)

                # Логирование точности обучения
                self.writer.add_scalar('Accuracy', bscore, n_restart * num_generations + generation)

                scores = scores ** 4  # Для усиления разниц в рейтингах
                population = self.get_new_population(population, scores, topK=elite_size, randomNum=random_size)

                if bscore > best_score:
                    best_score = bscore
                    best_thingey = population[0].cpu().numpy()
                    print('Рестарт: %d\tПоколение: %d\tЗначение: %.1f' % (n_restart + 1, generation, bscore))
                        
                    # Освобождение кэшируемой памяти GPU, если используется CUDA
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Log hyperparameters and metrics
        hparams = {
            'population_size': population_size,
            'random_size': random_size,
            'elite_size': elite_size,
            'num_generations': num_generations,
            'num_repeats': num_repeats,
            'num_restarts': num_restarts,
        }
        
        self.log_hparams_and_metrics(hparams, best_score)
        # Создание директории models, если она не существует
        os.makedirs('models', exist_ok=True)
        # Сохраняем лучшие веса и смещения с учетом номера эксперимента
        file_name = f'models/weights_{self.unique_experiment_name}.js'
        with open(file_name, 'w') as f:
            f.write('var W = %s;\n' % (json.dumps([[int(1e5 * w) / 1e5 for w in W] for W in best_thingey])))


def parse_arguments():
    """Парсит аргументы командной строки для настройки параметров запуска игры Ping Pong.

    Returns:
        Namespace: Объект с аргументами командной строки.
    """
    parser = argparse.ArgumentParser(description='Настройки для Pong AI оптимизации.')
    parser.add_argument('-p', '--population_size', type=int, default=128, help='Размер популяции.')
    parser.add_argument('-r', '--random_size', type=int, default=20, help='Размер случайной популяции.')
    parser.add_argument('-e', '--elite_size', type=int, default=5, help='Размер выборки элиты.')
    parser.add_argument('-g', '--num_generations', type=int, default=100, help='Количество поколений.')
    parser.add_argument('-t', '--num_repeats', type=int, default=3, help='Количество повторений для усреднения.')
    parser.add_argument('-s', '--num_restarts', type=int, default=5, help='Количество перезапусков алгоритма.')
    parser.add_argument('-d', '--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Устройство вычисления: cpu или cuda')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    game = PingPongGame(device=args.device)
    
    game.train(
        population_size=args.population_size,
        random_size=args.random_size,
        elite_size=args.elite_size,
        num_generations=args.num_generations,
        num_repeats=args.num_repeats,
        num_restarts=args.num_restarts
    )