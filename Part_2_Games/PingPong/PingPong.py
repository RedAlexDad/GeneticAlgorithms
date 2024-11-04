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
        return torch.device('cuda' if (select in [None, 'cuda'] and torch.cuda.is_available()) else 'cpu')

    def initialize_log_dir(self, experiment_name):
        """Создает уникальную директорию для эксперимента и инициализирует SummaryWriter."""
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
        # Получение текущего времени в читаемом формате
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

        # Логирование гиперпараметров и метрик
        self.writer.add_hparams(hparams, {'best_score': best_score}, run_name=run_name)
        
    def init_paddle(self, x_position):
        return {
            'x': x_position,
            'y': self.height / 2 - self.paddle_height / 2,
            'width': self.grid,
            'height': self.paddle_height,
            'dy': 0
        }

    def init_ball(self):
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
        return (
            obj1['x'] < obj2['x'] + obj2['width'] and
            obj1['x'] + obj1['width'] > obj2['x'] and
            obj1['y'] < obj2['y'] + obj2['height'] and
            obj1['y'] + obj1['height'] > obj2['y']
        )

    def restart(self):
        self.ball.update({'resetting': False, 'x': self.width / 2, 'y': self.height / 2, 'score': 0})
        self.left_paddle.update({'y': self.height / 2 - self.paddle_height / 2})
        self.right_paddle.update({'y': self.height / 2 - self.paddle_height / 2})

    def loop(self):
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
        return torch.normal(mean=0.0, std=1.0, size=(15, 6), device=self.device)

    def get_action(self, W):
        features = torch.tensor(self.get_features(), device=self.device, dtype=torch.float32)
        return (W.t().matmul(features)).argmax().item()

    def get_score(self, W, patience=100):
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
        dW = self.get_one()
        dM = self.get_one() > 0
        return W + dW * dM * mutation_rate

    def crossover(self, W1, W2):
        maskW = torch.rand_like(W1, device=W1.device) < 0.5
        return W1 * maskW + W2 * (~maskW)

    def generate_random(self, population, size):
        new_population = []
        for _ in range(size):
            if np.random.random() < 0.5:
                new_population.append(self.get_one())
            else:
                new_population.append(self.mutate(population[0]))
        return new_population

    def selection(self, population, scores, topK=2):
        scores = np.array(scores) * 1.0
        scores = scores / scores.sum()
        
        elitismTopK = np.argsort(scores)[::-1][:topK // 2]
        roulleteTopK = np.random.choice(len(scores), p=scores, size=topK // 2)

        return [population[i].clone() for i in elitismTopK] + \
               [population[i].clone() for i in roulleteTopK]

    def breed(self, population, scores, nChilds=10):
        scores = np.array(scores) * 1.0
        scores /= scores.sum()
        
        parents = np.random.choice(len(scores), p=scores, size=(nChilds, 2))

        return [self.mutate(self.crossover(population[pA], population[pB])) for pA, pB in parents]

    def get_new_population(self, population, scores, topK=4, randomNum=10):
        return (
            self.selection(population, scores, topK) +
            self.breed(population, scores, nChilds=max(0, len(population) - randomNum - topK)) +
            self.generate_random(population, randomNum)
        )

    # @timeit
    def get_scores(self, population, patience=100):
        return [self.get_score(W, patience) for W in population]
    
    def train(self, population_size=128, random_size=20, elite_size=5, num_generations=100, num_repeats=3, num_restarts=5):
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