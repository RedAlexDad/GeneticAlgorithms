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


class SnakeOptimizer:
    def __init__(self, grid_size=16, width=400, height=400, device=None, experiment_name='snake'):
        self.grid = grid_size
        self.width = width
        self.height = height
        self.snake, self.apple = self.restart()
        
        self.device = self.get_device(device)
        print(f'Использованное устройство: {self.device}')
        # Инициализация логгера
        self.writer = self.initialize_log_dir(experiment_name)
        
    @staticmethod
    def get_device(select=None):
        if select is None or select == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device('cpu')

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
        run_name = f"run_{timestamp}"  # Читаемое имя

        # Логирование гиперпараметров и метрик
        self.writer.add_hparams(hparams, {
            'best_score': best_score,
        }, run_name=run_name)
        
    def generate_apple(self):
        # Генерация нового местоположения для яблока
        self.apple['x'] = random.randint(0, 25) * self.grid
        self.apple['y'] = random.randint(0, 25) * self.grid

    def loop(self):
        # Обновление положения змейки
        self.snake['x'] += self.snake['dx']
        self.snake['y'] += self.snake['dy']

        # Обработка выхода змейки за границы поля
        if self.snake['x'] < 0:
            self.snake['x'] = self.width - self.grid
        elif self.snake['x'] >= self.width:
            self.snake['x'] = 0

        if self.snake['y'] < 0:
            self.snake['y'] = self.height - self.grid
        elif self.snake['y'] >= self.height:
            self.snake['y'] = 0

        # Обновление ячеек тела змейки
        self.snake['cells'] = [(self.snake['x'], self.snake['y'])] + self.snake['cells']

        if len(self.snake['cells']) > self.snake['maxCells']:
            self.snake['cells'].pop()

        for index, cell in enumerate(self.snake['cells']):
            # Проверка, съедено ли яблоко
            if cell[0] == self.apple['x'] and cell[1] == self.apple['y']:
                self.snake['maxCells'] += 1
                self.generate_apple()

            for i in range(index + 1, len(self.snake['cells'])):
                # Проверка на столкновение с собой
                if (cell[0] == self.snake['cells'][i][0] and
                    cell[1] == self.snake['cells'][i][1]):
                    return -1  # Перезапуск игры

        return 0

    def restart(self):
        # Инициализация змейки и яблока
        snake = dict(x=160, y=160, dx=self.grid, dy=0, cells=[], maxCells=4)
        apple = dict(x=320, y=320)
        return snake, apple

    def apply_action(self, actionId):
        actionMap = {0: 37, 1: 38, 2: 39, 3: 40}
        key = actionMap[actionId]

        # Обновление направления движения змейки на основе действий
        if (key == 37 and self.snake['dx'] == 0):  # Влево
            self.snake['dx'] = -self.grid
            self.snake['dy'] = 0
        elif (key == 38 and self.snake['dy'] == 0):  # Вверх
            self.snake['dx'] = 0
            self.snake['dy'] = -self.grid
        elif (key == 39 and self.snake['dx'] == 0):  # Вправо
            self.snake['dx'] = self.grid
            self.snake['dy'] = 0
        elif (key == 40 and self.snake['dy'] == 0):  # Вниз
            self.snake['dx'] = 0
            self.snake['dy'] = self.grid

    def get_features(self):
        # Получение признаков для текущего состояния игры
        sensors = [
            np.sign(self.snake['dx']),
            np.sign(self.snake['dy']),
            (self.snake['x'] - self.snake['cells'][-1][0])/self.width if len(self.snake['cells']) else 0,
            (self.snake['y'] - self.snake['cells'][-1][1])/self.height if len(self.snake['cells']) else 0,
            self.snake['x'] == self.apple['x'],
            self.snake['y'] == self.apple['y'],
            (self.snake['x'] - self.apple['x'])/self.width > 0,
            (self.snake['x'] - self.apple['x'])/self.width < 0,
            (self.snake['y'] - self.apple['y'])/self.height > 0,
            (self.snake['y'] - self.apple['y'])/self.height < 0,
            any([(self.snake['x'] == cell[0] and self.snake['dy'] == 0) for cell in self.snake['cells'][1:]]),
            any([(self.snake['y'] == cell[1] and self.snake['dx'] == 0) for cell in self.snake['cells'][1:]]),
            any([(self.snake['x'] == cell[0] and self.snake['dy'] > 0) for cell in self.snake['cells'][1:]]),
            any([(self.snake['y'] == cell[1] and self.snake['dx'] > 0) for cell in self.snake['cells'][1:]]),
            any([(self.snake['x'] == cell[0] and self.snake['dy'] < 0) for cell in self.snake['cells'][1:]]),
            any([(self.snake['y'] == cell[1] and self.snake['dx'] < 0) for cell in self.snake['cells'][1:]]),
        ]
        return sensors

    def get_one(self):
        # Генерация случайной матрицы весов и смещений для нейросети
        W = np.random.normal(size=(16, 4))
        b = np.random.normal(size=(4,))
        return W, b

    def getAction(self, W, b):
        # Определение действия на основе признаков и параметров нейросети
        return (W.T.dot(self.get_features()) + b).argmax()

    def get_score(self, W, b, patience=100):
        # Получение оценки производительности текущих параметров
        self.snake, self.apple = self.restart()
        maxCells_patience = patience
        maxCells_prev = self.snake['maxCells']
        while self.loop() != -1:
            self.apply_action(self.getAction(W, b))
            if self.snake['maxCells'] > maxCells_prev:
                maxCells_prev = self.snake['maxCells']
                maxCells_patience = patience
            maxCells_patience -= 1
            if maxCells_patience < 0:
                self.snake['maxCells'] /= 2
                break
        return self.snake['maxCells']

    def mutate(self, W, b, mutation_rate=0.02):
        # Мутация параметров нейросети с заданной вероятностью
        dW, db = self.get_one()
        dWM, dbM = self.get_one()
        return (W + dW * (dWM > 0) * mutation_rate,
                b + db * (dbM > 0) * mutation_rate)

    def crossover(self, W1, b1, W2, b2):
        # Кроссовер параметров между двумя нейросетями
        maskW = np.random.random(W1.shape) < 0.5
        maskb = np.random.random(b1.shape) < 0.5
        return W1 * maskW + W2 * (~maskW), b1 * maskb + b2 * (~maskb)

    def generate_random(self, population, size):
        # Генерация случайной популяции
        new_population = []
        for _ in range(size):
            if np.random.random() < 0.5:
                new_population.append(self.get_one())
            else:
                new_population.append(self.mutate(*population[0]))
        return new_population

    def selection(self, population, scores, topK=2):
        # Отбор лучших кандидатов в популяции
        scores = np.array(scores) * 1.0
        scores /= scores.sum()
        elitismTopK = np.argsort(scores)[::-1][:topK // 2]
        roulleteTopK = np.random.choice(len(scores), p=scores, size=topK // 2)

        new_population = [tuple(map(lambda x: x.copy(), population[i])) for i in elitismTopK] + \
                         [tuple(map(lambda x: x.copy(), population[i])) for i in roulleteTopK]

        return new_population

    def breed(self, population, scores, nChilds=10):
        # Вывод потомства на основе текущей популяции
        scores = np.array(scores) * 1.0
        scores /= scores.sum()
        parents = np.random.choice(len(scores), p=scores, size=(nChilds, 2))

        new_population = []
        for parentA, parentB in parents:
            new_population.append(self.mutate(*self.crossover(*population[parentA], *population[parentB])))

        return new_population

    def get_new_population(self, population, scores, topK=4, randomNum=10):
        # Получение новой популяции
        return (self.selection(population, scores, topK) + 
                self.breed(population, scores, nChilds=max(0, len(population) - randomNum - topK)) +
                self.generate_random(population, randomNum))

    def get_scores(self, population, patience=100):
        # Получение оценок для всей популяции
        scores = [self.get_score(W, b, patience) for W, b in population]
        return scores
    
    def train(self, population_size=64, num_generations=10, num_repeats=3, num_restarts=5):
        PATIENCE = lambda x: 100 * ((x + 5) // 5)
        best_thingey = None
        best_score = 0
        hparams = {
            'population_size': population_size,
            'num_generations': num_generations,
            'num_repeats': num_repeats,
            'num_restarts': num_restarts
        }

        for n_restart in range(num_restarts):
            print('=' * 50)
            print('Старт перезапуска №%d' % (n_restart + 1))
            print('Лучшая пока что: %.1f' % best_score)
            print('=' * 50)

            population = [self.get_one() for _ in range(population_size)]

            for generation in range(num_generations):
                scores = 0
                for _ in range(num_repeats):
                    scores += np.array(self.get_scores(population, PATIENCE(generation)))
                scores /= num_repeats
                bscore = max(scores)
                # Логирование точности
                self.writer.add_scalar('Accuracy', bscore, n_restart * num_generations + generation)
                scores **= 4
                population = self.get_new_population(population, scores, topK=5, randomNum=20)
                if bscore > best_score:
                    best_score = bscore
                    best_thingey = np.concatenate([population[0][0], [population[0][1]]])
                    print('Рестарт: %d\tПоколение: %d\tЗначение: %.1f' % (n_restart + 1, generation, bscore))

        # Записываем параметры и лучшие результаты
        self.log_hparams_and_metrics(hparams, best_score)

        # Сохраняем лучшие веса и смещения
        with open('snake_weights.js', 'w') as f:
            f.write('var W = %s;\n'%(json.dumps([[int(1e5*w)/1e5 for w in W] for W in best_thingey])))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Запуск оптимизатора змейки с указанными параметрами.')
    parser.add_argument('-p', '--population_size', type=int, default=64, help='Размер популяции.')
    parser.add_argument('-g', '--num_generations', type=int, default=10, help='Количество поколений для обработки.')
    parser.add_argument('-r', '--num_repeats', type=int, default=3, help='Количество повторений для надежности.')
    parser.add_argument('-n', '--num_restarts', type=int, default=5, help='Количество перезапусков в оптимизации.')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Парсинг аргументов командной строки
    args = parse_arguments()
    optimizer = SnakeOptimizer()

    # Извлечение параметров из аргументов
    optimizer.train(
        population_size=args.population_size,
        num_generations=args.num_generations,
        num_repeats=args.num_repeats,
        num_restarts=args.num_restarts
    )
