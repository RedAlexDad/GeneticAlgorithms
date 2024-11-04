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

class DoodleJumpGame:
    def __init__(self, width=375, height=667, platform_width=65, platform_height=20, experiment_name='doodlejump', device=None):
        self.settings = self.init_settings(width, height, platform_width, platform_height)
        self.doodle, self.platforms = self.init_states()
        self.sensor_web = self.create_sensor_web()
        
        self.device = self.get_device(device)
        print(f'Использованное устройство: {self.device}')
        # Инициализация логгера
        self.writer = self.initialize_log_dir(experiment_name)
    
    @staticmethod
    def get_device(select=None):
        return torch.device('cuda' if (select in [None, 'cuda'] and torch.cuda.is_available()) else 'cpu')

    def initialize_log_dir(self, experiment_name):
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

        # Логирование гиперпараметров и метрик
        self.writer.add_hparams(hparams, {'best_score': best_score}, run_name=run_name)

    def init_settings(self, width, height, platform_width, platform_height):
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
        sensor_web = np.meshgrid(
            np.arange(-self.settings['width'] * 2 // 3, self.settings['width'] * 2 // 3, 50),
            np.arange(-self.settings['height'] * 2 // 3, self.settings['height'] * 2 // 3, 75)
        )
        return np.concatenate([sensor_web[0].flatten()[:, None], sensor_web[1].flatten()[:, None]], axis=1)

    def restart(self):
        self.doodle, self.platforms = self.init_states()

    def loop(self):
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
        doodle, platforms, settings = self.doodle, self.platforms, self.settings
        points = np.array([(p['x'], p['y']) for p in platforms], dtype=np.float32)

        sensor_x = (self.sensor_web[:, 0] + doodle['x']) % settings['width']
        sensor_y = np.clip(self.sensor_web[:, 1] + doodle['y'], 1, settings['height'] - 1)

        xx = sensor_x.reshape(-1, 1) - points[:, 0]
        yy = sensor_y.reshape(-1, 1) - points[:, 1]
        cond1 = (xx - settings['platformWidth']) < 0
        cond2 = xx > 0
        cond3 = (yy - settings['platformHeight']) < 0
        cond4 = yy > 0

        sensors = ((cond1 & cond2 & cond3 & cond4).any(axis=1)).astype(np.float32)
        features = np.concatenate([sensors, [doodle['dx'], doodle['dy'], 1]], dtype=np.float32)
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def get_one(self, h1=5, n_classes=3):
        W = torch.randn((self.sensor_web.shape[0] + 3, h1), dtype=torch.float32, device=self.device)
        W2 = torch.randn((h1, n_classes), dtype=torch.float32, device=self.device)
        return W, W2

    @staticmethod
    def softmax(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)  # Убедить, что x находится на нужном устройстве
        xe = torch.exp(x - x.max())
        return xe / xe.sum()
    
    def get_action(self, weights):
        W, W2 = [w.to(self.device) for w in weights]  # Убедить, что оба веса на нужном устройстве
        features = self.get_features()
        logits = torch.relu(features.matmul(W)).matmul(W2)
        probas = self.softmax(logits)
        return np.random.choice(len(probas), p=probas.cpu().numpy())  # Конвертация tензора обратно в NumPy массив только для вызова choice

    def get_score(self, weights, patience=100, return_actions=False):
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
        W, W2 = weights
        # Убедитесь, что `get_one` возвращает тензоры на нужном устройстве
        dW, dW2 = self.get_one()
        dM, dM2 = self.get_one()
        
        # Переносим все тензоры на одно устройство
        W, W2 = W.to(self.device), W2.to(self.device)
        dW, dW2 = dW.to(self.device), dW2.to(self.device)
        dM, dM2 = dM.to(self.device), dM2.to(self.device)

        return W + dW * (dM > 0).float() * mutation_rate, W2 + dW2 * (dM2 > 0).float() * mutation_rate
   
    def crossover(self, W1, W2):
        result = []
        for w1, w2 in zip(W1, W2):
            # Перемещаем все тензоры на одно устройство
            w1 = w1.to(self.device)
            w2 = w2.to(self.device)
            maskW = torch.rand(w1.shape, device=self.device) < 0.5
            result.append(w1 * maskW + w2 * (~maskW.to(torch.bool)))
        return result

    def generate_random(self, population, size):
        new_population = []
        for _ in range(size):
            if np.random.random() < 0.5:
                new_population.append(self.get_one())
            else:
                new_population.append(self.mutate(population[0]))
        return new_population

    def selection(self, population, scores, topK=2):
        scores = np.array(scores).astype(np.float32)
        scores = scores / scores.sum()
        elitismTopK = np.argsort(scores)[::-1][:topK // 2]
        roulleteTopK = np.random.choice(len(scores), p=scores, size=topK // 2)

        new_population = [tuple(map(lambda x: np.copy(x.cpu()), population[i])) for i in elitismTopK] + \
                         [tuple(map(lambda x: np.copy(x.cpu()), population[i])) for i in roulleteTopK]
        return new_population

    def breed(self, population, scores, nChilds=10):
        scores = np.array(scores).astype(np.float32)
        scores = scores / scores.sum()
        parents = np.random.choice(len(scores), p=scores, size=(nChilds, 2))

        new_population = []
        for parentA, parentB in parents:
            new_population.append(self.mutate(self.crossover(population[parentA], population[parentB])))
        return new_population

    def get_new_population(self, population, scores, topK=4, randomNum=10):
        return self.factorize(
            self.selection(population, scores, topK) +
            self.breed(population, scores, nChilds=max(0, len(population) - randomNum - topK)) +
            self.generate_random(population, randomNum)
        )

    @staticmethod
    def factorize(population, factor=3):
        for i, p in enumerate(population):
            population[i] = tuple([torch.tensor([[int(10 ** factor * w) / 10 ** factor for w in W] for W in pp]) for pp in p])
        return population

    @timeit
    def get_scores(self, population, patience=100):
        return [self.get_score(W, patience) for W in population]

    def save_thingey(self, best_thingey, score):
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
    parser = argparse.ArgumentParser(description='Настройки для DoodleJump AI оптимизации.')
    parser.add_argument('-p', '--population_size', type=int, default=64, help='Размер популяции.')
    parser.add_argument('-r', '--random_size', type=int, default=20, help='Размер случайной популяции.')
    parser.add_argument('-e', '--elite_size', type=int, default=4, help='Размер выборки элиты.')
    parser.add_argument('-g', '--num_generations', type=int, default=100, help='Количество поколений.')
    parser.add_argument('-t', '--num_repeats', type=int, default=3, help='Количество повторений для усреднения.')
    parser.add_argument('-s', '--num_restarts', type=int, default=5, help='Количество перезапусков алгоритма.')
    parser.add_argument('-d', '--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Устройство вычисления: cpu или cuda')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    game = DoodleJumpGame(device=args.device)
    
    game.train(
        population_size=args.population_size,
        random_size=args.random_size,
        elite_size=args.elite_size,
        num_generations=args.num_generations,
        num_repeats=args.num_repeats,
        num_restarts=args.num_restarts
    )
