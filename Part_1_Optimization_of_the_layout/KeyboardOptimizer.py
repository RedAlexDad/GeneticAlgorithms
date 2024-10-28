import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from functools import wraps
from PIL import Image, ImageDraw, ImageFont


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

class KeyboardOptimizer:
    def __init__(self, messages, font_name='Roboto-Bold.ttf', working_dir=os.getcwd(), device=None):
        self.keyboard_img_path = os.path.join(working_dir, 'keyboard.png')
        self.font_path = os.path.join(working_dir, font_name)
        self.ROWS_CONTENT = [
            list('1234567890'),
            list('йцукенгшщзх'),
            list('фывапролджэ'),
            list('ячсмитьбю'),
            list(', .\n'),
        ]
        self.KEYBINDS = [
            [(68,68), (174, 68), (280, 68), (385, 68), (485, 68), (585, 68), (685, 68), (790, 68), (905, 68), (1010, 68)],
            [(60, 201),(155, 201),(255, 201),(345, 201),(445, 201), (540, 201),(635, 201),(730, 201),(820, 201),(920, 201),(1015, 201)],
            [(60, 350),(155, 350),(255, 350),(345, 350),(445, 350), (540, 350),(635, 350),(730, 350),(820, 350),(920, 350),(1015, 350)],
            [(155, 500),(255, 500),(345, 500),(445, 500), (540, 500),(635, 500),(730, 500),(820, 500),(920, 500)],
            [(224, 645),(530, 645),(855, 645),(980, 645)],
        ]
        self.prepare_data(messages)
        
        self.device = self.get_device(device)
        print(f'Использованное устройство: {self.device}')

    @staticmethod
    def get_device(select=None):
        if select is None or select == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device('cpu')
    
    # @timeit
    def prepare_data(self, messages):
        df = pd.DataFrame({'msg': np.concatenate(messages)})
        df.msg = df.msg.str.lower().replace('ё', 'е').replace(u'\xa0', u' ').replace('-', ' ')
        df.msg = df.msg.replace('[^a-zа-я0-9\s?,.!]', '', regex=True)
        
        self.sequence = list(''.join(df.msg.dropna().values))
        self.charmap = np.unique(self.sequence)

        diffs_mask = df.msg.dropna().str.len().cumsum().values[:-1] - 1
        diffs_boolean_mask = np.ones(len(self.sequence) - 1, dtype=bool)
        diffs_boolean_mask[diffs_mask] = False
        
        bisequence = pd.Series(self.sequence[:-1]) + pd.Series(self.sequence[1:])
        self.BISEQUENCE_FREQS = bisequence.loc[diffs_boolean_mask].value_counts().reset_index()
        self.BISEQUENCE_FREQS.columns = ['biseq', 'freq']

    # @timeit
    def generate_one(self):
        mapper = {}
        for k, row in enumerate(self.ROWS_CONTENT):
            for i, s in enumerate(row):
                mapper[s] = (self.KEYBINDS[k][i][0]//10, self.KEYBINDS[k][i][1]//10)
        return mapper

    # @timeit
    def plot_keyboard(self, mapper):
        keyboard_img = Image.open(self.keyboard_img_path).convert('RGB')
        draw = ImageDraw.Draw(keyboard_img)
        font = ImageFont.truetype(self.font_path, 30)

        for s, v in mapper.items():
            display_char = 'Ent' if s == '\n' else ('__' if s == ' ' else s)
            x, y = v[0] * 10, v[1] * 10
            draw.text((x, y), display_char, font=font, fill=(255, 255, 255, 255))
        
        return keyboard_img.resize((500, 250))
    
    # @timeit
    # Метод для расчета функции приспособленности
    def get_scores_cpu(self, population):
        scores = []
        for mapper in population:
            cache = {}
            total_distance = 0

            # Кэширование расстояний
            for i in self.charmap:
                for j in self.charmap:
                    if (i + j) not in cache:
                        pos_i = np.array(mapper.get(i, [-100, -100]))
                        pos_j = np.array(mapper.get(j, [-100, -100]))
                        distance = np.linalg.norm(pos_i - pos_j)
                        cache[i + j] = distance
                    
            # Вычисление оценки на основе кэша
            weights = self.BISEQUENCE_FREQS.biseq.map(cache)
            total_distance += self.BISEQUENCE_FREQS['freq'].dot(weights)
            scores.append(total_distance)
            
        return scores
    
    # @timeit
    # Метод для расчета функции приспособленности
    def get_scores_gpu(self, population):
        scores = []
        
        # Мэппинг символов в индексы
        char_to_index = {char: idx for idx, char in enumerate(self.charmap)}
        bisequence_keys = self.BISEQUENCE_FREQS.biseq.to_numpy()
        bisequence_freqs = torch.tensor(self.BISEQUENCE_FREQS['freq'].to_numpy(), dtype=torch.float32, device=self.device)

        for mapper in population:
            # Преобразование позиции в единый numpy.ndarray
            pos_list = [mapper.get(c, [-100, -100]) for c in self.charmap]
            pos_array_np = np.array(pos_list, dtype=np.float32)
            pos_array = torch.tensor(pos_array_np, device=self.device)

            # Вычисление всех попарных расстояний
            pos_i = pos_array.unsqueeze(0)
            pos_j = pos_array.unsqueeze(1)
            distances = torch.norm(pos_i - pos_j, dim=2)

            # Кэширование
            cache_tensor = {}
            for idx_i, i in enumerate(self.charmap):
                for idx_j, j in enumerate(self.charmap):
                    key = i + j
                    cache_tensor[key] = distances[idx_i, idx_j]

            # Получение весов и вычисление итоговой дистанции
            weights = torch.tensor([cache_tensor.get(key, 0.0) for key in bisequence_keys], dtype=torch.float32, device=self.device)
            total_distance = torch.dot(bisequence_freqs, weights)

            scores.append(total_distance.item())

        return scores

    # @timeit
    # Фрагмент метода мутации
    def mutation(self, mapper, mutation_rate=0.05):
        # получение ключей и значений текущей раскладки
        keys = list(mapper.keys())
        values = list(mapper.values())
        
        # устанавливаем мутирующие индексы
        mutation_indices = (np.random.rand(len(values)) < mutation_rate).nonzero()[0]
        swap_indices = np.random.choice(len(values), size=len(mutation_indices), replace=False)
        
        # перемешиваем выбранные пары
        for i, j in zip(mutation_indices, swap_indices):
            # Swap positions
            values[i], values[j] = values[j], values[i]
        
        # возвращаем новую мутированную раскладку
        return {k: v for k, v in zip(keys, values)}
    
    # @timeit
    # Метод кроссовера
    def crossover(self, parent1, parent2):
        keysA = list(parent1.keys())
        valuesA = list(parent1.values())

        keysB = list(parent2.keys())
        valuesB = list(parent2.values())

        ranks = np.argsort(keysA)

        keysA = np.array(keysA)[ranks].copy()
        keysB = np.array(keysB)[ranks].copy()
        valuesA = np.array(valuesA)[ranks].copy()
        valuesB = np.array(valuesB)[ranks].copy()

        offset = np.random.randint(1, max(2, len(valuesA) - 1))
        offspring = {k: v for k, v in zip(keysA[:offset], valuesA[:offset])}

        keysO = list(offspring.keys())
        valuesO = list(offspring.values())

        keys_rest = list(filter(lambda k: k not in offspring, parent2.keys()))
        valuesRest = valuesA[offset:]
        values = valuesB[offset:]
        ranking = (
            values +
            (np.max(values) + 1) ** np.arange(len(values[0])).reshape(1, -1)
        ).sum(axis=1).argsort()

        for k, v in zip(keys_rest, valuesRest[ranking]):
            offspring[k] = v

        # Проверка уникальности и полноты
        assert set(offspring.keys()).symmetric_difference(parent1.keys()).__len__() == 0
        assert set(map(tuple, offspring.values())).symmetric_difference(set(map(tuple, parent1.values()))).__len__() == 0
        
        return offspring

    # @timeit
    # Метод для создания начальной популяции из случайных раскладок
    def generate_initial(self, population_size):
        return [self.generate_one() for _ in range(population_size)]
    
    # @timeit
    # Метод для генерации новой популяции на основе текущей
    def generate_new_population(self, population, scores, mutation_rate=0.1):
        new_population = []

        # Селекция: выбираем лучших родителей
        parents = sorted(zip(scores, population), key=lambda x: x[0])[:len(population)//2]
        parents = [p for _, p in parents]

        # Генерация потомков через кроссовер
        for _ in range(len(population) - len(parents)):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            child = self.crossover(parent1, parent2)
            new_population.append(child)
        
        # Мутация потомков
        mutated_offspring = [self.mutation(child, mutation_rate) for child in new_population]

        # Новая популяция
        return parents + mutated_offspring

    # @timeit
    def evolve(self, population_size, elitism_top_k, random_size, num_generations, num_restarts):
        best_score = np.inf
        best_image = None
        stats = []

        print('-' * 120)
        for restart in range(num_restarts):
            print(f'Рестарт: {restart + 1}')
            population = self.generate_initial(population_size)
            
            print('-' * 120)
            for generation in range(num_generations):
                # scores = self.get_scores_cpu(population)
                scores = self.get_scores_gpu(population)
                population = self.generate_new_population(population, scores)

                stats.append({
                    'restart': restart + 1,
                    'generation': generation,
                    'best_score': min(scores),
                    'worst_score': max(scores),
                    'mean_score': np.mean(scores)
                })

                if min(scores) < best_score:
                    best_score = min(scores)
                    best_image = self.plot_keyboard(population[np.argmin(scores)])
                    print(f'Поколение: {generation}\tЛучшее расстояние: {min(scores):.1f}\t'
                          f'Худшее расстояние: {max(scores):.1f}\t'
                          f'Среднее расстояние в популяции: {np.mean(scores):.1f}')
            print('-' * 120)
                
        return best_score, best_image, stats

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Keyboard Optimizer with specified parameters.')
    parser.add_argument('-p', '--population_size', type=int, default=200, help='Size of the population.')
    parser.add_argument('-e', '--elitism_top_k', type=int, default=10, help='Number of top individuals to retain for elitism.')
    parser.add_argument('-r', '--random_size', type=int, default=100, help='Number of random individuals to add.')
    parser.add_argument('-g', '--num_generations', type=int, default=100, help='Number of generations to process.')
    parser.add_argument('-n', '--num_restarts', type=int, default=10, help='Number of restarts in optimization.')
    parser.add_argument('-m', '--message_word', type=str, default='высокоинтеллектуальное', help='Word used in message generation.')
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to run the optimizer on.')
    
    return parser.parse_args()

# Использование класса
if __name__ == '__main__':
    # Парсинг аргументов командной строки
    args = parse_arguments()
        
    messages = [np.random.choice([args.message_word], size=1000)]   
    optimizer = KeyboardOptimizer(messages, device=args.device)

    # Извлечение параметров из аргументов
    POPULATION_SIZE = args.population_size
    ELITISM_TOPK = args.elitism_top_k
    RANDOM_SIZE = args.random_size
    NUM_GENERATIONS = args.num_generations
    NUM_RESTARTS = args.num_restarts
    
    best_score, best_image, _ = optimizer.evolve(
        population_size=POPULATION_SIZE,
        elitism_top_k=ELITISM_TOPK,
        random_size=RANDOM_SIZE,
        num_generations=NUM_GENERATIONS,
        num_restarts=NUM_RESTARTS
    )

    if best_image:
        output_path = "best_layout.png"
        best_image.save(output_path, "PNG")

        import os
        os.system(f"xdg-open {output_path}")