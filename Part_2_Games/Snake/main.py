
import json
import random
import numpy as np

def generate_apple(apple):
    apple['x'] = random.randint(0, 25) * grid
    apple['y'] = random.randint(0, 25) * grid


def loop(snake, apple):
    snake['x'] += snake['dx']
    snake['y'] += snake['dy']

    if snake['x'] < 0:
        snake['x'] = width - grid
    elif snake['x'] >= width:
        snake['x'] = 0

    if snake['y'] < 0:
        snake['y'] = height - grid
    elif snake['y'] >= height:
        snake['y'] = 0

    snake['cells'] = [(snake['x'], snake['y'])] + snake['cells']

    if len(snake['cells']) > snake['maxCells']:
        snake['cells'].pop()

    for index, cell in enumerate(snake['cells']):
        if cell[0] == apple['x'] and cell[1] == apple['y']:
            snake['maxCells'] += 1
            generate_apple(apple)

        for i in range(index + 1, len(snake['cells'])):
            # snake occupies same space as a body part. reset game
            if (cell[0] == snake['cells'][i][0] and
                cell[1] == snake['cells'][i][1]):
                return -1 #restart(snake, apple)

    return 0

def restart():
    snake = dict(x=160, y=160, dx=grid, dy=0, cells=[], maxCells=4)
    apple = dict(x=320, y=320)
    return snake, apple

def apply_action(snake, actionId):
    actionMap = {0: 37, 1: 38, 2: 39, 3: 40}
    key = actionMap[actionId]

    if (key == 37 and snake['dx'] == 0):
        snake['dx'] = -grid
        snake['dy'] = 0
    elif (key == 38 and snake['dy'] == 0):
        snake['dx'] = 0
        snake['dy'] = -grid
    elif (key == 39 and snake['dx'] == 0):
        snake['dx'] = grid
        snake['dy'] = 0
    elif (key == 40 and snake['dy'] == 0):
        snake['dx'] = 0
        snake['dy'] = grid

def get_features(snake, apple):
    sensors = [
        np.sign(snake['dx']),
        np.sign(snake['dy']),
        (snake['x'] - snake['cells'][-1][0])/width if len(snake['cells']) else 0,
        (snake['y'] - snake['cells'][-1][1])/height if len(snake['cells']) else 0,
        snake['x'] == apple['x'],
        snake['y'] == apple['y'],
        (snake['x'] - apple['x'])/width>0,
        (snake['x'] - apple['x'])/width<0,
        (snake['y'] - apple['y'])/height>0,
        (snake['y'] - apple['y'])/height<0,
        any([(snake['x'] == cell[0] and snake['dy'] == 0) for cell in snake['cells'][1:]]),
        any([(snake['y'] == cell[1] and snake['dx'] == 0) for cell in snake['cells'][1:]]),
        any([(snake['x'] == cell[0] and snake['dy'] > 0) for cell in snake['cells'][1:]]),
        any([(snake['y'] == cell[1] and snake['dx'] > 0) for cell in snake['cells'][1:]]),
        any([(snake['x'] == cell[0] and snake['dy'] < 0) for cell in snake['cells'][1:]]),
        any([(snake['y'] == cell[1] and snake['dx'] < 0) for cell in snake['cells'][1:]]),
    ]
    return sensors

def get_one():
    W = np.random.normal(size=(16, 4))
    b = np.random.normal(size=(4,))
    return W, b


def getAction(snake, apple, W, b):
    return (W.T.dot(get_features(snake, apple)) + b).argmax()

def get_score(W, b, patience=100):
    snake, apple = restart()
    maxCells_patience = patience
    maxCells_prev = snake['maxCells']
    while loop(snake, apple) != -1:
        apply_action(snake, getAction(snake, apple, W, b))
        if snake['maxCells'] > maxCells_prev:
            maxCells_prev = snake['maxCells']
            maxCells_patience = patience
        maxCells_patience -= 1
        if maxCells_patience < 0:
            snake['maxCells'] = snake['maxCells']/2
            break
    return snake['maxCells']

def mutate(W, b, mutation_rate=0.02):
    dW, db = get_one()
    dWM, dbM = get_one()
    return (W + dW * (dWM > 0) * mutation_rate,
            b + db * (dbM > 0) * mutation_rate)


def crossover(W1, b1, W2, b2):
    maskW = np.random.random(W1.shape) < 0.5
    maskb = np.random.random(b1.shape) < 0.5
    return W1 * maskW + W2 * (~maskW), b1 * maskb + b2 * (~maskb)

def generate_random(population, size):
    new_population = []
    for _ in range(size):
        if np.random.random()<0.5:
            new_population.append(get_one())
        else:
            new_population.append(mutate(*population[0]))
    return new_population


def selection(population, scores, topK=2):
    scores = np.array(scores)*1.
    scores /= scores.sum()
    elitismTopK = np.argsort(scores)[::-1][:topK//2]
    roulleteTopK = np.random.choice(len(scores),
                                    p=scores,
                                    size=topK//2)

    new_population = [tuple(map(lambda x: x.copy(), population[i])) for i in elitismTopK]+\
                     [tuple(map(lambda x: x.copy(), population[i])) for i in roulleteTopK]

    return new_population


def breed(population, scores, nChilds=10):
    scores = np.array(scores)*1.
    scores /= scores.sum()
    parents = np.random.choice(len(scores),
                               p=scores,
                               size=(nChilds, 2))

    new_population = []
    for parentA, parentB in parents:
        new_population.append(mutate(*crossover(*population[parentA], *population[parentB])))

    return new_population


def get_new_population(population, scores, topK=4, randomNum=10):
    return (
        selection(population, scores, topK) + \
        breed(population, scores, nChilds=max(0, len(population) - randomNum - topK)) + \
        generate_random(population, randomNum)
    )

def get_scores(population, patience=100):
    scores = []
    for W, b in population:
        scores.append(get_score(W, b, patience))
    return scores

if __name__ in '__main__':
    width = 400
    height = 400
    grid = 16
    count = 0
    snake = dict(x=160, y=160, dx=grid, dy=0, cells=[], maxCells=4)
    apple = dict(x=320, y=320)
    
    get_features(snake, apple)
        
    getAction(snake, apple, *get_one())

    POPULATION_SIZE = 64
    NUM_GENERATIONS = 10
    NUM_REPEATS = 3 # зачем?
    NUM_RESTARTS = 5
    PATIENCE = lambda x: 100*((x+5)//5)

    best_thingey = None
    best_score = 0

    for n_restart in range(NUM_RESTARTS):
        print('='*50)
        print('Cтарт перезапуска №%d'%(n_restart+1))
        print('Лучшая пока что: %.1f'%best_score)
        print('='*50)
        population = [get_one() for _ in range(POPULATION_SIZE)]

        for generation in range(NUM_GENERATIONS):
            scores = 0
            for _ in range(NUM_REPEATS):
                scores += np.array(get_scores(population, PATIENCE(generation)))
            scores /= NUM_REPEATS
            bscore = max(scores)

            scores **= 4 # зачем?
            population = get_new_population(population, scores, topK=5, randomNum=20)
            if bscore > best_score:
                best_score = bscore
                best_thingey = np.concatenate([population[0][0],     # W
                                            [population[0][1]]])  # b
                print('Рестарт: %d\tПоколение: %d\tЗначение: %.1f'%(n_restart+1,
                                                                    generation,
                                                                    bscore))

    with open('snake_weights.js', 'w') as f:
        f.write('var W = %s;\n'%(json.dumps([[int(1e3*w)/1e3 for w in W] for W in best_thingey])))
        

