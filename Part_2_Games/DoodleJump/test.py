import json
import random
import numpy as np

def init_states():
    settings = dict(
        width = 375,
        height = 667,
        platformWidth = 65,
        platformHeight = 20,

        gravity = 0.33,
        drag = 0.3,
        bounceVelocity = -12.5,

        minPlatformSpace = 15,
        maxPlatformSpace = 20,
        keydown = False,
        score = 0,
    )
    settings['platformStart'] = settings['height'] - 50

    platforms = [dict(x=settings['width'] / 2 - settings['platformWidth'] / 2,
                      y=settings['platformStart'])]
    y = settings['platformStart']
    while (y > 0):
        y -= settings['platformHeight'] + \
             np.random.randint(settings['minPlatformSpace'],
                               settings['maxPlatformSpace'])

        while True:
            x = np.random.uniform(25, settings['width'] \
                                  - 25 - settings['platformWidth'])
            if not ((y > settings['height'] / 2) and
                    (x > settings['width'] / 2 
                     - settings['platformWidth'] * 1.5) and
                    (x < settings['width'] / 2 
                     + settings['platformWidth'] / 2)):
                break
        platforms.append(dict(x=x, y=y))

    doodle = dict(
      width=40,
      height=60,
      x=settings['width'] / 2 - 20,
      y=settings['platformStart'] - 60,
      dx=0,
      dy=0,
      playerDir=0,
      prevDoodleY=settings['platformStart'] - 60,
    )

    return doodle, platforms, settings

doodle, platforms, settings = init_states()

def restart():
    doodle, platforms, settings = init_states()
    return doodle, platforms, settings


def loop(doodle, platforms, settings):
    doodle['dy'] += settings['gravity']

    if (doodle['y'] < settings['height'] / 2 and doodle['dy'] < 0):
        for i, _ in enumerate(platforms):
            platforms[i]['y'] -= doodle['dy']

        while (platforms[-1]['y'] > 0):
            platforms.append(dict(
                x=np.random.uniform(25,
                                    settings['width'] - 25
                                    - settings['platformWidth']),
                y=np.random.uniform(platforms[-1]['y'] -
                                    (settings['platformHeight'] +
                                     np.random.uniform(
                                         settings['minPlatformSpace'],
                                         settings['maxPlatformSpace']))
                                    )
                                )
                            )

            settings['minPlatformSpace'] = min(settings['minPlatformSpace'] 
                                               + 0.5,
                                               settings['height'] / 2 - 0.5)
            settings['maxPlatformSpace'] = min(settings['maxPlatformSpace'] 
                                               + 0.5,
                                               settings['height'] / 2)
    else:
        doodle['y'] += doodle['dy']

    if not settings['keydown']:
        if (doodle['playerDir'] < 0):
            doodle['dx'] += settings['drag'];
            if (doodle['dx'] > 0):
                doodle['dx'] = 0
                doodle['playerDir'] = 0
        elif (doodle['playerDir'] > 0):
            doodle['dx'] -= settings['drag']

            if (doodle['dx'] < 0):
                doodle['dx'] = 0
                doodle['playerDir'] = 0

    doodle['x'] += doodle['dx']

    if (doodle['x'] + doodle['width'] < 0):
        doodle['x'] = settings['width']
    elif (doodle['x'] > settings['width']):
        doodle['x'] = -doodle['width']

    for platform in platforms:
        if (
          (doodle['dy'] > 0) and
          (doodle['prevDoodleY'] + doodle['height'] <= platform['y']) and
          (doodle['x'] < platform['x'] + settings['platformWidth']) and
          (doodle['x'] + doodle['width'] > platform['x']) and
          (doodle['y'] < platform['y'] + settings['platformHeight']) and
          (doodle['y'] + doodle['height'] > platform['y'])
        ):
            doodle['y'] = platform['y'] - doodle['height']
            doodle['dy'] = settings['bounceVelocity']

    doodle['prevDoodleY'] = doodle['y']
    platforms_cleared = len(platforms)
    platforms = list(filter(lambda platform: platform['y'] < settings['height'],
                            platforms))
    platforms_cleared -= len(platforms)
    settings['score'] += platforms_cleared

    if doodle['y'] > settings['height'] + doodle['height']:
        return -1
    return 0

actionMap = {0: 37, # движение влево
             1: 39, # движение вправо
             2: -1} # ожидание

def apply_action(doodle, platforms, settings, actionId):
    key = actionMap[actionId]

    if key == 37:
        settings['keydown'] = True
        settings['playerDir'] = -1
        doodle['dx'] = -3
    elif key == 39:
        settings['keydown'] = True
        settings['playerDir'] = 1
        doodle['dx'] = 3
    else:
        settings['keydown'] = False
        
# агент видел значения в этих пикселях
sensor_web = np.meshgrid(np.arange(-settings['width']*2//3,
                                   +settings['width']*2//3, 50),
                         np.arange(-settings['height']*2//3,
                                   +settings['height']*2//3, 75))
sensor_web = np.concatenate([sensor_web[0].flatten()[:, None],
                             sensor_web[1].flatten()[:, None]], axis=1)

def get_features(doodle, platforms, settings):
    points = np.array([(p['x'], p['y']) for p in platforms])

    sensor_x = (sensor_web[:, 0]*1 + doodle['x']) % settings['width']
    sensor_y = np.clip((sensor_web[:, 1]*1 + doodle['y']),
                       1, settings['height']-1)

    xx = sensor_x.reshape(-1, 1) - points[:, 0]
    yy = sensor_y.reshape(-1, 1) - points[:, 1]
    cond1 = (xx - settings['platformWidth']) < 0
    cond2 = (xx) > 0
    cond3 = (yy - settings['platformHeight']) < 0
    cond4 = (yy) > 0

    sensors = ((cond1*cond2*cond3*cond4).any(axis=1))*1.
    return np.concatenate([sensors, [doodle['dx'],
                                     doodle['dy'],
                                     1]])

get_features(doodle, platforms, settings)

# Многослойный персептрон из ЛР3 передаёт привет:)
def get_one(h1=5, n_classes=3):
    W = np.random.normal(size=(sensor_web.shape[0]+3, h1))
    W2 = np.random.normal(size=(h1, n_classes))
    return W, W2

def softmax(x):
    xe = np.exp(x-x.max())
    return xe/xe.sum()

def getAction(doodle, platforms, settings, weights):
    W, W2 = weights
    logits = np.maximum(W.T.dot(get_features(doodle, platforms, settings)),
                        0).dot(W2)
    # действия выбираются не детерминированно, а вероятностно
    return np.random.choice(np.arange(logits.size), p=softmax(logits))

getAction(doodle, platforms, settings, get_one())

def get_score(W, patience=100, return_actions=False):
    doodle, platforms, settings = restart()
    maxScore_patience = patience
    maxScore_prev = settings['minPlatformSpace']
    actions = []
    xcoords = []
    action = getAction(doodle, platforms, settings, W)
    for _ in range(int(5e4)):
        if loop(doodle, platforms, settings) == -1:
            break
        # симуляция запоздалой реакции агента
        if np.random.random() < 0.25:
            action = getAction(doodle, platforms, settings, W)
        actions.append(action)
        xcoords.append(doodle['x'])
        apply_action(doodle, platforms, settings, action)
        if  settings['minPlatformSpace'] > maxScore_prev:
            maxScore_prev = settings['minPlatformSpace']
            maxScore_patience = patience
        maxScore_patience -= 1
        if maxScore_patience < 0:
            break
    if return_actions:
        return actions, xcoords, settings['minPlatformSpace']
    return settings['minPlatformSpace']

def mutate(weights, mutation_rate=0.01):
    W, W2 = weights
    dW, dW2 = get_one()
    dM, dM2 = get_one()
    return W + dW*(dM>0)*mutation_rate, W2 + dW2*(dM2>0)*mutation_rate


def crossover(W1, W2):
    result = []
    for w1, w2 in zip(W1, W2):
        maskW = np.random.random(w1.shape)<0.5
        result.append(w1*maskW+w2*(~maskW))
    return result

def generate_random(population, size):
    new_population = []
    for _ in range(size):
        if np.random.random()<0.5:
            new_population.append(get_one())
        else:
            new_population.append(mutate(population[0]))
    return new_population


def selection(population, scores, topK=2):
    scores = np.array(scores)*1.
    scores /= scores.sum()
    elitismTopK = np.argsort(scores)[::-1][:topK//2]
    roulleteTopK = np.random.choice(len(scores),
                                    p=scores,
                                    size=topK//2)

    new_population = [tuple(map(lambda x: x.copy(), population[i]))
                      for i in elitismTopK]+\
                     [tuple(map(lambda x: x.copy(), population[i]))
                      for i in roulleteTopK]

    return new_population


def breed(population, scores, nChilds=10):
    scores = np.array(scores)*1.
    scores /= scores.sum()
    parents = np.random.choice(len(scores),
                               p=scores,
                               size=(nChilds, 2))

    new_population = []
    for parentA, parentB in parents:
        new_population.append(mutate(crossover(population[parentA],
                                               population[parentB])))

    return new_population

# зачем?
def factorize(population, factor=3):
    for i, p in enumerate(population):
        population[i] = tuple([np.array([[int(10**factor*w)/10**factor
                                          for w in W]
                                         for W in pp])
                               for pp in p])
    return population


def get_new_population(population, scores, topK=4, randomNum=10):
    return factorize(
    selection(population, scores, topK) + \
    breed(population, scores,
          nChilds=max(0, len(population) - randomNum - topK)) + \
    generate_random(population, randomNum)
    )
    
def get_scores(population, patience=100):
    scores = []
    for W in population:
        scores.append(get_score(W, patience))
    return scores

# сохранение чекпоинта "мозгов" интеллектуального агента
def save_thingey(best_thingey, score):
    with open('doodlejump_weights_%.1f.js'%score, 'w') as f:
        f.write('var sensorWeb = %s;\n\nvar W = %s;\n\nvar W2 = %s;\n'%
                (json.dumps([[int(w) for w in W] for W in sensor_web]),
                 json.dumps([[int(1e2*w)/1e2 for w in W]
                             for W in best_thingey[0]]),
                 json.dumps([[int(1e2*w)/1e2 for w in W]
                             for W in best_thingey[1]])))
        
POPULATION_SIZE = 64
RANDOM_SIZE = 20
ELITE_SIZE = 4
NUM_GENERATIONS = 100
NUM_REPEATS = 3 # зачем?
NUM_RESTARTS = 5
PATIENCE = lambda x: 100*((x+2)//2)
population = [get_one() for _ in range(POPULATION_SIZE)]

best_thingey = None
best_score = 0

for n_restart in range(NUM_RESTARTS):
    print('='*50)
    print('Cтарт перезапуска №%d'%(n_restart+1))
    print('Лучшая пока что: %.1f'%best_score)
    print('='*50)
    population = [get_one() for _ in range(POPULATION_SIZE)]
    for generation in range(NUM_GENERATIONS):
        scores = 0.
        for _ in range(NUM_REPEATS):
            scores += np.array(get_scores(population, PATIENCE(generation)))**4
        scores /= NUM_REPEATS

        population = get_new_population(population, scores,
                                        topK=ELITE_SIZE,
                                        randomNum=RANDOM_SIZE)
        bscore = max(scores)**0.25
        if bscore > best_score:
            best_score = bscore
            best_thingey = [x.copy() for x in population[0]]
            print('Рестарт: %d\tПоколение: %d\tЗначение: %.1f'%(n_restart+1,
                                                                generation,
                                                                bscore))
            if bscore > 100:
                save_thingey(best_thingey, best_score)
                
save_thingey(best_thingey, best_score)

