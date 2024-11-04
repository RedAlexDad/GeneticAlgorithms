import json
import random
import numpy as np

width = 750
height = 585
grid = 15
paddleHeight = grid*5
maxPaddleY = height - grid - paddleHeight
paddleSpeed = 6
ballSpeed = 5

leftPaddle = dict(x=grid*2,
                  y=height/2 - paddleHeight/2,
                  width=grid,
                  height=paddleHeight,
                  dy=0)
rightPaddle = dict(x=width-grid*3,
                   y = height / 2 - paddleHeight/2,
                   width=grid,
                   height=paddleHeight,
                   dy=0)
ball = dict(x=width/2,
            y=height/2,
            width=grid,
            height=grid,
            resetting=False,
            dx=ballSpeed,
            dy=-ballSpeed,
            score=0)

def collides(obj1, obj2):
    return (
        obj1['x'] < obj2['x'] + obj2['width'] and
        obj1['x'] + obj1['width'] > obj2['x'] and
        obj1['y'] < obj2['y'] + obj2['height'] and
        obj1['y'] + obj1['height'] > obj2['y']
    )


def restart(leftPaddle, rightPaddle, ball):
    ball['resetting'] = False
    ball['x'] = width / 2
    ball['y'] = height / 2
    ball['score'] = 0

    leftPaddle['x'] = grid*2
    leftPaddle['y'] = height/2 - paddleHeight/2

    rightPaddle['x'] = width - grid*3
    rightPaddle['y'] = height / 2 - paddleHeight/2


def loop(leftPaddle, rightPaddle, ball):
    leftPaddle['y'] += leftPaddle['dy']
    rightPaddle['y'] += rightPaddle['dy']

    if (leftPaddle['y'] < grid):
        leftPaddle['y'] = grid
    elif (leftPaddle['y'] > maxPaddleY):
        leftPaddle['y'] = maxPaddleY

    if (rightPaddle['y'] < grid):
        rightPaddle['y'] = grid
    elif (rightPaddle['y'] > maxPaddleY):
        rightPaddle['y'] = maxPaddleY

    ball['x'] += ball['dx']
    ball['y'] += ball['dy']

    if (ball['y'] < grid):
        ball['y'] = grid
        ball['dy'] *= -1
    elif (ball['y'] + grid > height - grid):
        ball['y'] = height - grid * 2
        ball['dy'] *= -1

    if ( (ball['x'] < 0 or ball['x'] > width) and not ball['resetting']):
        return -1

    if (collides(ball, leftPaddle)):
        ball['dx'] *= -1
        ball['x'] = leftPaddle['x'] + leftPaddle['width']
        ball['score'] += 1
    elif (collides(ball, rightPaddle)):
        ball['dx'] *= -1
        ball['x'] = rightPaddle['x'] - rightPaddle['width']
        ball['score'] += 1

    return 0

actionMap = {0: 38, # правый игрок вверх
             1: 40, # правый игрок вниз
             2: 87, # левый игрок вверх
             3: 83, # левый игрок вниз
             4: -1, # правый игрок ожидает
             5: -2} # вниз игрок ожидает

def apply_action(leftPaddle, rightPaddle, actionId):
    key = actionMap[actionId]

    if key == 38:
        rightPaddle['dy'] = -paddleSpeed
    elif key == 40:
        rightPaddle['dy'] = +paddleSpeed
    elif key == -1:
        rightPaddle['dy'] = 0
    elif key == 87:
        leftPaddle['dy'] = -paddleSpeed
    elif key == 83:
        leftPaddle['dy'] = +paddleSpeed
    elif key == -2:
        leftPaddle['dy'] = 0
        
def get_features(leftPaddle, rightPaddle, ball):
    sensors = [
        np.sign(leftPaddle['y'] - ball['y']),
        np.abs(leftPaddle['y'] - ball['y']) / height,
        np.abs(leftPaddle['x'] - ball['x']) / width,
        np.sign(rightPaddle['y'] - ball['y']),
        np.abs(rightPaddle['y'] - ball['y']) / height,
        np.abs(rightPaddle['x'] - ball['x']) / width,
        np.sign(leftPaddle['dy']),
        np.sign(leftPaddle['dy'])==0,
        np.sign(rightPaddle['dy']),
        np.sign(rightPaddle['dy'])==0,
        np.sign(ball['dx']),
        np.sign(ball['dy']),
        np.sign(ball['x'] - width//2),
        np.sign(ball['y'] - height//2),
        1 # что это?
    ]

    return sensors

get_features(leftPaddle, rightPaddle, ball)

def get_one():
    W = np.random.normal(size=(15, 6))
    return W

def getAction(leftPaddle, rightPaddle, ball, W):
    return (W.T.dot(get_features(leftPaddle, rightPaddle, ball))).argmax()

getAction(leftPaddle, rightPaddle, ball, get_one())

def get_score(W, patience=100):
    restart(leftPaddle, rightPaddle, ball)
    maxScore_patience = patience
    maxScore_prev = ball['score']
    action = getAction(leftPaddle, rightPaddle, ball, W)
    for _ in range(int(2e4)):
        if loop(leftPaddle, rightPaddle, ball) == -1:
            break
        # симуляция запоздалой реакции агента
        if np.random.random() < 0.5:
            action = getAction(leftPaddle, rightPaddle, ball, W)
        apply_action(leftPaddle, rightPaddle, action)
        if  ball['score'] > maxScore_prev:
            maxScore_prev = ball['score']
            maxScore_patience = patience
        maxScore_patience -= 1
        if maxScore_patience < 0:
            break
    return ball['score']

def mutate(W, mutation_rate=0.02):
    dW = get_one()
    dM = get_one() > 0
    return W + dW * dM * mutation_rate


def crossover(W1, W2):
    maskW = np.random.random(W1.shape) < 0.5
    return W1 * maskW + W2 * (~maskW)

def generate_random(population, size):
    new_population = []
    for _ in range(size):
        if np.random.random() < 0.5:
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

    new_population = [population[i].copy() for i in elitismTopK] + \
                     [population[i].copy() for i in roulleteTopK]

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


def get_new_population(population, scores, topK=4, randomNum=10):
    return (
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

POPULATION_SIZE = 128
RANDOM_SIZE = 20
ELITE_SIZE = 5
NUM_GENERATIONS = 100
NUM_REPEATS = 3
NUM_RESTARTS = 5
PATIENCE = lambda x: 1000*((x+2)//2)

best_thingey = None
best_score = 0

for n_restart in range(NUM_RESTARTS):
    print('='*50)
    print('Cтарт перезапуска №%d'%(n_restart+1))
    print('Лучшая пока что: %.1f'%best_score)
    print('='*50)
    population = [get_one() for _ in range(POPULATION_SIZE)]

    for generation in range(NUM_GENERATIONS):
        scores = 1e-10
        for _ in range(NUM_REPEATS):
            scores += np.array(get_scores(population, PATIENCE(generation)))
        scores /= NUM_REPEATS
        bscore = max(scores)

        scores **= 4 # зачем?
        population = get_new_population(population, scores,
                                        topK=ELITE_SIZE,
                                        randomNum=RANDOM_SIZE)
        if bscore > best_score:
            best_score = bscore
            best_thingey = np.array(population[0])
            print('Рестарт: %d\tПоколение: %d\tЗначение: %.1f'%(n_restart+1,
                                                                generation,
                                                                bscore))