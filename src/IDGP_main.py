# python packages
import random
import time
import operator
import evalGP_main as evalGP
# only for strongly typed GP
import gp_restrict
import pandas as pd
import numpy as np
# deap package
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Img, Region, Vector
import feature_function as fe_fs
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
#small helper function
from dataset_reader_example_code import load_dataset  


'FLGP'

dataSetName='f1'
randomSeeds=2

x_train = np.load(dataSetName + '_train_data.npy')/ 255.0
y_train = np.load(dataSetName + '_train_label.npy')
x_test = np.load(dataSetName + '_test_data.npy')/ 255.0
y_test = np.load(dataSetName + '_test_label.npy')

print(x_train.shape)

# parameters:
population = 40
generation = 15
cxProb = 0.8
mutProb = 0.2
elitismProb = 0.05
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 4
maxDepth = 6

bound1, bound2 = x_train[1, :, :].shape
##GP

pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
#Feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector, name='FeaCon2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector, name='FeaCon3')
# Global feature extraction
pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
pset.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')
# Local feature extraction
pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
pset.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')
# Region detection operators
pset.addPrimitive(fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
pset.addPrimitive(fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
# Terminals
pset.renameArguments(ARG0='Grey')
pset.addEphemeralConstant('X', lambda: random.randint(0, bound1 - 20), Int1)
pset.addEphemeralConstant('Y', lambda: random.randint(0, bound2 - 20), Int2)
pset.addEphemeralConstant('Size', lambda: random.randint(20, 51), Int3)

#fitnesse evaluaiton
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)

def evalTrain(individual):
    # print(individual)
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    # print(train_norm.shape)
    lsvm = LinearSVC(max_iter=100)
    accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=3).mean(), 2)
    return accuracy,

def evalTrainb(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y_train)):
            train_tf.append(np.asarray(func(x_train[i, :, :])))
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
        lsvm = LinearSVC(max_iter=100)
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=3).mean(), 2)
    except:
        accuracy = 0
    return accuracy,

# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof

'''
def evalTest(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    for j in range(0, len(y_test)):
        test_tf.append(np.asarray(func(x_test[j, :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm= LinearSVC()
    lsvm.fit(train_norm, y_train)
    accuracy = round(100*lsvm.score(test_norm, y_test),2)
    return train_tf.shape[1], accuracy
'''

def evalTest(individual):
    func = toolbox.compile(expr=individual)

    train_tf = []
    test_tf = []
    for i in range(len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    for j in range(len(y_test)):
        test_tf.append(np.asarray(func(x_test[j, :, :])))

    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)

    # Normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)

    # Train SVM
    lsvm = LinearSVC(max_iter=1000)
    lsvm.fit(train_norm, y_train)
    accuracy = round(100 * lsvm.score(test_norm, y_test), 2)

    # Create DataFrames
    n_features = train_norm.shape[1]
    train_with_labels = np.column_stack((train_norm, y_train))
    test_with_labels = np.column_stack((test_norm, y_test))

    train_columns = [f"feature_{i}" for i in range(n_features)] + ["label"]
    test_columns = [f"feature_{i}" for i in range(n_features)] + ["label"]

    train_df = pd.DataFrame(train_with_labels, columns=train_columns)
    test_df = pd.DataFrame(test_with_labels, columns=test_columns)

    # Save pattern files as CSVs (features + label)
    train_df.to_csv(f"{dataSetName}_train_patterns.csv", index=False)
    test_df.to_csv(f"{dataSetName}_test_patterns.csv", index=False)

    #  Return results (same as before)
    return train_tf.shape[1], accuracy, train_df, test_df



if __name__ == "__main__":
    datasets = ["f1", "f2"]

    for dataSetName in datasets:
        print(f"\n=== Processing dataset: {dataSetName} ===")

        # You must load dataset before running GPMain
        x_train, y_train, x_test, y_test = load_dataset(dataSetName)

        beginTime = time.process_time()
        pop, log, hof = GPMain(randomSeeds)
        endTime = time.process_time()
        trainTime = endTime - beginTime

        # Evaluate best individual from Hall of Fame
        num_features, testResults, train_df, test_df = evalTest(hof[0])

        # Save pattern files
        train_df.to_csv(f"{dataSetName}_train_patterns.csv", index=False)
        test_df.to_csv(f"{dataSetName}_test_patterns.csv", index=False)

        endTime1 = time.process_time()
        testTime = endTime1 - endTime

        print('Best individual ', hof[0])
        print('Test results  ', testResults)
        print('Train time  ', trainTime)
        print('Test time  ', testTime)
        print(f"Pattern files saved for {dataSetName}")

        # Save best tree
        with open(f"{dataSetName}_best_tree.txt", "w") as f:
            f.write(str(hof[0]))

    print('End of Problem 4.1')



#pseudocode
'''
Input: Training images (X_train), Training labels (y_train)
Output: Best GP-based feature extractor (GP_best), Extracted features

1. Initialize population P of GP trees with random combinations of feature functions
   (e.g., Global_HOG, Local_SIFT, Region_R, etc.)

2. For each generation (g = 1 to MaxGenerations):
       For each individual tree T in P:
           a. Compile T into a callable function f(image)
           b. Extract features F_train = [f(x) for x in X_train]
           c. Normalize F_train using MinMaxScaler
           d. Evaluate fitness = mean cross-validation accuracy of LinearSVM(F_train, y_train)

       e. Select individuals for reproduction (Tournament selection)
       f. Apply crossover and mutation to create offspring
       g. Apply elitism to keep best individuals
       h. Update population P

3. Select best individual GP_best from Hall of Fame (highest accuracy)

4. Evaluate GP_best on test set:
       a. Extract F_test = [GP_best(x) for x in X_test]
       b. Normalize and evaluate LinearSVM accuracy

5. Save extracted features and GP_best tree
'''
