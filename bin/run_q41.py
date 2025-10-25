import time
from IDGP_main import GPMain, evalTest
from dataset_reader_example_code import load_dataset  # the helper you created

# Set your datasets and random seed
datasets = ["f1", "f2"]
randomSeeds = 2

for dataSetName in datasets:
    print(f"\n=== Processing dataset: {dataSetName} ===")
    
    # Load dataset (train/test split)
    x_train, y_train, x_test, y_test = load_dataset(dataSetName)

    # Start training GP
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    # Evaluate best individual
    num_features, testResults, train_df, test_df = evalTest(hof[0])

    # Save pattern files
    train_df.to_csv(f"{dataSetName}_train_patterns.csv", index=False)
    test_df.to_csv(f"{dataSetName}_test_patterns.csv", index=False)

    endTime1 = time.process_time()
    testTime = endTime1 - endTime

    # Print summary
    print('Best individual:', hof[0])
    print('Test results:', testResults)
    print('Train time:', trainTime)
    print('Test time:', testTime)
    print(f"Pattern files saved for {dataSetName}")

    # Save best tree
    with open(f"{dataSetName}_best_tree.txt", "w") as f:
        f.write(str(hof[0]))

print("End of Problem 4.1")
