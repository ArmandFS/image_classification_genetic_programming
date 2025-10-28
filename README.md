# Genetic Programming for Image Classification

A university project (AIML426) implementing **Genetic Programming (GP)** for automated feature extraction in computer vision and image classification tasks. This project demonstrates how evolutionary computation can be used to discover optimal feature extraction strategies for image classification problems.

## 🎯 Project Overview

This project tackles the challenge of **automated feature extraction** for image classification using Genetic Programming. Instead of manually designing feature extractors, the system evolves optimal feature extraction functions through evolutionary computation, which are then used to train machine learning classifiers.

### Key Features

- **Automated Feature Discovery**: Uses GP to evolve feature extraction functions
- **Strongly-Typed GP**: Implements type-safe genetic programming with custom data types
- **Multiple Feature Extractors**: Integrates HOG, LBP, SIFT, and custom feature extractors
- **End-to-End Pipeline**: Complete workflow from feature extraction to classification
- **Two-Dataset Support**: Works with multiple image classification datasets

## 🏗️ Project Architecture

```
Q4_GP_ImageCls/
├── src/                          # Core source code
│   ├── IDGP_main.py             # Main GP evolution engine
│   ├── evalGP_main.py           # GP evaluation functions
│   ├── feature_function.py      # GP primitive functions
│   ├── feature_extractors.py    # Feature extraction implementations
│   ├── sift_features.py         # SIFT feature extraction
│   ├── strongGPDataType.py      # Type system for strongly-typed GP
│   ├── gp_restrict.py           # GP constraints and restrictions
│   └── dataset_reader_example_code.py  # Dataset loading utilities
├── bin/                         # Executable scripts
│   ├── run_q41.py              # Feature extraction pipeline
│   └── run_q42.py              # Classification pipeline
├── data/                        # Dataset and results
│   ├── f1_*, f2_*              # Dataset files (train/test data/labels)
│   └── *_patterns.csv          # Generated feature patterns
└── pattern_files/              # Compressed pattern data
```

## 🧬 Genetic Programming Implementation

### Strongly-Typed GP System

The project implements a **strongly-typed genetic programming** approach with custom data types:

- **`Img`**: Represents input images
- **`Region`**: Image regions for local feature extraction
- **`Vector`**: Feature vectors (output of feature extractors)
- **`Int1`, `Int2`, `Int3`**: Integer parameters for feature extractors

### Primitive Functions

The GP system uses a rich set of primitive functions for feature extraction:

- **Feature Concatenation**: `FeaCon2`, `FeaCon3` for combining feature vectors
- **HOG Features**: Histogram of Oriented Gradients extraction
- **LBP Features**: Local Binary Pattern extraction
- **SIFT Features**: Scale-Invariant Feature Transform
- **Custom Extractors**: Domain-specific feature extraction functions

### GP Parameters

```python
population = 40          # Population size
generation = 15          # Number of generations
cxProb = 0.8            # Crossover probability
mutProb = 0.2           # Mutation probability
elitismProb = 0.05      # Elitism rate
maxDepth = 6            # Maximum tree depth
```

## 🚀 Quick Start

### Prerequisites

Install the required Python packages:

```bash
pip install numpy pandas matplotlib pillow scikit-learn scipy deap scikit-image
```

### Running the Complete Pipeline

#### Step 1: Feature Extraction (Problem 4.1)
```bash
python bin/run_q41.py
```

This script:
- Loads training and test datasets (f1, f2)
- Runs GP evolution to find optimal feature extractors
- Generates feature patterns for both datasets
- Saves the best evolved trees and pattern files

#### Step 2: Image Classification (Problem 4.2)
```bash
python bin/run_q42.py
```

This script:
- Loads the generated feature patterns
- Trains Linear SVM classifiers
- Evaluates performance on both training and test sets
- Reports classification accuracy

## 📊 Results and Output

### Generated Files

After running the pipeline, you'll find:

- **`*_train_patterns.csv`**: Training feature patterns
- **`*_test_patterns.csv`**: Test feature patterns  
- **`*_best_tree.txt`**: Best evolved GP trees
- **`*_properties.txt`**: GP run properties and statistics

### Performance Metrics

The system reports:
- **Training Accuracy**: Performance on training data
- **Test Accuracy**: Performance on unseen test data
- **Evolution Time**: Time taken for GP evolution
- **Feature Count**: Number of features extracted

## 🔬 Technical Details

### Feature Extraction Pipeline

1. **Image Preprocessing**: Images are normalized (divided by 255.0)
2. **GP Evolution**: Population of feature extractors evolves over generations
3. **Feature Generation**: Best individuals extract features from all images
4. **Pattern Storage**: Features are saved as CSV files for classification

### Classification Pipeline

1. **Pattern Loading**: Feature patterns are loaded from CSV files
2. **SVM Training**: Linear SVM is trained on extracted features
3. **Performance Evaluation**: Accuracy is measured on train/test sets
4. **Results Reporting**: Classification metrics are displayed

### GP Evolution Process

1. **Initialization**: Random population of feature extraction trees
2. **Evaluation**: Each individual extracts features and is evaluated
3. **Selection**: Best individuals are selected for reproduction
4. **Crossover**: Selected individuals produce offspring
5. **Mutation**: Random modifications to maintain diversity
6. **Replacement**: New generation replaces the old one
7. **Termination**: Process repeats until maximum generations

## 🎓 Academic Context

This project was developed for **AIML426 - Evolutionary Computation** and demonstrates:

- **Evolutionary Algorithm Design**: Implementation of genetic programming
- **Computer Vision Applications**: Feature extraction for image classification
- **Machine Learning Integration**: Combining GP with traditional ML classifiers
- **Type-Safe Programming**: Strongly-typed GP implementation

## 📁 Dataset Information

The project works with two datasets (f1, f2) containing:
- **Training Images**: Used for GP evolution and classifier training
- **Test Images**: Used for final performance evaluation
- **Labels**: Ground truth classifications for supervised learning

## 🛠️ Development and Extension

### Adding New Feature Extractors

To add new feature extraction functions:

1. Implement the function in `src/feature_extractors.py`
2. Add it to the primitive set in `src/IDGP_main.py`
3. Define appropriate input/output types

### Modifying GP Parameters

Adjust evolution parameters in `src/IDGP_main.py`:
- Population size and generations
- Crossover and mutation rates
- Tree depth constraints
- Selection strategies

### Adding New Datasets

To work with new datasets:

1. Prepare data in the required format
2. Update `src/dataset_reader_example_code.py`
3. Modify dataset loading in the run scripts

## 📚 Dependencies

- **DEAP**: Genetic programming framework
- **scikit-learn**: Machine learning algorithms
- **scikit-image**: Image processing and feature extraction
- **NumPy/Pandas**: Data manipulation
- **Matplotlib**: Visualization (if needed)

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is part of academic coursework. Please respect academic integrity guidelines when using this code.

---

**Note**: This project demonstrates the power of evolutionary computation in automated feature engineering for computer vision tasks. The GP system can discover novel and effective feature extraction strategies that might not be obvious to human designers.
