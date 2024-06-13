# Trading Proyects

---

### Team Members: Gustavo de Anda, Ignacio Maximiliano Jimenez Ramirez, Andres Martinez Covarrubias, and Sebastian Monsalve Gallegos
The objective of this project is to create an algorithm that optimizes buy and sell signals as effectively as possible for trading using different models and strategies. We tested the code we created with two different assets: AAPL stock and BTC-USD cryptocurrency.

---
## Proyect 2: Technical Analysis
The main objective of this project is to optimize an algorithm that helps us perform efficient trading using different technical indicators such as RSI, MACD, Bollinger Bands, and ATR. What we implemented in the algorithm is the optimization of buy and sell signals by taking the different parameters used by each technical indicator, seeking the best combination of these different technical indicators to achieve positive results. The results obtained can be found in the Jupyter notebook.

To get started with this repository, you will need to have Python installed on your machine. We recommend using Python 3.10 or higher.
1. Fork this repository
2. Clone the repo

```python
git clone <repo_url>
cd official_second_proyect
```
3. Create a virtual environment

```python
python -m venv venv
source venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```
4. Install the required dependencies

```python
pip install -r requirements.txt
```
## Repository structure

---

### data

---
In this folder, you will find everything related to the datasets we used throughout the project to achieve the results obtained in the Jupyter notebook. Specifically, it includes the CSV files for BTC and AAPL, in both train and test modalities, with 5-minute and 1-minute intervals. Additionally, the best combinations of all the datasets for BTC and AAPL in 1-minute and 5-minute intervals are saved in TXT format.
### technical_analysis

---


In this directory, you will find the three main functions. The first one is technical_analysis, whose main objective is to optimize all the technical indicators with their different data inputs and save the results in a DataFrame.

Secondly, we have the technical_indicators library, which simply serves to visually display the different technical indicators when running the training sessions.

Lastly, we have the profit_calculator function, which is responsible for loading the data from technical_analysis to develop the profit function, taking into account the optimized variables, and making both long and short trades.

Additionally, in the technical_analysis folder, we have four training files for BTC and AAPL, which can be run thanks to the optimal portfolio results saved in a separate TXT file. These were created to compare them with the benchmark. On the other hand, we have the test files, from which we extracted information from the training files in the data directory using the optimal indicators for the respective asset. 

A test file named trials8.py was created to maximize the use of processors and run the code faster.

## Proyect 3: Machine learning applied to trading