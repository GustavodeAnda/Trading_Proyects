import pandas as pd
import technical_analysis
import profit_calculator
import technical_indicators
from itertools import combinations
import optuna
from concurrent.futures import ProcessPoolExecutor
import json
import logging

# Configurar el registro
logging.basicConfig(filename='optimization_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def optimize_combination(args):
    combination, data = args
    study = optuna.create_study(direction='maximize')
    study.optimize(func=lambda trial: profit_calculator.profit(trial, data, combination), n_trials=30)
    best_params = study.best_params
    best_value = study.best_value

    logging.info(f"Combination {combination}: Best parameters: {best_params}")
    logging.info(f"Combination {combination}: Best value: {best_value}")

    return combination, best_params, best_value

if __name__ == '__main__':
    # Load data
    data_btc = pd.read_csv("./data/btc_project_train.csv").dropna()
    data_btc = data_btc.rename(columns={data_btc.columns[0]: 'btc'})

    # Calculate technical indicators
    btc_technical_data = technical_analysis.calculate_technical_indicators(data_btc)

    # Plot all technical indicators for BTC
    technical_indicators.plot_technical_indicators(btc_technical_data, title="BTC Technical Indicators")

    # Generate all possible combinations of indicators
    indicators = ["RSI", "MACD", "Bollinger Bands", "ATR"]
    all_combinations = []

    for i in range(1, len(indicators) + 1):
        comb = combinations(indicators, i)
        all_combinations.extend(comb)

    # Prepare arguments for multiprocessing
    args = [(combination, data_btc) for combination in all_combinations]

    # Evaluate all combinations of indicators for BTC using multiple processes
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(optimize_combination, args))

    best_combination_btc = None
    best_value_btc = -float("inf")
    best_params_btc = None

    for combination, best_params, best_value in results:
        logging.info(f"Final combination {combination}: Best parameters: {best_params}")
        logging.info(f"Final combination {combination}: Best value: {best_value}")

        if best_value > best_value_btc:
            best_value_btc = best_value
            best_combination_btc = combination
            best_params_btc = best_params

    logging.info(" ")
    logging.info(" ")
    logging.info(f"The best combination of indicators for BTC is: {best_combination_btc} with a value of: {best_value_btc}")

    # Crear un diccionario con el mejor resultado
    best_outcome_btc = {
        "combination": best_combination_btc,
        "value": best_value_btc,
        "params": best_params_btc
    }

    logging.info(f"Best outcome: {best_outcome_btc}")

    # Creating a JSON to avoid unnecessary testing
    best_outcome_json = json.dumps(best_outcome_btc, indent=4)

    # Saving file
    with open("best_outcome_btc.txt", "w") as file:
        file.write(best_outcome_json)
