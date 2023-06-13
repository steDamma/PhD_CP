import hydra
from omegaconf import DictConfig

import logging
import os
import pandas as pd

from utilities import store_res
from models.shallow import ShallowClassifier

@hydra.main(config_path="../conf", config_name="main", version_base=None)
def main(config: DictConfig):
    os.chdir('..')

    logging.info('Loading dataset...\n')
    X = pd.read_csv(config.project.target.features, sep=';', decimal='.', header=0)

    clf = ShallowClassifier(config)
    
    logging.info('Training starts...\n')
    results = clf.run_classifier(X)

    logging.info('Storing Results...\n')
    store_res(config, results)

if __name__ == '__main__':
    main()

