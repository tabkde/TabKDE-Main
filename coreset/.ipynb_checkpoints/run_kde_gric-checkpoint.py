import itertools
import argparse
from train_kde_main import run_kde

def grid_search(data_name, bandwidths, coreset_sizes, **kwargs):
    for coreset_size, bandwidth in itertools.product(coreset_sizes, bandwidths):
        args = argparse.Namespace(
            data_name=data_name,
            n_samples=kwargs.get('n_samples', 300000),
            bandwidth=bandwidth,
            steps=kwargs.get('steps', 10),
            T=kwargs.get('T', 1000),
            coreset_size=coreset_size,
            batch_size=kwargs.get('batch_size', 8192),
            epochs=kwargs.get('epochs', 30),
            lr=kwargs.get('lr', 1e-2),
            lambda_reg=kwargs.get('lambda_reg', 1e-2),
            scheduler=kwargs.get('scheduler', 'cosine'),
            precision=kwargs.get('precision', 'float32')
        )
        print(f"\n🚀 Running for bandwidth={bandwidth}, coreset_size={coreset_size}")
        run_kde(args)

if __name__ == "__main__":
    grid_search(
        data_name='adult_equal',
        bandwidths=[.1, .2, .3, .4, .5 , .6, .7, .8, .9 , 1.],
        coreset_sizes= [5000],
        precision='float32'
    )
