# eecs542_final

## Datasets

* CorrelatedMNIST (accepts arbitrary transformation)
* CorrelatedCamelyon17-WILDS 
* CorrelatedIWildcam-WILDS

To create a DataLoader for iteration, you write
```
    dataloader = DataLoader(dataset, batch_size=<batch_size>)
```
for the CorrelatedMNIST dataset, and
```
    dataloader = DataLoader(dataset, batch_size=<batch_size>, sampler=dataset.get_correlation_sampler(spurious_match_prob))
```
for the others.

For the CorrelatedMNIST dataset, the degree of spurious correlation is specified at construction time, while the CorrelatedWILDS datasets can give you samplers with arbitrary correlation between domain and label.

The DataLoaders will yield tuples of the form `(X, y, z)`, where X is an image, y is the label, and z is the domain info.

### Testing the correlation sampler

This command will check to see if Pr[Y == Z] in a batch is near (`--tol`) the expected probability specified using the `--corr` argument.
```
python datasets.py --seed [YOUR_SEED_HERE] --dataset ['mnist' or 'wilds'] --corr [YOUR_CORRELATION] --tol [YOUR_ERROR_TOLERANCE]
```

