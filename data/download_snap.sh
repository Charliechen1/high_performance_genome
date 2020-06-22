rm -rf snap
mkdir snap && cd snap

wget http://snap.stanford.edu/gnn-pretrain/data/bio_dataset.zip
unzip bio_dataset.zip

wget ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_ensembl.csv.gz