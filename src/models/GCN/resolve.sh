source activate hpg_gcn_env
pip uninstall torch-cluster torch-scatter torch-sparse torch-spline-conv
pip install torch-cluster==latest+cu101 torch-scatter==latest+cu101 torch-sparse==latest+cu101 torch-spline-conv==latest+cu101 -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.4.0.html
conda deactivate