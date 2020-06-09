# high_performance_genome
## Dependencies
<p> This project uses the Anaconda platform to create custom environments. These environments will be created with all the required dependencies if they don't already exist. </p>

Anaconda can be installed at https://www.anaconda.com/products/individual or miniconda can be installed at https://docs.conda.io/en/latest/miniconda.html.

This project uses Python 3.

## Downloading Data
### Pfam Data
<p> Pfam data is downloaded from Google as both a random split and as a clustered split. To download this data, first navigate to the data directory and run the download.sh script. </p>
<pre><code> cd data </code></pre>
<pre><code> ./download.sh </code></pre>
<p> If permission is denied, this can be fixed with </p>
<pre><code> chmod +x download.sh </code></pre>

### PDB Data

## Preprocessing
### Generating Contact Maps
<p> First install the environment if not already done: </p>
<pre><code> cd src/contact_map </code></pre>
<pre><code> conda env create -f contact_map_env.yml </code></pre>
<p> Now we can launch the environment and run the contact map script: </p>
<pre><code> conda activate contact_map_env </code></pre>
<pre><code> python contactmap.py </code></pre>

## Training LM
### Configuration
The parameters and hyperparameters can be set in <pre><code> conf/model.conf </code></pre>

### Running
<p> First install the environment if not already done: </p>
<pre><code> cd src/models/LM </code></pre>
<pre><code> conda env create -f hpg_gpu_env.yml </code></pre>
<p> Now we can launch the environment and run the training script: </p>
<pre><code> conda activate hpg_gpu_env </code></pre>
<pre><code> python train.py --config "../../../config/main.conf" </code></pre>

## Training GCN
### Running
<p> First install the environment if not already done: </p>
<pre><code> cd src/models/GCN </code></pre>
<pre><code> conda env create -f hpg_gcn_env.yml </code></pre>
<p> Now we can launch the environment and run the training script: </p>
<pre><code> conda activate hpg_gcn_env </code></pre>
<pre><code> python train.py </code></pre>
