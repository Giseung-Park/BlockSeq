
<Installation>

1. Create Anaconda Environment

conda create -n pomdp_block python=3.6

source activate pomdp_block


2. Install required packages

pip install --upgrade pip

pip install torch==1.4.0 numpy==1.18.1 gym==0.12.5 matplotlib==2.1.2

(scipy is automatically installed)




<Experiments>

Seed 0,1,2,3,4


1. Mountain Hike

python pomdp_block_main.py --env 'MountainHike' --steps=50000 --trans_seq_interval 16 --top_k 2 --seed 0


2. Pendulum - random sensor missing

python pomdp_block_main.py --env 'Pendulum-v0' --steps=50000 --trans_seq_interval 32 --top_k 2 --seed 0


3. Sequential target-reaching tasks (R_value represents the value of R)

python pomdp_block_main.py --env 'Sequential_Rchange' --R_value 15 --steps=100000 --trans_seq_interval 32 --top_k 3 --seed 0





<Plot>

After running seed 0,1,2,3,4, we can plot the graph as follows:

python plotter.py --env 'MountainHike' --steps=50000 --trans_seq_interval 16 --seed_num 5

python plotter.py --env 'Pendulum-v0' --steps=50000 --trans_seq_interval 32 --seed_num 5

python plotter.py --env 'Sequential_Rchange' --R_value 15 --steps=100000 --trans_seq_interval 32 --seed_num 5

Note that 'seed num=n' requires seed 0, 1, ..., n-1. Graphs are saved in 'data' folder.



