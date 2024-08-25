In my experimentation, I started with simpler models that yielded poor results, with accuracies hovering around 5-6%. 
Realizing the need for more complexity, I introduced multiple convolutional layers and hidden layers, 
which significantly improved performance.

My breakthrough came when I implemented a model with 2 convolutional layers and 2 hidden layers (128 and 64 nodes), 
achieving 67.03% accuracy. I then experimented with various configurations, including a 3-convolutional layer model with
128-64 nodes, which reached 65.75% accuracy. I found that reverting to 2 convolutional layers but increasing hidden 
layer sizes to 128-128 nodes boosted accuracy to 85.67%. With this information, I further increased hidden layer sizes 
to 256-128 nodes, resulting in 83.35% accuracy. A major leap forward occurred when I simplified to 2 convolutional 
layers and a single hidden layer with 512 nodes, reaching 92.43% accuracy. I experimented with even larger hidden layer 
sizes (384, 320, 288 nodes), and observed slightly better returns with accuracies around 93-95%.

Through methodical testing, I narrowed down the optimal configuration to 2 convolutional layers for feature extraction 
and 1 hidden layer with 256 nodes. This final model achieved my best result: 
96.81% accuracy with a loss of 0.1262.