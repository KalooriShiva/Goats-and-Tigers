# Goats & Tigers


Develop a Reinforcement Learning algorithm on strategy games.

To run,

```python
python game.py
```

## Project Structure

```plaintext
├── README.md
├── dataset                 # contains generated dataset
│   └── data.txt
├── functions.py
├── game.py                 # Handles the GUI
├── huligutta.py            # Game code
├── images
├── notebooks
│   ├── Playground.ipynb    # Experimental notebook
│   └── RL.ipynb            # Some data visualizations
├── rl_goat.py              # Reniforcement Learning for Goat aganist statistical Tiger.
├── rl_tiger.py             # Reniforcement Learning for Tiger aganist trained Goat.
```

## How to Play

Goats

* Click any empty positions to place a goat on the board
* To move goats, click the goat, then click on a valid empty position

Tigers

* To move tigers, click the tiger, then click on a valid empty position
* To capture, click on a valid empty position the tiger goes to.


## Dependencies

* networkx==2.5
* pillow==8.0.1
* numpy==1.19.2
* scipy==1.5.2

<!-- install dependencies by -->

<!-- ```bash
conda install networkx
conda install pillow
conda install numpy
conda install scipy
``` -->

