# connectX-agent
My agent from the ongoing <a href="https://www.kaggle.com/competitions/connectx/overview">kaggle Connect X competition</a>. When active, it hovers between the top 13% and 20% of the rolling leaderboard.
I used minimax search with alpha–beta pruning, plus a pre-computed opening book (relatively shallow).
I experimented with various versions of heuristics, this one seems to work best for now.
It was a fun project; I will revisit it to enhance the algorithm and refactor the code in future.


## An example Connect X game played by my agent (blue pieces) on kaggle:

https://github.com/user-attachments/assets/cc27142d-f505-45a2-a320-e3b55b253535
